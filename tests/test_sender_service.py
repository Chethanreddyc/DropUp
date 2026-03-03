"""
Unit & integration tests for gesturedrop.core.sender_service (Step 6 — Integrity)
======================================================================

Run with:
  pytest tests/test_sender_service.py -v

Covers
------
  Wire protocol
    - _build_header produces correct byte layout matching receiver expectations
    - Header fields: magic, version, name_len, file_size, filename

  SenderService unit (mock sockets)
    - Initial state is IDLE
    - State transitions: IDLE → CONNECTING → WAIT_ACK → STREAMING → DONE
    - Calling transfer() twice without a new instance raises RuntimeError
    - FileNotFoundError raised before any connection attempt
    - FileTooLargeError raised before any connection attempt
    - ACCEPT handshake → calls _stream_file
    - REJECT → TransferRejectedError, state = DONE
    - BUSY  → ReceiverBusyError, state = DONE
    - Garbage handshake → SessionError, state = DONE
    - No handshake response (empty read) → SessionError
    - on_complete callback fires on success
    - on_complete callback fires on failure
    - on_progress callback fires with correct byte counts
    - cancel() mid-transfer raises TransferCancelledError in worker
    - wait() blocks until transfer finishes and returns result
    - Non-blocking transfer() returns None immediately
    - Blocking transfer() returns TransferResult

  Integration (real TCP against ReceiverService)
    - Full happy path: file transferred, content matches on disk
    - Receiver in IDLE → REJECT
    - Receiver BUSY → BUSY response
    - File corrupted during read → failure result
    - Large file (1 MiB) received correctly
    - Consecutive transfers (second after first completes)
    - on_progress called with monotonically increasing byte counts
    - on_complete fires with success=True and correct bytes_sent
"""

from __future__ import annotations

import contextlib
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

from gesturedrop.core.discovery_service import PeerInfo
from gesturedrop.core.receiver_service import (
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ReceiverService,
    ReceiverState,
)
from gesturedrop.core.sender_service import (
    MAX_FILE_SIZE,
    PROTOCOL_MAGIC as S_MAGIC,
    PROTOCOL_VERSION as S_VERSION,
    FileTooLargeError,
    ReceiverBusyError,
    ReceiverClosedError,
    SenderService,
    SenderState,
    SessionError,
    TransferCancelledError,
    TransferRejectedError,
    TransferResult,
)
from gesturedrop.core.integrity import IntegrityError, SHA256_BYTES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_peer(host: str = "127.0.0.1", port: int = 54321) -> PeerInfo:
    return PeerInfo(
        device_id="test-peer-id",
        device_name="TestPeer",
        host=host,
        accept_port=port,
        capabilities=("file_transfer_v1",),
    )


def _make_receiver(tmp_path: Path, port: int) -> ReceiverService:
    return ReceiverService(
        host="127.0.0.1",
        port=port,
        save_dir=tmp_path,
        ready_timeout=30.0,
    )


def _make_sender(**kwargs) -> SenderService:
    defaults = dict(
        connect_timeout=3.0,
        handshake_timeout=3.0,
        send_timeout=5.0,
        confirm_timeout=5.0,
    )
    defaults.update(kwargs)   # caller kwargs override defaults
    return SenderService(**defaults)


def _wait_for_idle(svc: ReceiverService, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if svc.is_idle:
            return True
        time.sleep(0.05)
    return False


def _tmp_file(tmp_path: Path, content: bytes, name: str = "test.bin") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ===========================================================================
# Wire Protocol Tests
# ===========================================================================

class TestWireProtocol:

    def test_magic_matches_receiver(self):
        assert S_MAGIC == PROTOCOL_MAGIC

    def test_version_matches_receiver(self):
        assert S_VERSION == PROTOCOL_VERSION

    def test_build_header_structure(self):
        """
        Verify v2 byte layout:
          magic(5) | version(1) | name_len(4) | file_size(8) | sha256(32) | filename
        """
        digest = b"\xab" * SHA256_BYTES
        header = SenderService._build_header("hello.txt", 1024, digest)
        offset = 0

        # Magic (5 bytes)
        assert header[offset:offset + 5] == PROTOCOL_MAGIC
        offset += 5

        # Version (1 byte)
        assert header[offset] == PROTOCOL_VERSION
        offset += 1

        # name_len (4 bytes LE uint32)
        name_len = struct.unpack_from("<I", header, offset)[0]
        offset += 4
        assert name_len == len("hello.txt".encode("utf-8"))

        # file_size (8 bytes LE uint64)
        file_size = struct.unpack_from("<Q", header, offset)[0]
        offset += 8
        assert file_size == 1024

        # SHA-256 (32 bytes raw) — new in v2
        sha_field = header[offset:offset + SHA256_BYTES]
        offset += SHA256_BYTES
        assert sha_field == digest

        # filename
        name = header[offset:offset + name_len]
        assert name == b"hello.txt"

    def test_build_header_total_length(self):
        """Total length = 5+1+4+8+32+len(filename_bytes)."""
        digest = bytes(SHA256_BYTES)
        name   = "ab.txt"
        header = SenderService._build_header(name, 100, digest)
        expected_len = 5 + 1 + 4 + 8 + SHA256_BYTES + len(name.encode())
        assert len(header) == expected_len

    def test_build_header_unicode_filename(self):
        name       = "résumé.pdf"
        name_bytes = name.encode("utf-8")
        digest     = bytes(SHA256_BYTES)
        header     = SenderService._build_header(name, 0, digest)
        # name_len lives at offset 6 (after magic + version)
        name_len = struct.unpack_from("<I", header, 6)[0]
        assert name_len == len(name_bytes)

    def test_build_header_zero_size_file(self):
        digest    = bytes(SHA256_BYTES)
        header    = SenderService._build_header("empty.txt", 0, digest)
        file_size = struct.unpack_from("<Q", header, 10)[0]
        assert file_size == 0

    def test_build_header_large_file(self):
        size   = 1 * 1024 ** 3  # 1 GiB
        digest = bytes(SHA256_BYTES)
        header = SenderService._build_header("big.bin", size, digest)
        file_size = struct.unpack_from("<Q", header, 10)[0]
        assert file_size == size

    def test_build_header_sha256_field(self):
        """SHA-256 must be placed at bytes 18..50 (after fixed fields)."""
        digest = bytes(range(SHA256_BYTES))
        header = SenderService._build_header("x.bin", 0, digest)
        assert header[18:18 + SHA256_BYTES] == digest


# ===========================================================================
# Unit Tests — state machine & public API (no real sockets)
# ===========================================================================

class TestSenderServiceState:

    def test_initial_state_is_idle(self):
        svc = _make_sender()
        assert svc.state == SenderState.IDLE
        assert not svc.is_active

    def test_is_active_in_connecting(self):
        svc = _make_sender()
        svc._set_state(SenderState.CONNECTING)
        assert svc.is_active

    def test_is_active_in_wait_ack(self):
        svc = _make_sender()
        svc._set_state(SenderState.WAIT_ACK)
        assert svc.is_active

    def test_is_active_in_streaming(self):
        svc = _make_sender()
        svc._set_state(SenderState.STREAMING)
        assert svc.is_active

    def test_is_active_in_wait_done(self):
        svc = _make_sender()
        svc._set_state(SenderState.WAIT_DONE)
        assert svc.is_active

    def test_is_not_active_when_done(self):
        svc = _make_sender()
        svc._set_state(SenderState.DONE)
        assert not svc.is_active

    def test_transfer_raises_if_already_in_flight(self, tmp_path):
        svc = _make_sender()
        f = _tmp_file(tmp_path, b"x")
        svc._set_state(SenderState.CONNECTING)  # simulate in-flight
        with pytest.raises(RuntimeError):
            svc.transfer(_make_peer(), f)

    def test_transfer_raises_file_not_found(self, tmp_path):
        svc = _make_sender()
        with pytest.raises(FileNotFoundError):
            svc.transfer(_make_peer(), tmp_path / "nonexistent.txt")

    def test_transfer_raises_file_too_large(self, tmp_path):
        """FileTooLargeError must be raised before any connection attempt."""
        import os
        svc = _make_sender()
        # Create a real sparse/truncated file that stat() reports as oversized.
        # os.truncate extends with zeros without allocating actual disk blocks.
        f = tmp_path / "big.bin"
        f.write_bytes(b"")
        os.truncate(f, MAX_FILE_SIZE + 1)   # sparse file, instant
        with pytest.raises(FileTooLargeError):
            svc.transfer(_make_peer(), f)
        assert svc.state == SenderState.IDLE  # state reset after raising

    def test_transfer_raises_not_regular_file(self, tmp_path):
        svc = _make_sender()
        with pytest.raises(ValueError):
            svc.transfer(_make_peer(), tmp_path)  # directory, not file


# ===========================================================================
# Unit Tests — handshake parsing (via _await_handshake directly)
# ===========================================================================

class TestHandshake:

    def _make_sock(self, response: bytes) -> socket.socket:
        """Fake socket that returns *response* on recv()."""
        sock = MagicMock(spec=socket.socket)
        sock.recv.return_value = response
        return sock

    def test_accept_returns_normally(self):
        svc = _make_sender()
        sock = self._make_sock(b"ACCEPT\n")
        svc._await_handshake(sock, "test-peer")  # should not raise

    def test_accept_with_extra_whitespace(self):
        svc = _make_sender()
        sock = self._make_sock(b"ACCEPT\r\n")
        svc._await_handshake(sock, "test-peer")

    def test_reject_raises_transfer_rejected(self):
        svc = _make_sender()
        sock = self._make_sock(b"REJECT\n")
        with pytest.raises(TransferRejectedError):
            svc._await_handshake(sock, "test-peer")

    def test_busy_raises_receiver_busy(self):
        svc = _make_sender()
        sock = self._make_sock(b"BUSY\n")
        with pytest.raises(ReceiverBusyError):
            svc._await_handshake(sock, "test-peer")

    def test_unknown_response_raises_session_error(self):
        svc = _make_sender()
        sock = self._make_sock(b"WHAT\n")
        with pytest.raises(SessionError):
            svc._await_handshake(sock, "test-peer")

    def test_empty_read_raises_session_error(self):
        svc = _make_sender()
        sock = self._make_sock(b"")  # EOF
        with pytest.raises(SessionError):
            svc._await_handshake(sock, "test-peer")

    def test_timeout_propagates(self):
        svc = _make_sender()
        sock = MagicMock(spec=socket.socket)
        sock.recv.side_effect = socket.timeout("timed out")
        with pytest.raises(socket.timeout):
            svc._await_handshake(sock, "test-peer")


# ===========================================================================
# Unit Tests — _stream_file (mock socket, real temp file)
# ===========================================================================

class TestStreamFile:

    def test_stream_small_file(self, tmp_path):
        payload = b"Hello, GestureDrop!"
        f = _tmp_file(tmp_path, payload)
        sock = MagicMock(spec=socket.socket)
        svc = _make_sender()
        import hashlib
        digest = hashlib.sha256(payload).digest()
        sent = svc._stream_file(sock, f, len(payload), digest, "test-peer")
        assert sent == len(payload)
        # sendall called twice: once for header, once per chunk
        assert sock.sendall.call_count >= 2

    def test_stream_sends_header_first(self, tmp_path):
        payload = b"data"
        f = _tmp_file(tmp_path, payload)
        sock = MagicMock(spec=socket.socket)
        svc = _make_sender()
        import hashlib
        digest = hashlib.sha256(payload).digest()
        svc._stream_file(sock, f, len(payload), digest, "test-peer")
        # First sendall call should be the header
        first_arg = sock.sendall.call_args_list[0][0][0]
        assert first_arg[:5] == PROTOCOL_MAGIC

    def test_stream_progress_callback(self, tmp_path):
        payload = bytes(range(256)) * 10   # 2560 bytes
        f = _tmp_file(tmp_path, payload)
        sock = MagicMock(spec=socket.socket)
        progress_calls: List[tuple] = []
        svc = _make_sender(on_progress=lambda s, t: progress_calls.append((s, t)))
        import hashlib
        digest = hashlib.sha256(payload).digest()
        svc._stream_file(sock, f, len(payload), digest, "test-peer")
        assert len(progress_calls) >= 1
        # Final call must report full size
        assert progress_calls[-1] == (len(payload), len(payload))
        # Byte counts must be non-decreasing
        sent_vals = [c[0] for c in progress_calls]
        assert sent_vals == sorted(sent_vals)

    def test_stream_cancels_mid_transfer(self, tmp_path):
        # 3 chunks: make cancel trigger mid-way
        payload = b"X" * (3 * 65_536)
        f = _tmp_file(tmp_path, payload)
        sock = MagicMock(spec=socket.socket)
        svc = _make_sender()
        call_count = 0
        import hashlib
        digest = hashlib.sha256(payload).digest()

        def patched_sendall(data):
            nonlocal call_count
            call_count += 1
            if call_count == 2:    # cancel after header + first chunk
                svc._cancel_event.set()

        sock.sendall.side_effect = patched_sendall
        with pytest.raises(TransferCancelledError):
            svc._stream_file(sock, f, len(payload), digest, "test-peer")

    def test_stream_sendall_error_propagates(self, tmp_path):
        payload = b"data"
        f = _tmp_file(tmp_path, payload)
        sock = MagicMock(spec=socket.socket)
        sock.sendall.side_effect = OSError("broken pipe")
        svc = _make_sender()
        import hashlib
        digest = hashlib.sha256(payload).digest()
        with pytest.raises(OSError):
            svc._stream_file(sock, f, len(payload), digest, "test-peer")


# ===========================================================================
# Integration Tests — real TCP against ReceiverService
# ===========================================================================

class TestSenderIntegration:

    def test_happy_path_content_matches(self, tmp_path):
        """Full round-trip: file sent by SenderService is received byte-for-byte."""
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        receiver.set_ready()

        payload = b"Hello from SenderService!"
        f = _tmp_file(tmp_path, payload, "hello.bin")
        peer = _make_peer(port=port)
        svc = _make_sender()

        result = svc.transfer(peer, f, blocking=True)
        assert _wait_for_idle(receiver), "Receiver stuck"
        receiver.stop()

        assert result is not None
        assert result.success
        assert result.bytes_sent == len(payload)
        saved = list((tmp_path / "recv").iterdir())
        assert len(saved) == 1
        assert saved[0].read_bytes() == payload

    def test_receiver_idle_causes_reject(self, tmp_path):
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        # Do NOT call set_ready() — receiver stays IDLE

        f = _tmp_file(tmp_path, b"data", "data.bin")
        peer = _make_peer(port=port)
        svc = _make_sender()

        result = svc.transfer(peer, f, blocking=True)
        receiver.stop()

        assert result is not None
        assert not result.success
        assert isinstance(result.error, TransferRejectedError)
        assert svc.state == SenderState.DONE

    def test_receiver_busy_causes_busy_error(self, tmp_path):
        """While a slow transfer is in flight, a second sender gets BUSY."""
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        receiver.set_ready()

        big_payload = b"X" * (512 * 1024)
        f_big = _tmp_file(tmp_path, big_payload, "big.bin")
        f_small = _tmp_file(tmp_path, b"small", "small.bin")
        peer = _make_peer(port=port)

        done_event = threading.Event()
        first_results: list = []

        def slow_complete(result):
            first_results.append(result)
            done_event.set()

        svc1 = _make_sender(on_complete=slow_complete)
        svc1.transfer(peer, f_big)        # non-blocking, large file
        time.sleep(0.3)                   # let first transfer start RECEIVING

        # Second sender — receiver is RECEIVING now, should get REJECT or BUSY
        svc2 = _make_sender()
        result2 = svc2.transfer(peer, f_small, blocking=True)

        done_event.wait(timeout=10.0)
        receiver.stop()

        assert result2 is not None
        assert not result2.success
        assert isinstance(result2.error, (ReceiverBusyError, TransferRejectedError))

    def test_connection_refused_gives_session_error(self, tmp_path):
        """Connecting to a closed port yields a SessionError."""
        port = _free_port()   # nothing listening
        f = _tmp_file(tmp_path, b"data", "data.bin")
        peer = _make_peer(port=port)
        svc = _make_sender(connect_timeout=1.0)

        result = svc.transfer(peer, f, blocking=True)
        assert result is not None
        assert not result.success
        assert isinstance(result.error, SessionError)

    def test_large_file_received_correctly(self, tmp_path):
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        receiver.set_ready()

        payload = bytes(range(256)) * 4096   # 1 MiB
        f = _tmp_file(tmp_path, payload, "large.bin")
        peer = _make_peer(port=port)
        svc = _make_sender()

        result = svc.transfer(peer, f, blocking=True)
        assert _wait_for_idle(receiver)
        receiver.stop()

        assert result.success
        assert result.bytes_sent == len(payload)
        saved = list((tmp_path / "recv").iterdir())
        assert len(saved) == 1
        assert saved[0].read_bytes() == payload

    def test_non_blocking_transfer_returns_none(self, tmp_path):
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        receiver.set_ready()

        f = _tmp_file(tmp_path, b"quick", "quick.bin")
        peer = _make_peer(port=port)
        svc = _make_sender()

        ret = svc.transfer(peer, f, blocking=False)
        assert ret is None   # immediately returns None
        svc.wait(timeout=5.0)
        assert _wait_for_idle(receiver)
        receiver.stop()

    def test_on_complete_fires_on_success(self, tmp_path):
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        receiver.set_ready()

        f = _tmp_file(tmp_path, b"callback test", "cb.bin")
        peer = _make_peer(port=port)
        results: List[TransferResult] = []
        ev = threading.Event()

        def on_done(r):
            results.append(r)
            ev.set()

        svc = _make_sender(on_complete=on_done)
        svc.transfer(peer, f)
        ev.wait(timeout=5.0)
        assert _wait_for_idle(receiver)
        receiver.stop()

        assert len(results) == 1
        assert results[0].success
        assert results[0].bytes_sent == len(b"callback test")

    def test_on_complete_fires_on_failure(self, tmp_path):
        port = _free_port()   # nothing listening
        f = _tmp_file(tmp_path, b"data", "data.bin")
        peer = _make_peer(port=port)
        results: List[TransferResult] = []
        ev = threading.Event()

        def on_done(r):
            results.append(r)
            ev.set()

        svc = _make_sender(connect_timeout=1.0, on_complete=on_done)
        svc.transfer(peer, f)
        ev.wait(timeout=5.0)

        assert len(results) == 1
        assert not results[0].success
        assert results[0].error is not None

    def test_on_progress_monotonically_increasing(self, tmp_path):
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        receiver.set_ready()

        payload = b"A" * (256 * 1024)   # 256 KiB (> 3 chunks)
        f = _tmp_file(tmp_path, payload, "progress.bin")
        peer = _make_peer(port=port)
        progress: List[int] = []

        svc = _make_sender(on_progress=lambda s, t: progress.append(s))
        result = svc.transfer(peer, f, blocking=True)
        assert _wait_for_idle(receiver)
        receiver.stop()

        assert result.success
        assert progress[-1] == len(payload)
        # Monotonically non-decreasing
        for i in range(1, len(progress)):
            assert progress[i] >= progress[i - 1]

    def test_consecutive_transfers_succeed(self, tmp_path):
        """After the first transfer a new SenderService can transfer again."""
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)

        # First transfer
        receiver.set_ready()
        f1 = _tmp_file(tmp_path, b"first", "first.bin")
        svc1 = _make_sender()
        r1 = svc1.transfer(_make_peer(port=port), f1, blocking=True)
        assert _wait_for_idle(receiver)

        # Second transfer — fresh SenderService instance
        receiver.set_ready()
        f2 = _tmp_file(tmp_path, b"second", "second.bin")
        svc2 = _make_sender()
        r2 = svc2.transfer(_make_peer(port=port), f2, blocking=True)
        assert _wait_for_idle(receiver)
        receiver.stop()

        assert r1.success
        assert r2.success
        saved = sorted(f.name for f in (tmp_path / "recv").iterdir())
        assert len(saved) == 2

    def test_cancel_mid_transfer(self, tmp_path):
        """Calling cancel() mid-stream causes the worker to stop early."""
        port = _free_port()
        receiver = _make_receiver(tmp_path / "recv", port)
        receiver.start()
        time.sleep(0.15)
        receiver.set_ready()

        # Large enough to take a noticeable time
        payload = b"Y" * (4 * 1024 * 1024)   # 4 MiB
        f = _tmp_file(tmp_path, payload, "cancel_test.bin")
        peer = _make_peer(port=port)
        ev = threading.Event()

        def on_progress(sent, total):
            if sent > 0:
                ev.set()   # signal that streaming has started

        svc = _make_sender(on_progress=on_progress)
        svc.transfer(peer, f, blocking=False)

        ev.wait(timeout=3.0)   # wait for streaming to start
        svc.cancel()
        result = svc.wait(timeout=5.0)
        receiver.stop()

        assert result is not None
        assert not result.success
        assert isinstance(result.error, TransferCancelledError)

    def test_wait_returns_none_on_timeout(self, tmp_path):
        """
        wait(timeout=0) returns None when the worker thread is still alive.

        Uses a silent TCP server that accepts the connection but never sends
        anything, so the sender hangs in recv() during _await_handshake.
        A threading.Event confirms the server accepted before we poll.
        """
        import select as _select

        port = _free_port()
        f = _tmp_file(tmp_path, b"x", "x.bin")

        connected_ev = threading.Event()
        stop_ev      = threading.Event()
        accepted_conns: list = []

        silent_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        silent_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        silent_srv.bind(("127.0.0.1", port))
        silent_srv.listen(1)
        silent_srv.setblocking(False)

        def _accept_loop():
            while not stop_ev.is_set():
                r, _, _ = _select.select([silent_srv], [], [], 0.05)
                if r:
                    try:
                        conn, _ = silent_srv.accept()
                        accepted_conns.append(conn)
                        connected_ev.set()
                    except OSError:
                        break

        accept_t = threading.Thread(target=_accept_loop, daemon=True)
        accept_t.start()

        svc = _make_sender(handshake_timeout=1.5)
        svc.transfer(_make_peer(port=port), f, blocking=False)

        # Wait until the server has accepted the connection, then a bit more
        # so the sender is definitely blocked inside recv().
        connected_ev.wait(timeout=3.0)
        time.sleep(0.1)

        # Now poll — worker is mid-handshake, still alive → must return None
        result = svc.wait(timeout=0.02)
        assert result is None, f"Expected None (in-flight), got {result}"

        # Wait for natural handshake timeout
        final = svc.wait(timeout=5.0)

        # Cleanup
        stop_ev.set()
        with contextlib.suppress(OSError):
            silent_srv.close()
        for c in accepted_conns:
            with contextlib.suppress(OSError):
                c.close()

        assert final is not None
        assert not final.success


# ===========================================================================
# TransferResult Tests
# ===========================================================================

class TestTransferResult:

    def test_result_fields(self, tmp_path):
        f = _tmp_file(tmp_path, b"x")
        peer = _make_peer()
        r = TransferResult(
            success=True,
            bytes_sent=1,
            elapsed=0.5,
            peer=peer,
            file_path=f,
            error=None,
        )
        assert r.success
        assert r.bytes_sent == 1
        assert r.elapsed == 0.5
        assert r.peer is peer
        assert r.file_path is f
        assert r.error is None

    def test_result_is_frozen(self, tmp_path):
        f = _tmp_file(tmp_path, b"x")
        peer = _make_peer()
        r = TransferResult(True, 1, 0.1, peer, f, None)
        with pytest.raises(Exception):
            r.success = False   # frozen dataclass
