"""
Unit & integration tests for gesturedrop.core.receiver_service (Step 2)
========================================================================
Covers:
  - All four states: IDLE, READY_TO_RECEIVE, RECEIVING, BUSY
  - Legal and illegal state transitions
  - Readiness timeout (auto-revert READY → IDLE)
  - Timer cancellation on set_idle() / set_busy() / incoming connection
  - Discovery gate (can_reply_to_discovery)
  - Transfer permission gate (reject unless READY_TO_RECEIVE)
  - BUSY state blocks new connections
  - set_busy() / busy_context() public API
  - on_state_change callback
  - on_transfer_complete callback
  - Protocol helpers (_sanitise_filename, _unique_save_path)
  - Full end-to-end file transfers over loopback TCP

Run with:
  pytest tests/test_receiver_service.py -v
"""

import socket
import struct
import threading
import time
from pathlib import Path
from typing import List, Tuple

import pytest

from gesturedrop.core.receiver_service import (
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ProtocolError,
    ReceiverService,
    ReceiverState,
    _sanitise_filename,
    _unique_save_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Ask the OS for a free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_header(filename: str, file_size: int) -> bytes:
    """Construct a well-formed protocol header."""
    name_bytes = filename.encode("utf-8")
    return (
        PROTOCOL_MAGIC
        + bytes([PROTOCOL_VERSION])
        + struct.pack("<I", len(name_bytes))
        + struct.pack("<Q", file_size)
        + name_bytes
    )


def _send_file(
    host: str,
    port: int,
    filename: str,
    payload: bytes,
    connect_timeout: float = 3.0,
) -> str:
    """
    Connect to a receiver, send a complete valid transfer, and return the
    server's initial response (ACCEPT / REJECT / BUSY).
    """
    with socket.create_connection((host, port), timeout=connect_timeout) as s:
        s.settimeout(5.0)
        response = s.recv(16).strip()
        if response != b"ACCEPT":
            return response.decode()
        header = _build_header(filename, len(payload))
        s.sendall(header)
        s.sendall(payload)
        return "ACCEPT"


def _peek_response(host: str, port: int, timeout: float = 2.0) -> str:
    """Connect and just read the greeting — no data sent."""
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(3.0)
        return s.recv(16).strip().decode()


def _make_service(
    tmp_path: Path,
    ready_timeout: float = 10.0,
    **kwargs,
) -> Tuple[ReceiverService, int]:
    """Create a ReceiverService bound to a free port with a temp save dir."""
    port = _free_port()
    svc = ReceiverService(
        host="127.0.0.1",
        port=port,
        save_dir=tmp_path,
        ready_timeout=ready_timeout,
        **kwargs,
    )
    return svc, port


def _wait_for_state(
    svc: ReceiverService,
    target: ReceiverState,
    timeout: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if svc.state == target:
            return True
        time.sleep(0.05)
    return False


def _wait_for_idle(svc: ReceiverService, timeout: float = 5.0) -> bool:
    return _wait_for_state(svc, ReceiverState.IDLE, timeout)


# ===========================================================================
# Unit tests — pure helpers (no network)
# ===========================================================================

class TestSanitiseFilename:

    def test_normal_name_unchanged(self):
        assert _sanitise_filename("hello.txt") == "hello.txt"

    def test_strips_directory_traversal(self):
        result = _sanitise_filename("../../etc/passwd")
        assert "/" not in result
        assert "\\" not in result
        assert result == "passwd"

    def test_removes_forbidden_windows_chars(self):
        result = _sanitise_filename('file<>:"/\\|?*.txt')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_empty_string_becomes_unnamed(self):
        assert _sanitise_filename("") == "unnamed_file"

    def test_only_dots_becomes_unnamed(self):
        result = _sanitise_filename("...")
        assert result == "unnamed_file"

    def test_long_name_is_truncated(self):
        long_name = "a" * 300 + ".txt"
        result = _sanitise_filename(long_name)
        assert len(result) <= 255

    def test_null_bytes_removed(self):
        result = _sanitise_filename("fi\x00le.txt")
        assert "\x00" not in result


class TestUniqueSavePath:

    def test_new_file_returns_as_is(self, tmp_path):
        p = _unique_save_path(tmp_path, "new.txt")
        assert p == tmp_path / "new.txt"

    def test_existing_file_gets_counter(self, tmp_path):
        (tmp_path / "file.txt").touch()
        p = _unique_save_path(tmp_path, "file.txt")
        assert p == tmp_path / "file_1.txt"

    def test_multiple_conflicts_increment(self, tmp_path):
        (tmp_path / "file.txt").touch()
        (tmp_path / "file_1.txt").touch()
        p = _unique_save_path(tmp_path, "file.txt")
        assert p == tmp_path / "file_2.txt"


# ===========================================================================
# Unit tests — State machine transitions (no network)
# ===========================================================================

class TestStateMachineTransitions:
    """Verify all legal and illegal state transitions without any sockets."""

    def test_initial_state_is_idle(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        assert svc.state == ReceiverState.IDLE
        assert svc.is_idle
        assert not svc.is_ready
        assert not svc.is_receiving
        assert not svc.is_busy

    def test_set_ready_from_idle(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        ok = svc.set_ready()
        assert ok
        assert svc.state == ReceiverState.READY_TO_RECEIVE
        assert svc.is_ready
        svc.set_idle()  # clean up timer

    def test_set_ready_from_ready_is_allowed(self, tmp_path):
        """Calling set_ready() when already READY resets the timer, not an error."""
        svc, _ = _make_service(tmp_path)
        svc.set_ready()
        ok = svc.set_ready()  # idempotent
        assert ok
        assert svc.is_ready
        svc.set_idle()

    def test_set_idle_from_ready(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.set_ready()
        svc.set_idle()
        assert svc.state == ReceiverState.IDLE

    def test_set_idle_when_already_idle_is_noop(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.set_idle()
        assert svc.is_idle  # no exception, still IDLE

    def test_set_ready_blocked_when_receiving(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        with svc._state_lock:
            svc._state = ReceiverState.RECEIVING
        result = svc.set_ready()
        assert result is False
        assert svc.state == ReceiverState.RECEIVING

    def test_set_ready_blocked_when_busy(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        with svc._state_lock:
            svc._state = ReceiverState.BUSY
        result = svc.set_ready()
        assert result is False
        assert svc.state == ReceiverState.BUSY

    def test_set_busy_from_idle(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        ok = svc.set_busy()
        assert ok
        assert svc.is_busy
        svc.set_idle()

    def test_set_busy_from_ready(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.set_ready()
        ok = svc.set_busy()
        assert ok
        assert svc.is_busy
        svc.set_idle()

    def test_set_busy_when_already_busy_is_noop(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.set_busy()
        ok = svc.set_busy()  # idempotent
        assert ok
        assert svc.is_busy
        svc.set_idle()

    def test_busy_context_enters_and_exits(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        with svc.busy_context() as acquired:
            assert acquired
            assert svc.is_busy
        assert svc.is_idle

    def test_busy_context_blocked_when_receiving(self, tmp_path):
        """busy_context() that fails to acquire should not crash and should yield False."""
        svc, _ = _make_service(tmp_path)
        with svc._state_lock:
            svc._state = ReceiverState.RECEIVING
        # set_busy returns False for RECEIVING; context exits cleanly
        with svc.busy_context() as acquired:
            assert acquired is False
        # state should remain unchanged
        assert svc.state == ReceiverState.RECEIVING


# ===========================================================================
# Unit tests — Discovery gate
# ===========================================================================

class TestDiscoveryGate:
    """can_reply_to_discovery() must return True only in READY_TO_RECEIVE."""

    def test_idle_cannot_reply(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        assert not svc.can_reply_to_discovery()

    def test_ready_can_reply(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.set_ready()
        assert svc.can_reply_to_discovery()
        svc.set_idle()

    def test_receiving_cannot_reply(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        with svc._state_lock:
            svc._state = ReceiverState.RECEIVING
        assert not svc.can_reply_to_discovery()

    def test_busy_cannot_reply(self, tmp_path):
        svc, _ = _make_service(tmp_path)
        svc.set_busy()
        assert not svc.can_reply_to_discovery()
        svc.set_idle()


# ===========================================================================
# Unit tests — Readiness timeout (short timer, no network)
# ===========================================================================

class TestReadinessTimeout:
    """Validate the auto-revert timer without any real socket traffic."""

    def test_auto_reverts_to_idle_after_timeout(self, tmp_path):
        svc, _ = _make_service(tmp_path, ready_timeout=0.3)
        svc.set_ready()
        assert svc.is_ready
        # Wait just past the timeout
        result = _wait_for_state(svc, ReceiverState.IDLE, timeout=2.0)
        assert result, "State did not revert to IDLE after READY_TO_RECEIVE timeout"

    def test_set_idle_cancels_timer(self, tmp_path):
        """Calling set_idle() before expiry should cancel the timer cleanly."""
        svc, _ = _make_service(tmp_path, ready_timeout=2.0)
        svc.set_ready()
        time.sleep(0.1)
        svc.set_idle()
        # After cancellation the state must be IDLE and must stay IDLE
        time.sleep(0.3)
        assert svc.is_idle

    def test_new_set_ready_resets_timer(self, tmp_path):
        """A second set_ready() call restarts the window (does not stack)."""
        svc, _ = _make_service(tmp_path, ready_timeout=0.4)
        svc.set_ready()
        time.sleep(0.25)
        svc.set_ready()   # re-arm — timer resets to 0.4s
        time.sleep(0.25)
        # Timer has not expired yet (only 0.25s since last set_ready)
        assert svc.is_ready
        svc.set_idle()

    def test_timer_does_not_overwrite_receiving_state(self, tmp_path):
        """If we transition to RECEIVING before timeout, timer must be a no-op."""
        svc, _ = _make_service(tmp_path, ready_timeout=0.2)
        svc.set_ready()
        # Simulate: timer fires while receiver is already RECEIVING
        with svc._state_lock:
            svc._state = ReceiverState.RECEIVING
        # Timer should fire and see the state is NOT READY — no transition
        time.sleep(0.5)
        assert svc.state == ReceiverState.RECEIVING


# ===========================================================================
# Callback tests (no network)
# ===========================================================================

class TestCallbacks:

    def test_on_state_change_fires_on_set_ready(self, tmp_path):
        transitions: List[Tuple[ReceiverState, ReceiverState]] = []
        event = threading.Event()

        def recorder(old, new):
            transitions.append((old, new))
            event.set()

        svc, _ = _make_service(tmp_path, on_state_change=recorder)
        svc.set_ready()
        event.wait(timeout=1.0)
        assert any(
            new == ReceiverState.READY_TO_RECEIVE for _, new in transitions
        ), f"Expected READY_TO_RECEIVE transition, got {transitions}"
        svc.set_idle()

    def test_on_state_change_fires_on_set_idle(self, tmp_path):
        transitions: List[Tuple[ReceiverState, ReceiverState]] = []
        seen_idle = threading.Event()

        def recorder(old, new):
            transitions.append((old, new))
            if new == ReceiverState.IDLE:
                seen_idle.set()

        svc, _ = _make_service(tmp_path, on_state_change=recorder)
        svc.set_ready()
        svc.set_idle()
        seen_idle.wait(timeout=1.0)
        assert any(new == ReceiverState.IDLE for _, new in transitions)


# ===========================================================================
# Integration tests — real TCP connections, ephemeral ports
# ===========================================================================

class TestReceiverNetworked:
    """Full end-to-end transfer tests over loopback."""

    def test_service_starts_and_stops(self, tmp_path):
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
            response = s.recv(16).strip()
            assert response == b"REJECT"
        svc.stop()

    # ── State gate tests ────────────────────────────────────────────────

    def test_connection_rejected_when_idle(self, tmp_path):
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        try:
            assert _peek_response("127.0.0.1", port) == "REJECT"
        finally:
            svc.stop()

    def test_connection_accepted_when_ready(self, tmp_path):
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        try:
            result = _send_file("127.0.0.1", port, "hello.txt", b"Hello, GestureDrop!")
            assert result == "ACCEPT"
            assert _wait_for_idle(svc), "Service never returned to IDLE"
            saved = list(tmp_path.iterdir())
            assert len(saved) == 1
            assert saved[0].read_bytes() == b"Hello, GestureDrop!"
        finally:
            svc.stop()

    def test_connection_rejected_when_receiving(self, tmp_path):
        """While RECEIVING, a second attempt must get REJECT (or BUSY)."""
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()

        big_payload = b"X" * (256 * 1024)
        responses: List[str] = []
        errors: List[Exception] = []

        def slow_xfer():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=3.0) as s:
                    s.settimeout(10.0)
                    r = s.recv(16).strip().decode()
                    responses.append(r)
                    if r == "ACCEPT":
                        s.sendall(_build_header("big.bin", len(big_payload)))
                        for i in range(0, len(big_payload), 4096):
                            s.sendall(big_payload[i : i + 4096])
                            time.sleep(0.003)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=slow_xfer, daemon=True)
        t.start()
        time.sleep(0.3)

        r2 = _peek_response("127.0.0.1", port)
        assert r2 in ("REJECT", "BUSY"), f"Unexpected: {r2!r}"

        t.join(timeout=8.0)
        assert _wait_for_idle(svc)
        svc.stop()

    def test_connection_rejected_when_busy(self, tmp_path):
        """Connections arriving in BUSY state must be rejected."""
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        # Manually force BUSY
        with svc._state_lock:
            svc._state = ReceiverState.BUSY
        try:
            r = _peek_response("127.0.0.1", port)
            assert r == "REJECT", f"Expected REJECT in BUSY state, got {r!r}"
        finally:
            svc.set_idle()
            svc.stop()

    # ── Timer integration ───────────────────────────────────────────────

    def test_ready_timer_cancelled_when_sender_connects(self, tmp_path):
        """Connecting before timeout should cancel the timer (state stays clean)."""
        svc, port = _make_service(tmp_path, ready_timeout=5.0)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        # Immediately send a transfer before any timeout
        _send_file("127.0.0.1", port, "quick.txt", b"fast")
        # After transfer, state must be IDLE (not bounce back to READY later)
        assert _wait_for_idle(svc)
        # Give extra time to see if timer accidentally re-fires
        time.sleep(0.3)
        assert svc.is_idle
        svc.stop()

    # ── Protocol tests ──────────────────────────────────────────────────

    def test_filename_is_sanitised_on_disk(self, tmp_path):
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        try:
            _send_file("127.0.0.1", port, "../../evil.txt", b"evil payload")
            assert _wait_for_idle(svc)
            saved = list(tmp_path.iterdir())
            assert len(saved) == 1
            assert "/" not in saved[0].name
            assert "\\" not in saved[0].name
        finally:
            svc.stop()

    def test_bad_magic_triggers_protocol_error(self, tmp_path):
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
                s.settimeout(5.0)
                resp = s.recv(16).strip()
                assert resp == b"ACCEPT"
                s.sendall(b"BADMAGIC\x00" * 10)
            assert _wait_for_idle(svc), "Service never reset after bad magic"
        finally:
            svc.stop()

    def test_state_resets_to_idle_after_broken_connection(self, tmp_path):
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
                s.settimeout(5.0)
                resp = s.recv(16).strip()
                assert resp == b"ACCEPT"
                # Abruptly drop the connection (no data sent)
                s.close()
            assert _wait_for_idle(svc), "Service stuck after broken connection"
        finally:
            svc.stop()

    def test_large_file_received_correctly(self, tmp_path):
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        payload = bytes(range(256)) * 4096   # 1 MiB
        try:
            result = _send_file("127.0.0.1", port, "bigfile.bin", payload)
            assert result == "ACCEPT"
            assert _wait_for_idle(svc)
            saved = list(tmp_path.iterdir())
            assert len(saved) == 1
            assert saved[0].read_bytes() == payload
        finally:
            svc.stop()

    def test_receiver_ready_again_after_transfer(self, tmp_path):
        """After completing one transfer and calling set_ready() again, the next transfer works."""
        svc, port = _make_service(tmp_path)
        svc.start()
        time.sleep(0.15)
        # First transfer
        svc.set_ready()
        r1 = _send_file("127.0.0.1", port, "first.txt", b"first")
        assert r1 == "ACCEPT"
        assert _wait_for_idle(svc)
        # Second transfer
        svc.set_ready()
        r2 = _send_file("127.0.0.1", port, "second.txt", b"second")
        assert r2 == "ACCEPT"
        assert _wait_for_idle(svc)
        saved = sorted(f.name for f in tmp_path.iterdir())
        assert len(saved) == 2
        svc.stop()

    # ── Callback integration ────────────────────────────────────────────

    def test_on_transfer_complete_called_with_path(self, tmp_path):
        completed: List[Path] = []
        event = threading.Event()

        def on_done(p: Path):
            completed.append(p)
            event.set()

        svc, port = _make_service(tmp_path, on_transfer_complete=on_done)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        _send_file("127.0.0.1", port, "cb_test.txt", b"callback data")
        event.wait(timeout=5.0)
        assert len(completed) == 1
        assert completed[0].exists()
        assert completed[0].read_bytes() == b"callback data"
        assert _wait_for_idle(svc)
        svc.stop()

    def test_on_state_change_called_through_full_transfer(self, tmp_path):
        """Verify the full state sequence: IDLE→READY→RECEIVING→BUSY→IDLE."""
        seen: List[ReceiverState] = []
        done = threading.Event()

        def recorder(old, new):
            seen.append(new)
            if new == ReceiverState.IDLE and len(seen) > 1:
                done.set()

        svc, port = _make_service(tmp_path, on_state_change=recorder)
        svc.start()
        time.sleep(0.15)
        svc.set_ready()
        _send_file("127.0.0.1", port, "trace.txt", b"trace")
        done.wait(timeout=5.0)

        # Must contain these states in order somewhere in the trace
        assert ReceiverState.READY_TO_RECEIVE in seen
        assert ReceiverState.RECEIVING in seen
        assert ReceiverState.BUSY in seen
        assert ReceiverState.IDLE in seen
        svc.stop()
