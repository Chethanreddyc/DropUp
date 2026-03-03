"""
GestureDrop — Sender Service (Step 6: Integrity + Confirmation)
================================================================

Changes from Step 4
--------------------
  • Wire protocol bumped to version 0x02
  • SHA-256 hash (32 raw bytes) embedded in header between file_size and filename
  • After streaming all payload bytes, sender waits for receiver to reply:
        "DONE\\n" — hash matched, file saved — marks success
        "FAIL\\n" — hash mismatch / receiver error — marks failure
  • New state: WAIT_DONE  (between STREAMING and DONE)
  • New exceptions: IntegrityError, ReceiverClosedError, SessionMismatchError
  • TransferResult gains: sha256_hex, hash_verified  (additive — no breakage)
  • Rich structured completion log (throughput, hash status)

Updated Wire Protocol (v2)
---------------------------
  [5  bytes]  magic      : b"GDROP"
  [1  byte]   version    : 0x02
  [4  bytes]  name_len   : little-endian uint32 – byte length of filename
  [8  bytes]  file_size  : little-endian uint64 – total payload bytes
  [32 bytes]  sha256     : raw SHA-256 digest of the complete payload
  [name_len]  filename   : UTF-8 string
  [file_size] payload    : raw file bytes

  After payload — receiver sends back:
    "DONE\\n"  → hash matched, file saved
    "FAIL\\n"  → hash mismatch, file deleted

Threading Model — unchanged from Step 4
"""

from __future__ import annotations

import contextlib
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

from gesturedrop.core.discovery_service import PeerInfo
from gesturedrop.core.integrity import (
    SHA256_BYTES,
    IntegrityError,
    compute_file_sha256,
    hash_as_hex,
)

# ---------------------------------------------------------------------------
# Protocol constants — MUST match receiver_service.py exactly
# ---------------------------------------------------------------------------

PROTOCOL_MAGIC:   bytes = b"GDROP"
PROTOCOL_VERSION: int   = 0x02          # bumped v1 → v2 (adds SHA-256 field)

CHUNK_SIZE:       int   = 65_536        # 64 KiB
MAX_FILE_SIZE:    int   = 2 * 1024 ** 3 # 2 GiB

# Timeouts
CONNECT_TIMEOUT:   float = 10.0
HANDSHAKE_TIMEOUT: float = 10.0
SEND_TIMEOUT:      float = 30.0
CONFIRM_TIMEOUT:   float = 30.0   # max wait for DONE/FAIL after upload

log = logging.getLogger("GestureDrop.Sender")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class SessionError(Exception):
    """Base class — any unrecoverable sender-side session failure."""

class TransferRejectedError(SessionError):
    """Receiver replied REJECT (was not ready)."""

class ReceiverBusyError(SessionError):
    """Receiver replied BUSY (transfer already in progress)."""

class TransferCancelledError(SessionError):
    """Transfer was cancelled by the local application."""

class FileTooLargeError(SessionError):
    """File exceeds the protocol's maximum size limit."""

class ReceiverClosedError(SessionError):
    """Receiver closed the connection unexpectedly during confirmation."""

class SessionMismatchError(SessionError):
    """Reserved for future session-ID validation (Step 6.3)."""


# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

class SenderState(Enum):
    IDLE       = auto()   # No transfer in flight
    CONNECTING = auto()   # TCP connection attempt
    WAIT_ACK   = auto()   # Waiting for ACCEPT / REJECT / BUSY
    STREAMING  = auto()   # Sending file data
    WAIT_DONE  = auto()   # Payload sent, waiting for DONE/FAIL
    DONE       = auto()   # Terminal — success or failure


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferResult:
    """
    Returned by transfer() and delivered via the on_complete callback.

    New in Step 6
    -------------
    sha256_hex    : sender-computed SHA-256 digest (64-char hex)
    hash_verified : True if the receiver confirmed the digest matched
    """
    success:       bool
    bytes_sent:    int
    elapsed:       float
    peer:          PeerInfo
    file_path:     Path
    error:         Optional[Exception]
    sha256_hex:    str  = ""     # sender-computed digest
    hash_verified: bool = False  # True ↔ receiver sent DONE


# ---------------------------------------------------------------------------
# SenderService
# ---------------------------------------------------------------------------

class SenderService:
    """
    Short-lived, stateful TCP file-transfer client for GestureDrop.

    One instance = one logical transfer.  Do NOT reuse after completion.

    Usage (non-blocking)
    --------------------
    svc = SenderService(on_complete=lambda r: print(r.hash_verified))
    svc.transfer(peer, Path("file.zip"))   # returns immediately

    Usage (blocking — useful in tests)
    ------------------------------------
    result = svc.transfer(peer, Path("file.zip"), blocking=True)
    """

    def __init__(
        self,
        connect_timeout:   float = CONNECT_TIMEOUT,
        handshake_timeout: float = HANDSHAKE_TIMEOUT,
        send_timeout:      float = SEND_TIMEOUT,
        confirm_timeout:   float = CONFIRM_TIMEOUT,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_complete: Optional[Callable[["TransferResult"], None]] = None,
    ) -> None:
        self._connect_timeout   = connect_timeout
        self._handshake_timeout = handshake_timeout
        self._send_timeout      = send_timeout
        self._confirm_timeout   = confirm_timeout
        self._on_progress       = on_progress
        self._on_complete       = on_complete

        self._state:         SenderState              = SenderState.IDLE
        self._state_lock:    threading.Lock            = threading.Lock()
        self._cancel_event:  threading.Event           = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._result:        Optional[TransferResult]   = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        peer:      PeerInfo,
        file_path: Path,
        blocking:  bool = False,
    ) -> Optional[TransferResult]:
        """
        Begin a file transfer to *peer*.

        Parameters
        ----------
        peer      : PeerInfo from DiscoveryService
        file_path : local file to send
        blocking  : if True, wait for the worker thread and return result

        Raises (pre-flight, before any network activity)
        -------------------------------------------------
        RuntimeError      — another transfer is already in flight
        FileNotFoundError — file_path does not exist
        ValueError        — path is not a regular file
        FileTooLargeError — file exceeds MAX_FILE_SIZE
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Not a regular file: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise FileTooLargeError(
                f"File is {file_size} bytes — exceeds {MAX_FILE_SIZE}-byte limit."
            )

        with self._state_lock:
            if self._state not in (SenderState.IDLE, SenderState.DONE):
                raise RuntimeError(
                    f"SenderService.transfer() called while state is "
                    f"{self._state.name} — create a new instance."
                )
            self._state = SenderState.CONNECTING

        self._cancel_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_session,
            args=(peer, file_path, file_size),
            name=f"GD-Sender-{peer.host}:{peer.accept_port}",
            daemon=True,
        )
        self._worker_thread.start()

        if blocking:
            self._worker_thread.join()
            return self._result
        return None

    def cancel(self) -> None:
        """Request cooperative cancellation.  Safe to call from any thread."""
        self._cancel_event.set()
        log.info("Cancellation requested.")

    def wait(self, timeout: Optional[float] = None) -> Optional[TransferResult]:
        """Block until the current transfer completes or *timeout* expires."""
        if self._worker_thread is None:
            return self._result
        self._worker_thread.join(timeout=timeout)
        return self._result if not self._worker_thread.is_alive() else None

    @property
    def state(self) -> SenderState:
        with self._state_lock:
            return self._state

    @property
    def is_active(self) -> bool:
        return self.state in (
            SenderState.CONNECTING,
            SenderState.WAIT_ACK,
            SenderState.STREAMING,
            SenderState.WAIT_DONE,
        )

    @property
    def result(self) -> Optional[TransferResult]:
        return self._result

    # ------------------------------------------------------------------
    # Internal — Worker
    # ------------------------------------------------------------------

    def _run_session(
        self,
        peer:      PeerInfo,
        file_path: Path,
        file_size: int,
    ) -> None:
        target = f"{peer.host}:{peer.accept_port}"
        start  = time.monotonic()

        error:         Optional[Exception] = None
        bytes_sent:    int  = 0
        sha256_hex:    str  = ""
        hash_verified: bool = False

        log.info(
            "Session start  peer=%s  file=%s  size=%d",
            target, file_path.name, file_size,
        )

        sock: Optional[socket.socket] = None
        try:
            # ── Pre-compute SHA-256 before opening the socket ──────────
            file_digest = compute_file_sha256(file_path)
            sha256_hex  = hash_as_hex(file_digest)
            log.debug(
                "Pre-computed SHA-256: %s  file=%s",
                sha256_hex[:16] + "...", file_path.name,
            )

            # ── CONNECT ────────────────────────────────────────────────
            self._set_state(SenderState.CONNECTING)
            sock = self._connect(peer)

            # ── HANDSHAKE ──────────────────────────────────────────────
            self._set_state(SenderState.WAIT_ACK)
            self._await_handshake(sock, target)

            # ── STREAM ─────────────────────────────────────────────────
            self._set_state(SenderState.STREAMING)
            bytes_sent = self._stream_file(
                sock, file_path, file_size, file_digest, target
            )

            # ── WAIT FOR CONFIRMATION ───────────────────────────────────
            self._set_state(SenderState.WAIT_DONE)
            hash_verified = self._await_confirmation(sock, target)

        except TransferCancelledError as exc:
            error = exc
            log.info("Transfer cancelled after %d bytes  peer=%s", bytes_sent, target)
        except (TransferRejectedError, ReceiverBusyError) as exc:
            error = exc
            log.warning("Transfer refused: %s  peer=%s", exc, target)
        except IntegrityError as exc:
            error = exc
            log.error("Integrity check failed  peer=%s: %s", target, exc)
        except ReceiverClosedError as exc:
            error = exc
            log.error("Receiver closed connection  peer=%s: %s", target, exc)
        except (socket.timeout, TimeoutError) as exc:
            error = SessionError(f"Timeout: {exc}")
            log.error("Transfer timed out  peer=%s: %s", target, exc)
        except OSError as exc:
            error = SessionError(f"OS error: {exc}")
            log.error("Transfer OS error  peer=%s: %s", target, exc)
        except Exception as exc:                    # pragma: no cover
            error = SessionError(f"Unexpected: {exc}")
            log.exception("Unexpected error  peer=%s: %s", target, exc)
        finally:
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()

        elapsed = time.monotonic() - start
        success = (error is None and hash_verified)

        result = TransferResult(
            success=success,
            bytes_sent=bytes_sent,
            elapsed=elapsed,
            peer=peer,
            file_path=file_path,
            error=error,
            sha256_hex=sha256_hex,
            hash_verified=hash_verified,
        )
        self._result = result
        self._set_state(SenderState.DONE)

        # ── Structured completion log ────────────────────────────────
        if success:
            mbs = bytes_sent / elapsed / 1024 / 1024 if elapsed > 0 else 0
            log.info(
                "\n"
                "  Transfer complete:\n"
                "    File     : %s\n"
                "    Size     : %s\n"
                "    Time     : %.2fs\n"
                "    Speed    : %.2f MB/s\n"
                "    SHA-256  : %s\n"
                "    Verified : YES",
                file_path.name,
                _fmt_size(bytes_sent),
                elapsed,
                mbs,
                sha256_hex,
            )
        else:
            log.warning(
                "Session failed  peer=%s  sent=%d  elapsed=%.2fs  error=%s",
                target, bytes_sent, elapsed, error,
            )

        if self._on_complete is not None:
            try:
                self._on_complete(result)
            except Exception as cb_exc:            # pragma: no cover
                log.exception("on_complete callback raised: %s", cb_exc)

    # ------------------------------------------------------------------
    # Internal — Phase implementations
    # ------------------------------------------------------------------

    def _connect(self, peer: PeerInfo) -> socket.socket:
        try:
            log.debug(
                "Connecting to %s:%d  (timeout=%.1fs)",
                peer.host, peer.accept_port, self._connect_timeout,
            )
            sock = socket.create_connection(
                (peer.host, peer.accept_port),
                timeout=self._connect_timeout,
            )
            sock.settimeout(self._handshake_timeout)
            log.debug("Connected to %s:%d", peer.host, peer.accept_port)
            return sock
        except (ConnectionRefusedError, ConnectionResetError) as exc:
            raise SessionError(f"Connection refused: {exc}") from exc
        except socket.timeout as exc:
            raise SessionError(f"Connection timed out: {exc}") from exc
        except OSError as exc:
            raise SessionError(f"Connection failed: {exc}") from exc

    def _await_handshake(self, sock: socket.socket, target: str) -> None:
        """Read the receiver's ACCEPT / REJECT / BUSY greeting."""
        try:
            raw = sock.recv(16)
        except socket.timeout as exc:
            raise socket.timeout(
                f"No handshake response from {target} within "
                f"{self._handshake_timeout}s"
            ) from exc

        if not raw:
            raise SessionError(
                f"Receiver {target} closed connection during handshake"
            )

        response = raw.strip().upper().decode("ascii", errors="replace")
        log.debug("Handshake response from %s: %r", target, response)

        if response == "ACCEPT":
            log.info("Receiver ACCEPT from %s — beginning transfer", target)
            sock.settimeout(self._send_timeout)
            return
        if response == "REJECT":
            raise TransferRejectedError(
                f"Receiver {target} rejected the connection (not in READY state)."
            )
        if response == "BUSY":
            raise ReceiverBusyError(
                f"Receiver {target} is busy — transfer already in progress."
            )
        raise SessionError(
            f"Unexpected handshake response from {target}: {response!r}"
        )

    def _stream_file(
        self,
        sock:        socket.socket,
        file_path:   Path,
        file_size:   int,
        file_digest: bytes,
        target:      str,
    ) -> int:
        """
        Send the v2 protocol header then the file payload in chunks.

        v2 header layout
        ----------------
        magic(5) | version(1) | name_len(4 LE) | file_size(8 LE)
        | sha256(32) | filename(name_len)
        """
        header = self._build_header(file_path.name, file_size, file_digest)
        sock.sendall(header)
        log.debug(
            "Header sent  name=%r  size=%d  sha256=%s...  to=%s",
            file_path.name, file_size, file_digest.hex()[:16], target,
        )

        bytes_sent = 0
        with open(file_path, "rb") as fh:
            while bytes_sent < file_size:
                if self._cancel_event.is_set():
                    raise TransferCancelledError("Cancelled by application.")

                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    raise OSError(
                        f"File truncated at {bytes_sent}/{file_size} bytes"
                    )

                sock.sendall(chunk)
                bytes_sent += len(chunk)

                if self._on_progress is not None:
                    try:
                        self._on_progress(bytes_sent, file_size)
                    except Exception:              # pragma: no cover
                        pass

        log.debug("Payload complete  bytes=%d  to=%s", bytes_sent, target)
        return bytes_sent

    def _await_confirmation(self, sock: socket.socket, target: str) -> bool:
        """
        Wait for DONE\\n or FAIL\\n from the receiver after payload delivery.

        Returns True on DONE.
        Raises IntegrityError on FAIL.
        Raises ReceiverClosedError if connection drops.
        """
        sock.settimeout(self._confirm_timeout)
        try:
            raw = sock.recv(16)
        except socket.timeout as exc:
            raise socket.timeout(
                f"No confirmation from {target} within {self._confirm_timeout}s"
            ) from exc

        if not raw:
            raise ReceiverClosedError(
                f"Receiver {target} closed connection without confirmation."
            )

        token = raw.strip().upper()
        log.debug("Confirmation token from %s: %r", target, token)

        if token == b"DONE":
            log.info("Hash verified — receiver sent DONE  peer=%s", target)
            return True

        if token == b"FAIL":
            # Receiver detected mismatch — surface as IntegrityError
            raise IntegrityError(b"\x00" * SHA256_BYTES, b"\xff" * SHA256_BYTES)

        raise SessionError(
            f"Unexpected confirmation token from {target}: {token!r}"
        )

    # ------------------------------------------------------------------
    # Internal — Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header(filename: str, file_size: int, file_digest: bytes) -> bytes:
        """
        Serialise the v2 GestureDrop wire-protocol header.

        Layout (total = 5+1+4+8+32+name_len bytes)
        ------------------------------------------
        GDROP | 0x02 | name_len(4 LE) | file_size(8 LE) | sha256(32) | filename
        """
        name_bytes = filename.encode("utf-8")
        assert len(file_digest) == SHA256_BYTES, "digest must be 32 bytes"
        return (
            PROTOCOL_MAGIC
            + bytes([PROTOCOL_VERSION])
            + struct.pack("<I", len(name_bytes))   # name_len  (4 bytes LE)
            + struct.pack("<Q", file_size)          # file_size (8 bytes LE)
            + file_digest                           # sha256    (32 bytes)
            + name_bytes                            # filename
        )

    def _set_state(self, new_state: SenderState) -> None:
        """Thread-safe state assignment with logging."""
        with self._state_lock:
            old = self._state
            self._state = new_state
        log.debug("SenderState: %s -> %s", old.name, new_state.name)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fmt_size(n: int) -> str:
    if n < 1024:        return f"{n} B"
    if n < 1024 ** 2:   return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"
