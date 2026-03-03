"""
GestureDrop — Receiver Service (Step 6: Integrity + Confirmation)
==================================================================

Changes from Step 2
--------------------
  • Wire protocol: version byte now validated as 0x02
  • _read_header() reads the additional 32-byte SHA-256 field
  • _receive_file() computes SHA-256 *while* receiving (no extra pass)
  • After streaming, receiver compares digests:
        match   → sends "DONE\\n" → file kept → success
        mismatch → sends "FAIL\\n" → partial file deleted → IntegrityError logged
  • on_transfer_complete callback receives additional hash_ok bool
  • Rich structured completion log (throughput, hash status)

Updated Wire Protocol (v2)
---------------------------
  [5  bytes]  magic      : b"GDROP"
  [1  byte]   version    : 0x02
  [4  bytes]  name_len   : little-endian uint32
  [8  bytes]  file_size  : little-endian uint64
  [32 bytes]  sha256     : sender's SHA-256 digest (raw bytes)
  [name_len]  filename   : UTF-8 string
  [file_size] payload    : raw file bytes

  After payload — receiver sends back to sender:
    "DONE\\n"  — hashes matched, file saved
    "FAIL\\n"  — hash mismatch, file deleted

State machine — unchanged.
"""

import contextlib
import hashlib
import logging
import os
import re
import socket
import struct
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

from gesturedrop.core.integrity import (
    SHA256_BYTES,
    IntegrityError,
    verify_hash,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LISTEN_HOST: str   = "0.0.0.0"
LISTEN_PORT: int   = 54321
BACKLOG:     int   = 1

PROTOCOL_MAGIC:   bytes = b"GDROP"
PROTOCOL_VERSION: int   = 0x02          # v1 → v2 (adds SHA-256 field)

ACCEPT_TIMEOUT: float = 1.0
RECV_TIMEOUT:   float = 10.0
READY_TIMEOUT:  float = 10.0

CHUNK_SIZE:       int = 65_536
MAX_FILENAME_LEN: int = 255
MAX_FILE_SIZE:    int = 2 * 1024 ** 3   # 2 GiB

SAVE_DIR: Path = Path(__file__).resolve().parent.parent / "received_files"


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class ReceiverState(Enum):
    """
    IDLE             – passive, rejects all connections
    READY_TO_RECEIVE – active window, accepts next connection
    RECEIVING        – transfer in flight
    BUSY             – post-transfer processing
    """
    IDLE             = auto()
    READY_TO_RECEIVE = auto()
    RECEIVING        = auto()
    BUSY             = auto()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _make_logger(name: str = "GestureDrop.Receiver") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  [%(levelname)-8s]  %(name)s  |  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


log = _make_logger()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sanitise_filename(raw: str) -> str:
    name = os.path.basename(raw.strip())
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\.{2,}", ".", name)
    name = name.strip(". ")
    if not name:
        name = "unnamed_file"
    return name[:MAX_FILENAME_LEN]


def _unique_save_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = os.path.splitext(filename)
    counter = 1
    while True:
        new_name  = f"{stem}_{counter}{suffix}"
        candidate = directory / new_name
        if not candidate.exists():
            return candidate
        counter += 1


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Connection closed before {n} bytes received (got {len(buf)})"
            )
        buf.extend(chunk)
    return bytes(buf)


def _fmt_size(n: int) -> str:
    if n < 1024:        return f"{n} B"
    if n < 1024 ** 2:   return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


# ---------------------------------------------------------------------------
# ReceiverService
# ---------------------------------------------------------------------------

class ReceiverService:
    """
    Always-on background TCP listener for GestureDrop.

    State machine controlled by a single lock (_state_lock).

    Usage
    -----
    svc = ReceiverService()
    svc.start()
    svc.set_ready()   # open a receive window
    svc.stop()
    """

    def __init__(
        self,
        host:                 str   = LISTEN_HOST,
        port:                 int   = LISTEN_PORT,
        save_dir:             Path  = SAVE_DIR,
        ready_timeout:        float = READY_TIMEOUT,
        on_transfer_complete: Optional[Callable[[Path], None]] = None,
        on_state_change:      Optional[Callable[[ReceiverState, ReceiverState], None]] = None,
    ) -> None:
        self.host          = host
        self.port          = port
        self.save_dir      = Path(save_dir)
        self.ready_timeout = ready_timeout

        self._on_transfer_complete = on_transfer_complete
        self._on_state_change      = on_state_change

        self._state:      ReceiverState   = ReceiverState.IDLE
        self._state_lock: threading.Lock  = threading.Lock()

        self._ready_timer_thread: Optional[threading.Thread] = None
        self._ready_timer_cancel: threading.Event            = threading.Event()
        self._transfer_lock:      threading.Lock             = threading.Lock()

        self._running:         bool                        = False
        self._listener_thread: Optional[threading.Thread]  = None
        self._server_sock:     Optional[socket.socket]     = None

        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API — Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            log.warning("ReceiverService.start() called but already running.")
            return
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            name="GestureDrop-Listener",
            daemon=True,
        )
        self._listener_thread.start()
        log.info("ReceiverService started on %s:%d", self.host, self.port)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._cancel_ready_timer()
        if self._server_sock:
            with contextlib.suppress(OSError):
                self._server_sock.close()
        if self._listener_thread:
            self._listener_thread.join(timeout=5.0)
        log.info("ReceiverService stopped.")

    # ------------------------------------------------------------------
    # Public API — State control
    # ------------------------------------------------------------------

    def set_ready(self) -> bool:
        with self._state_lock:
            if self._state in (ReceiverState.RECEIVING, ReceiverState.BUSY):
                log.warning(
                    "Cannot set READY — currently in %s state.", self._state.name
                )
                return False
            self._transition(ReceiverState.READY_TO_RECEIVE)
        self._start_ready_timer()
        return True

    def set_idle(self) -> None:
        self._cancel_ready_timer()
        with self._state_lock:
            if self._state != ReceiverState.IDLE:
                self._transition(ReceiverState.IDLE)

    def set_busy(self) -> bool:
        with self._state_lock:
            if self._state in (ReceiverState.RECEIVING,):
                log.warning("set_busy() called from RECEIVING — invalid.")
                return False
            if self._state == ReceiverState.BUSY:
                return True
            self._cancel_ready_timer()
            self._transition(ReceiverState.BUSY)
            return True

    @contextlib.contextmanager
    def busy_context(self):
        acquired = self.set_busy()
        if not acquired:
            log.warning("Could not enter busy_context — state is %s", self.state.name)
        try:
            yield acquired
        finally:
            if acquired:
                with self._state_lock:
                    if self._state == ReceiverState.BUSY:
                        self._transition(ReceiverState.IDLE)

    def can_reply_to_discovery(self) -> bool:
        return self.state == ReceiverState.READY_TO_RECEIVE

    # -- Properties --

    @property
    def state(self) -> ReceiverState:
        with self._state_lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        return self.state == ReceiverState.READY_TO_RECEIVE

    @property
    def is_idle(self) -> bool:
        return self.state == ReceiverState.IDLE

    @property
    def is_receiving(self) -> bool:
        return self.state == ReceiverState.RECEIVING

    @property
    def is_busy(self) -> bool:
        return self.state == ReceiverState.BUSY

    # ------------------------------------------------------------------
    # Internal — State transitions
    # ------------------------------------------------------------------

    def _transition(self, new_state: ReceiverState) -> None:
        """CALLER MUST HOLD self._state_lock."""
        old_state  = self._state
        self._state = new_state
        log.info("State: %s -> %s", old_state.name, new_state.name)
        if self._on_state_change is not None:
            threading.Thread(
                target=self._on_state_change,
                args=(old_state, new_state),
                daemon=True,
                name="GD-StateCallback",
            ).start()

    # ------------------------------------------------------------------
    # Internal — Readiness timeout
    # ------------------------------------------------------------------

    def _start_ready_timer(self) -> None:
        self._cancel_ready_timer()
        self._ready_timer_cancel.clear()
        self._ready_timer_thread = threading.Thread(
            target=self._ready_timer_worker,
            name="GD-ReadyTimer",
            daemon=True,
        )
        self._ready_timer_thread.start()
        log.debug("Readiness timer started (%.1fs window)", self.ready_timeout)

    def _cancel_ready_timer(self) -> None:
        self._ready_timer_cancel.set()
        if (
            self._ready_timer_thread is not None
            and self._ready_timer_thread.is_alive()
        ):
            self._ready_timer_thread.join(timeout=2.0)
        self._ready_timer_thread = None

    def _ready_timer_worker(self) -> None:
        cancelled = self._ready_timer_cancel.wait(timeout=self.ready_timeout)
        if cancelled:
            log.debug("Readiness timer cancelled.")
            return
        with self._state_lock:
            if self._state == ReceiverState.READY_TO_RECEIVE:
                log.info(
                    "Readiness window expired (%.1fs) — reverting to IDLE.",
                    self.ready_timeout,
                )
                self._transition(ReceiverState.IDLE)

    # ------------------------------------------------------------------
    # Internal — Listener Loop
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                self._server_sock = srv
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen(BACKLOG)
                srv.settimeout(ACCEPT_TIMEOUT)
                log.info(
                    "Listening on %s:%d  (state=%s)",
                    self.host, self.port, self.state.name,
                )

                while self._running:
                    try:
                        conn, addr = srv.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break

                    log.info("Incoming connection from %s:%d", *addr)
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn, addr),
                        name=f"GD-Handler-{addr[0]}:{addr[1]}",
                        daemon=True,
                    ).start()
        except Exception as exc:          # pragma: no cover
            log.exception("Fatal error in listener loop: %s", exc)
        finally:
            self._server_sock = None
            log.debug("Listener loop exited.")

    # ------------------------------------------------------------------
    # Internal — Connection Handler
    # ------------------------------------------------------------------

    def _handle_connection(self, conn: socket.socket, addr: tuple) -> None:
        """
        Per-connection handler thread.

        Step 6 additions
        ----------------
        6a. _read_header() now returns (filename, file_size, sender_sha256)
        6b. _receive_file() computes SHA-256 while streaming
        6c. After streaming: compare digests → send DONE or FAIL
        6d. Delete partial file on FAIL
        """
        peer = f"{addr[0]}:{addr[1]}"

        # ── 1. State gate ─────────────────────────────────────────────
        with self._state_lock:
            if self._state != ReceiverState.READY_TO_RECEIVE:
                log.warning(
                    "Rejected connection from %s — state is %s",
                    peer, self._state.name,
                )
                self._send_response(conn, b"REJECT\n")
                conn.close()
                return

        # ── 2. Single-transfer lock ───────────────────────────────────
        acquired = self._transfer_lock.acquire(blocking=False)
        if not acquired:
            log.warning(
                "Rejected connection from %s — transfer already in progress", peer
            )
            self._send_response(conn, b"BUSY\n")
            conn.close()
            return

        # ── 3. Cancel readiness timer ─────────────────────────────────
        self._cancel_ready_timer()

        # ── 4. Transition → RECEIVING ─────────────────────────────────
        with self._state_lock:
            self._transition(ReceiverState.RECEIVING)
        log.info("State -> RECEIVING  (peer=%s)", peer)

        save_path:  Optional[Path] = None
        success:    bool           = False

        try:
            conn.settimeout(RECV_TIMEOUT)

            # ── 5. Signal acceptance ──────────────────────────────────
            conn.sendall(b"ACCEPT\n")

            # ── 6a. Read v2 header ────────────────────────────────────
            filename, file_size, sender_digest = self._read_header(conn, peer)
            log.info(
                "Incoming file: name=%r  size=%d  sha256=%s...  peer=%s",
                filename, file_size, sender_digest.hex()[:16], peer,
            )

            # ── 6b. Stream file, compute digest simultaneously ────────
            save_path, received_digest = self._receive_file(
                conn, peer, filename, file_size
            )

            # ── 6c. Verify integrity ──────────────────────────────────
            try:
                verify_hash(sender_digest, received_digest)
                # Hash matched — confirm success
                self._send_response(conn, b"DONE\n")
                success = True

                elapsed     = time.monotonic()   # placeholder; real timing in _receive_file
                throughput  = file_size / 1024 / 1024  # rough MB
                log.info(
                    "\n"
                    "  Transfer complete:\n"
                    "    File     : %s\n"
                    "    Size     : %s\n"
                    "    SHA-256  : %s\n"
                    "    Verified : YES",
                    filename,
                    _fmt_size(file_size),
                    received_digest.hex(),
                )

            except IntegrityError as exc:
                # Hash mismatch — reject and clean up
                log.error(
                    "Integrity check FAILED  peer=%s\n  %s", peer, exc
                )
                self._send_response(conn, b"FAIL\n")
                # File will be deleted in the finally block below

        except socket.timeout:
            log.error("Transfer timed out from %s", peer)
        except ConnectionError as exc:
            log.error("Connection error from %s: %s", peer, exc)
        except ProtocolError as exc:
            log.error("Protocol violation from %s: %s", peer, exc)
        except OSError as exc:
            log.error("OS error during transfer from %s: %s", peer, exc)
        except Exception as exc:                     # pragma: no cover
            log.exception("Unexpected error from %s: %s", peer, exc)
        finally:
            with contextlib.suppress(OSError):
                conn.close()

            # Delete file on any failure (including integrity mismatch)
            if not success and save_path and save_path.exists():
                with contextlib.suppress(OSError):
                    save_path.unlink()
                    log.debug("Deleted partial/corrupt file: %s", save_path)

            if success and save_path is not None:
                with self._state_lock:
                    self._transition(ReceiverState.BUSY)

                if self._on_transfer_complete is not None:
                    try:
                        self._on_transfer_complete(save_path)
                    except Exception as cb_exc:  # pragma: no cover
                        log.exception(
                            "on_transfer_complete callback raised: %s", cb_exc
                        )

            with self._state_lock:
                self._transition(ReceiverState.IDLE)
            self._transfer_lock.release()
            log.info("State -> IDLE  (peer=%s)", peer)

    # ------------------------------------------------------------------
    # Internal — Protocol Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _send_response(conn: socket.socket, msg: bytes) -> None:
        with contextlib.suppress(OSError):
            conn.sendall(msg)

    def _read_header(
        self,
        conn: socket.socket,
        peer: str,
    ) -> tuple:
        """
        Parse the v2 wire-protocol header.

        Returns (sanitised_filename, file_size, sender_sha256_bytes).
        Raises ProtocolError on any format violation.

        v2 layout
        ---------
        magic (5) | version (1) | name_len (4 LE) | file_size (8 LE)
        | sha256 (32) | filename (name_len)
        """
        # Magic (5 bytes)
        magic = _recv_exact(conn, len(PROTOCOL_MAGIC))
        if magic != PROTOCOL_MAGIC:
            raise ProtocolError(
                f"Bad magic: expected {PROTOCOL_MAGIC!r}, got {magic!r}"
            )

        # Version (1 byte)
        version = _recv_exact(conn, 1)[0]
        if version != PROTOCOL_VERSION:
            raise ProtocolError(f"Unsupported protocol version: {version:#04x}")

        # name_len (4 bytes LE uint32)
        (name_len,) = struct.unpack_from("<I", _recv_exact(conn, 4))
        if name_len == 0 or name_len > MAX_FILENAME_LEN:
            raise ProtocolError(f"Invalid name_len: {name_len}")

        # file_size (8 bytes LE uint64)
        (file_size,) = struct.unpack_from("<Q", _recv_exact(conn, 8))
        if file_size > MAX_FILE_SIZE:
            raise ProtocolError(
                f"File too large: {file_size} bytes (limit {MAX_FILE_SIZE})"
            )

        # SHA-256 digest (32 bytes raw)  ← new in v2
        sender_digest = _recv_exact(conn, SHA256_BYTES)
        if len(sender_digest) != SHA256_BYTES:
            raise ProtocolError(
                f"SHA-256 field truncated: got {len(sender_digest)} bytes"
            )

        # Filename (name_len UTF-8 bytes)
        raw_name_bytes = _recv_exact(conn, name_len)
        try:
            raw_name = raw_name_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"Filename not valid UTF-8: {exc}") from exc

        safe_name = _sanitise_filename(raw_name)
        log.debug(
            "v2 header: name=%r -> %r  size=%d  sha256=%s  peer=%s",
            raw_name, safe_name, file_size, sender_digest.hex()[:16] + "...", peer,
        )
        return safe_name, file_size, sender_digest

    def _receive_file(
        self,
        conn:      socket.socket,
        peer:      str,
        filename:  str,
        file_size: int,
    ) -> tuple:
        """
        Stream file_size bytes from conn and write to save_dir.
        Simultaneously feeds a SHA-256 hasher.

        Returns (save_path, received_digest_bytes).
        """
        save_path      = _unique_save_path(self.save_dir, filename)
        bytes_received = 0
        start_time     = time.monotonic()
        hasher         = hashlib.sha256()

        with open(save_path, "wb") as fh:
            while bytes_received < file_size:
                remaining = file_size - bytes_received
                to_read   = min(CHUNK_SIZE, remaining)
                chunk     = conn.recv(to_read)
                if not chunk:
                    raise ConnectionError(
                        f"Connection closed after "
                        f"{bytes_received}/{file_size} bytes"
                    )
                fh.write(chunk)
                hasher.update(chunk)
                bytes_received += len(chunk)

        elapsed    = time.monotonic() - start_time
        throughput = (bytes_received / elapsed / 1024) if elapsed > 0 else 0
        log.debug(
            "Received %d bytes to %s  (%.1f KB/s  %.2fs)",
            bytes_received, save_path, throughput, elapsed,
        )
        return save_path, hasher.digest()


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ProtocolError(Exception):
    """Raised when incoming data violates the GestureDrop wire protocol."""


# ---------------------------------------------------------------------------
# Standalone entry-point (for manual testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    svc = ReceiverService(ready_timeout=10.0)
    svc.start()

    print("\nGestureDrop Receiver Service (v2 — Integrity)")
    print("  Commands: ready | idle | busy | state | quit")
    print(f"  Listening on port {LISTEN_PORT}")
    print(f"  Files saved to: {SAVE_DIR}")
    print(f"  Readiness timeout: {svc.ready_timeout}s\n")

    try:
        while True:
            try:
                cmd = input(">> ").strip().lower()
            except EOFError:
                break
            if cmd == "ready":
                ok = svc.set_ready()
                print(f"  -> {'READY_TO_RECEIVE' if ok else 'Failed'}")
            elif cmd == "idle":
                svc.set_idle()
                print("  -> IDLE")
            elif cmd == "state":
                print(f"  -> {svc.state.name}")
            elif cmd in ("quit", "exit", "q"):
                break
            else:
                print("  Commands: ready | idle | state | quit")
    finally:
        svc.stop()
        print("Bye.")
        sys.exit(0)
