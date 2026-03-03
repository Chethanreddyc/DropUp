"""
GestureDrop — Main Entry Point
================================

Boots the entire application stack:

  1. Logging         — structured, file + console
  2. DeviceIdentity  — load or generate a persistent UUID
  3. ReceiverService — always-on TCP listener (daemon thread)
  4. DiscoveryService — always-on UDP listener (daemon thread)
  5. ContentManager  — clipboard inspector / temp-file writer
  6. ToastService    — cross-platform notifications
  7. GestureController — orchestrates SEND / RECEIVE flows
  8. GestureDetector — camera loop (main thread), feeds GestureController

Run modes (choose via CLI flag)
--------------------------------
    python -m gesturedrop            # gesture-driven camera loop (default)
    python -m gesturedrop --headless # no camera, keyboard-driven demo
    python -m gesturedrop --send     # one-shot: send clipboard content now
    python -m gesturedrop --receive  # one-shot: open receive window now

Usage
-----
    cd Ecosystem
    python -m gesturedrop [--headless] [--send] [--receive] [--debug]
    python -m gesturedrop --help

Dependencies
------------
Core (always needed)
    pyperclip >= 1.8.2
    Pillow    >= 10.0.0
    mss       >= 9.0.1

Gesture detection (only for camera mode)
    opencv-python >= 4.8
    mediapipe     >= 0.10

Optional (better toasts)
    plyer or win10toast
"""

from __future__ import annotations

import argparse
import logging
import platform
import signal
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — ensure imports work whether run as a module or a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# GestureDrop imports
# ---------------------------------------------------------------------------
from gesturedrop.core.device_identity  import DeviceIdentity
from gesturedrop.core.receiver_service import ReceiverService, LISTEN_PORT
from gesturedrop.core.discovery_service import DiscoveryService, DEFAULT_TRANSFER_PORT
from gesturedrop.core.content_manager  import ContentManager
from gesturedrop.core.sender_service   import SenderService
from gesturedrop.core.gesture_controller import GestureController
from gesturedrop.ui.toast_service      import ToastService

APP_NAME    = "GestureDrop"
APP_VERSION = "0.3.0"
BANNER = f"""
+--------------------------------------+
|      GestureDrop  v{APP_VERSION}          |
|  Gesture-Driven Peer-to-Peer Share   |
+--------------------------------------+

  SEND gesture:    Open palm  ->  Fist
  RECEIVE gesture: Fist       ->  Open palm
  Quit:            Press  Q   in camera window
                   or  Ctrl-C  from terminal

"""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO

    fmt     = "%(asctime)s  [%(levelname)-8s]  %(name)-28s  %(message)s"
    datefmt = "%H:%M:%S"

    # Console handler — use errors='backslashreplace' via a custom stream
    # wrapper so Windows cp1252 terminals never raise UnicodeEncodeError.
    import codecs as _codecs

    class _SafeStream:
        """Wraps any text stream and replaces un-encodable characters."""
        def __init__(self, stream):
            self._stream  = stream
            self._enc     = getattr(stream, "encoding", "utf-8") or "utf-8"
        def write(self, msg: str) -> int:
            safe = msg.encode(self._enc, errors="replace").decode(self._enc)
            return self._stream.write(safe)
        def flush(self):
            self._stream.flush()

    console = logging.StreamHandler(_SafeStream(sys.stdout))
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # File handler — always DEBUG, always UTF-8
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "gesturedrop.log"
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_h)


log = logging.getLogger(APP_NAME)


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

class Application:
    """
    Owns the full service graph.

    All services are stored as attributes so they live as long as
    Application lives, and stop() can reach them cleanly.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args     = args
        self._running = True

        # ── 1. Identity ──────────────────────────────────────────────────
        self.identity = DeviceIdentity.load_or_create()
        log.info(
            "Identity: %s  (host=%s  os=%s)",
            self.identity, platform.node(), platform.system(),
        )

        # ── 2. Receiver Service ──────────────────────────────────────────
        save_dir = _PROJECT_ROOT / "received_files"
        self.receiver = ReceiverService(
            save_dir=save_dir,
            ready_timeout=10.0,
            on_transfer_complete=self._on_file_received,
            on_state_change=lambda old, new: log.info(
                "ReceiverState: %s -> %s", old.name, new.name
            ),
        )
        self.receiver.start()
        log.info("ReceiverService listening on port %d", LISTEN_PORT)

        # ── 3. Discovery Service ─────────────────────────────────────────
        self.discovery = DiscoveryService(
            identity=self.identity,
            receiver_svc=self.receiver,
            transfer_port=DEFAULT_TRANSFER_PORT,
            send_intent_wait=5.0,
            on_peer_found=lambda peer: log.info(
                "Peer found: %s @ %s:%d", peer.device_name, peer.host, peer.accept_port
            ),
        )
        self.discovery.start()
        log.info("DiscoveryService listening for peers")

        # ── 4. Content Manager ───────────────────────────────────────────
        self.content_manager = ContentManager(
            outbound_dir=_PROJECT_ROOT / "temp" / "outbound"
        )

        # ── 5. Toast Service ─────────────────────────────────────────────
        self.toast = ToastService(app_name=APP_NAME, duration=3)

        # ── 6. Gesture Controller ────────────────────────────────────────
        def sender_factory(on_complete=None):
            return SenderService(on_complete=on_complete)

        self.controller = GestureController(
            receiver_service=self.receiver,
            discovery_service=self.discovery,
            sender_factory=sender_factory,
            content_manager=self.content_manager,
            toast_service=self.toast,
        )

        log.info("Application stack initialised successfully")

        # ── 7. Graceful shutdown via Ctrl-C ─────────────────────────────
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_file_received(self, save_path: Path) -> None:
        size = save_path.stat().st_size if save_path.exists() else 0
        msg  = f"Received: {save_path.name} ({_fmt_size(size)})"
        log.info(msg)
        self.toast.show(f"📥 {msg}")

    def _handle_signal(self, signum, frame) -> None:
        log.info("Signal %s received — shutting down…", signum)
        self._running = False

    # ------------------------------------------------------------------
    # Run modes
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Dispatch to the appropriate run mode. Returns process exit code."""
        print(BANNER)
        log.info(
            "Starting GestureDrop  mode=%s  device=%r  version=%s",
            "headless" if self.args.headless else "camera",
            self.identity.device_name,
            APP_VERSION,
        )

        if self.args.send:
            return self._run_one_shot_send()

        if self.args.receive:
            return self._run_one_shot_receive()

        if self.args.headless:
            return self._run_headless()

        return self._run_camera()

    def _run_one_shot_send(self) -> int:
        """
        --send mode: prepare clipboard content and send immediately.
        Useful for scripting / testing without gesture hardware.
        """
        print("  [ONE-SHOT SEND]  Reading clipboard…\n")
        self.controller.on_gesture("SEND")
        # Wait long enough for the full discovery + transfer to complete
        time.sleep(12)
        return 0

    def _run_one_shot_receive(self) -> int:
        """
        --receive mode: open the receive window and wait for a file.
        """
        print("  [ONE-SHOT RECEIVE]  Opening receive window for 15s…\n")
        self.controller.on_gesture("RECEIVE")
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and self._running:
            state = self.receiver.state
            _print_status(self.identity, state.name)
            if state.name == "IDLE" and time.monotonic() > deadline - 18:
                # File was received (transitioned back to IDLE)
                break
            time.sleep(1)
        return 0

    def _run_headless(self) -> int:
        """
        --headless / keyboard mode: no camera needed.
        Useful on machines without a webcam or for demos.

        Commands
        --------
          s   → trigger SEND gesture
          r   → trigger RECEIVE gesture
          q   → quit
        """
        print("  [HEADLESS MODE]  Keyboard commands: s=SEND  r=RECEIVE  q=QUIT\n")
        self.toast.show(f"GestureDrop ready  ({self.identity.device_name})")

        while self._running:
            try:
                _print_status(self.identity, self.receiver.state.name)
                raw = input("  >> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if raw == "s":
                print("  Gesture: SEND\n")
                self.controller.on_gesture("SEND")
            elif raw == "r":
                print("  Gesture: RECEIVE\n")
                self.controller.on_gesture("RECEIVE")
            elif raw in ("q", "quit", "exit"):
                break
            elif raw == "":
                pass
            else:
                print(f"  Unknown command: {raw!r}  (use s / r / q)")

        return 0

    def _run_camera(self) -> int:
        """
        Default mode — open webcam and process gestures in real time.
        Requires: opencv-python, mediapipe.
        """
        try:
            import cv2
            from gesturedrop.core.gesture import GestureDetector
        except ImportError as exc:
            print(
                f"\n  ERROR: Camera mode requires cv2 and mediapipe.\n"
                f"  {exc}\n\n"
                f"  Install: pip install opencv-python mediapipe\n"
                f"  Or run in headless mode: python -m gesturedrop --headless\n"
            )
            return 1

        detector = GestureDetector()
        cap      = cv2.VideoCapture(0)

        if not cap.isOpened():
            print(
                "\n  ERROR: Could not open webcam (index 0).\n"
                "  Check your camera connection, or use --headless mode.\n"
            )
            return 1

        self.toast.show(
            f"GestureDrop ready  ({self.identity.device_name})"
        )
        log.info("Camera opened — entering gesture loop")

        # ── Status overlay drawing helper ────────────────────────────────
        def _draw_hud(frame, state: str) -> None:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"GestureDrop v{APP_VERSION}  |  {self.identity.device_name}  |  {state}",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1,
            )
            # Centre-zone guide box (middle 40%)
            x0, x1 = int(w * 0.30), int(w * 0.70)
            y0, y1 = int(h * 0.30), int(h * 0.70)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 220), 1)

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Camera read failed — exiting loop")
                    break

                frame = cv2.flip(frame, 1)
                state = self.receiver.state.name

                frame, action = detector.process_frame(frame)
                _draw_hud(frame, state)

                if action:
                    log.info("Gesture detected: %s", action)
                    self.controller.on_gesture(action)

                cv2.imshow("GestureDrop", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("User pressed Q — exiting camera loop")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Gracefully stop all background services."""
        log.info("Shutting down services…")
        try:
            self.discovery.stop()
        except Exception as exc:
            log.debug("discovery.stop() raised: %s", exc)
        try:
            self.receiver.stop()
        except Exception as exc:
            log.debug("receiver.stop() raised: %s", exc)
        log.info("GestureDrop stopped cleanly.")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gesturedrop",
        description="GestureDrop — gesture-driven peer-to-peer file transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run modes:
  (default)   Open webcam, detect gestures via MediaPipe
  --headless  Keyboard-driven mode (no camera required)
  --send      One-shot: immediately read clipboard and send
  --receive   One-shot: open receive window and wait

Examples:
  python -m gesturedrop
  python -m gesturedrop --headless
  python -m gesturedrop --send --debug
  python -m gesturedrop --receive
        """,
    )
    p.add_argument(
        "--headless", action="store_true",
        help="Run without webcam — use keyboard commands (s/r/q)",
    )
    p.add_argument(
        "--send", action="store_true",
        help="One-shot: read clipboard and send immediately, then exit",
    )
    p.add_argument(
        "--receive", action="store_true",
        help="One-shot: open receive window and wait for a file, then exit",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging to console",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_size(n: int) -> str:
    if n < 1024:        return f"{n} B"
    if n < 1024 ** 2:   return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _print_status(identity: DeviceIdentity, state: str) -> None:
    """Print a one-line status bar to stdout."""
    ts = time.strftime("%H:%M:%S")
    width = 60
    line = f"  [{ts}]  {identity.device_name}  |  Receiver: {state}"
    print(f"\r{line:<{width}}", end="", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    _setup_logging(debug=args.debug)

    print(f"\n  {APP_NAME} v{APP_VERSION}  —  starting on {platform.node()}")

    app = Application(args)
    try:
        return app.run()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.\n")
        return 0
    finally:
        app.stop()
        print(f"\n  {APP_NAME} exited cleanly.\n")


if __name__ == "__main__":
    sys.exit(main())
