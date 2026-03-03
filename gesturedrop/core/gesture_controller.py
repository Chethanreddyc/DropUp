"""
GestureDrop — Gesture Controller (Step 5B)
===========================================

Responsibilities
----------------
  • Bridge between GestureDetector (camera loop) and the rest of the system
  • Enforce per-action cooldowns at the controller level (belt-and-braces)
  • Run the full SEND flow in a background worker thread
  • Run the lightweight RECEIVE flow synchronously on the calling thread
  • Show toast messages for every meaningful event
  • Clean up ContentManager temp files after a successful transfer

What it does NOT do
-------------------
  • Open sockets
  • Read the clipboard directly
  • Handle wire protocol
  • Manage threads beyond spawning a single worker per SEND

SEND Flow (strict order)
-------------------------
  1. Enforce cooldown (discard if within SEND_COOLDOWN of last send)
  2. content = ContentManager.prepare_content()
  3. None → toast "Nothing to send", abort
  4. Toast "📁/🖼/📝/📸 Sending <summary>"
  5. peers = DiscoveryService.broadcast_send_intent()
  6. peer = DiscoveryService.select_peer(peers)
  7. No peer → toast "❌ No receiver found", cleanup, abort
  8. sender = sender_factory(on_complete=<callback>)
  9. sender.transfer(peer, content.path)       ← non-blocking
 10. on_complete callback:
        if success and is_temp → cleanup
        toast success / failure

RECEIVE Flow
-------------
  1. receiver_service.set_ready()
  2. Toast "📥 Ready to receive…"

Threading model
---------------
  Camera thread  →  on_gesture(action)
                       │
                       ├─ "RECEIVE" → synchronous (just sets state, very fast)
                       └─ "SEND"    → spawns GD-SendFlow daemon thread

Dependencies (all injected — no global singletons)
---------------------------------------------------
  receiver_service  : ReceiverService
  discovery_service : DiscoveryService
  sender_factory    : Callable[[on_complete], SenderService]
                      e.g. lambda on_complete=None: SenderService(on_complete=on_complete)
  content_manager   : ContentManager
  toast_service     : ToastService (or any object with .show(msg: str))
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from gesturedrop.core.content_manager import (
    ContentManager,
    ContentType,
    PreparedContent,
    ScreenshotError,
    TextTooLargeError,
)
from gesturedrop.core.discovery_service import DiscoveryService

log = logging.getLogger("GestureDrop.GestureController")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEND_COOLDOWN:    float = 2.0   # seconds
RECEIVE_COOLDOWN: float = 2.0   # seconds

# Content-type to emoji mapping for toast messages
_CONTENT_EMOJI = {
    ContentType.FILE:       "📁",
    ContentType.IMAGE:      "🖼",
    ContentType.TEXT:       "📝",
    ContentType.SCREENSHOT: "📸",
}


# ---------------------------------------------------------------------------
# GestureController
# ---------------------------------------------------------------------------

class GestureController:
    """
    Orchestrates the full SEND / RECEIVE flows triggered by gestures.

    Parameters
    ----------
    receiver_service  : ReceiverService — controls ready/idle state
    discovery_service : DiscoveryService — broadcasts SEND_INTENT
    sender_factory    : callable(on_complete=None) → SenderService
                        Creates a fresh SenderService per transfer.
                        Example:
                          lambda on_complete=None: SenderService(on_complete=on_complete)
    content_manager   : ContentManager — prepares clipboard content
    toast_service     : object with .show(message: str) — UI notifications
    """

    def __init__(
        self,
        receiver_service,
        discovery_service:  DiscoveryService,
        sender_factory:     Callable,
        content_manager:    ContentManager,
        toast_service,
    ) -> None:
        self._receiver   = receiver_service
        self._discovery  = discovery_service
        self._sender_fac = sender_factory
        self._content    = content_manager
        self._toast      = toast_service

        # Belt-and-braces cooldowns at controller level
        # (GestureDetector already enforces its own, but controller-level
        #  cooldowns protect against spurious on_gesture() calls from tests
        #  or alternative input sources.)
        self._last_send_time:    float = 0.0
        self._last_receive_time: float = 0.0
        self._cooldown_lock:     threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API — called by the camera loop
    # ------------------------------------------------------------------

    def on_gesture(self, action: str) -> None:
        """
        Entry point for detected gestures.

        Parameters
        ----------
        action : 'SEND' or 'RECEIVE'

        Thread-safety: safe to call from any thread (camera loop, test, UI).
        SEND is dispatched to a daemon worker thread and returns immediately.
        RECEIVE is handled synchronously — it is very fast (just a flag flip).
        """
        if action == "SEND":
            self._dispatch_send()
        elif action == "RECEIVE":
            self._dispatch_receive()
        else:
            log.debug("on_gesture: unknown action %r — ignored", action)

    # ------------------------------------------------------------------
    # RECEIVE flow — lightweight, synchronous
    # ------------------------------------------------------------------

    def _dispatch_receive(self) -> None:
        import time as _time
        with self._cooldown_lock:
            now = _time.monotonic()
            if now - self._last_receive_time < RECEIVE_COOLDOWN:
                log.debug("RECEIVE within cooldown — ignored")
                return
            self._last_receive_time = now

        ok = self._receiver.set_ready()
        if ok:
            log.info("RECEIVE gesture → receiver READY_TO_RECEIVE")
            self._toast.show("📥 Ready to receive…  (window open)")
        else:
            log.warning("RECEIVE gesture — failed to enter READY_TO_RECEIVE (state=%s)",
                        self._receiver.state.name)
            self._toast.show("⚠ Cannot receive right now")

    # ------------------------------------------------------------------
    # SEND flow — dispatches to worker thread
    # ------------------------------------------------------------------

    def _dispatch_send(self) -> None:
        import time as _time
        with self._cooldown_lock:
            now = _time.monotonic()
            if now - self._last_send_time < SEND_COOLDOWN:
                log.debug("SEND within cooldown — ignored")
                return
            self._last_send_time = now

        t = threading.Thread(
            target=self._handle_send,
            name="GD-SendFlow",
            daemon=True,
        )
        t.start()

    def _handle_send(self) -> None:
        """
        Full SEND workflow — runs entirely inside GD-SendFlow thread.
        Never blocks the camera loop.
        """
        # ── Step 2: Prepare content ────────────────────────────────────
        content: Optional[PreparedContent] = None
        try:
            content = self._content.prepare_content()

        except TextTooLargeError as exc:
            log.warning("SEND aborted — text too large: %s", exc)
            self._toast.show("⚠ Text too large to send")
            return

        except ScreenshotError as exc:
            log.warning("SEND aborted — screenshot failed: %s", exc)
            self._toast.show("⚠ Screenshot capture failed")
            return

        except Exception as exc:
            log.exception("SEND aborted — unexpected error in prepare_content: %s", exc)
            self._toast.show("⚠ Could not read clipboard")
            return

        # ── Step 3: Nothing usable ─────────────────────────────────────
        if content is None:
            log.info("SEND aborted — clipboard empty / nothing usable")
            self._toast.show("Nothing to send")
            return

        # ── Step 4: Announce intent ────────────────────────────────────
        emoji   = _CONTENT_EMOJI.get(content.content_type, "📦")
        summary = content.summary()
        self._toast.show(f"{emoji} Sending {summary}")
        log.info("SEND started  type=%s  summary=%s", content.content_type.name, summary)

        # ── Step 5: Discover peers ─────────────────────────────────────
        try:
            peers = self._discovery.broadcast_send_intent()
        except Exception as exc:
            log.error("SEND aborted — discovery raised: %s", exc)
            self._toast.show("⚠ Discovery error")
            self._safe_cleanup(content)
            return

        # ── Step 6: Select peer ────────────────────────────────────────
        peer = DiscoveryService.select_peer(peers)

        # ── Step 7: No peer ────────────────────────────────────────────
        if peer is None:
            log.info("SEND aborted — no peers replied")
            self._toast.show("❌ No receiver found")
            self._safe_cleanup(content)
            return

        log.info(
            "SEND: peer selected  name=%s  host=%s  port=%d",
            peer.device_name, peer.host, peer.accept_port,
        )

        # ── Step 8 & 9: Transfer + completion callback ─────────────────
        def _on_complete(result) -> None:
            if result.success:
                # Cleanup temp file on success
                self._safe_cleanup(content)
                self._toast.show(
                    f"✅ Transfer complete — {result.bytes_sent:,} bytes "
                    f"to {peer.device_name} ({result.elapsed:.1f}s)"
                )
                log.info(
                    "Transfer success  peer=%s  bytes=%d  elapsed=%.2fs",
                    peer.device_name, result.bytes_sent, result.elapsed,
                )
            else:
                err_msg = str(result.error) if result.error else "unknown error"
                self._toast.show(f"⚠ Transfer failed — {err_msg}")
                log.warning(
                    "Transfer failed  peer=%s  error=%s",
                    peer.device_name, result.error,
                )

        try:
            sender = self._sender_fac(on_complete=_on_complete)
            sender.transfer(peer, content.path)   # non-blocking — returns immediately
        except Exception as exc:
            log.exception("SEND aborted — sender raised before transfer: %s", exc)
            self._toast.show(f"⚠ Could not start transfer: {exc}")
            self._safe_cleanup(content)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_cleanup(self, content: Optional[PreparedContent]) -> None:
        """Delete temp file silently; never raises."""
        if content is None or not content.is_temp:
            return
        try:
            self._content.cleanup(content)
        except Exception as exc:                      # pragma: no cover
            log.debug("Temp cleanup silently failed: %s", exc)
