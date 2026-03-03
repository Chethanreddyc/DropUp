"""
GestureDrop — Toast Service (Step 5B)
======================================

Minimal OS-level notification service.

API
---
    toast = ToastService()
    toast.show("✅ Transfer complete")

Priority chain
--------------
  1. plyer   — cross-platform (Windows Notification Center on Win 10/11)
  2. win10toast — Windows-only fallback (lighter dep)
  3. Console  — always-available fallback (prints + logs)

Installing a visual backend (optional)
---------------------------------------
    pip install plyer          ← recommended (cross-platform)
    pip install win10toast     ← Windows-only alternative

If neither backend is installed, messages are printed to stdout and
written to the GestureDrop logger.  The controller functionality is
completely unaffected — toasts are advisory-only.

Thread safety
-------------
  show() is safe to call from any thread.
  If a GUI toolkit requires the main thread (e.g. tkinter), the
  backend is invoked in a dedicated daemon thread automatically.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

log = logging.getLogger("GestureDrop.Toast")


class ToastService:
    """
    Cross-platform notification service for GestureDrop.

    Parameters
    ----------
    app_name : display name shown in the notification header
    duration : seconds the notification stays visible (where supported)
    """

    def __init__(
        self,
        app_name: str = "GestureDrop",
        duration: int = 3,
    ) -> None:
        self._app_name = app_name
        self._duration = duration
        self._backend  = self._detect_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, message: str) -> None:
        """
        Display a notification.

        Logs at INFO level regardless of backend availability.
        Fires in a daemon thread so the caller is never blocked.

        Parameters
        ----------
        message : the text to display (emoji supported on most platforms)
        """
        log.info("TOAST ▶ %s", message)
        threading.Thread(
            target=self._fire,
            args=(message,),
            daemon=True,
            name="GD-Toast",
        ).start()

    # ------------------------------------------------------------------
    # Internal — backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        """
        Return the name of the best available notification backend.

        Priority: 'plyer' > 'win10toast' > 'console'
        """
        try:
            import plyer.notification  # type: ignore[import]
            log.debug("Toast backend: plyer")
            return "plyer"
        except ImportError:
            pass

        try:
            import win10toast  # type: ignore[import]
            log.debug("Toast backend: win10toast")
            return "win10toast"
        except ImportError:
            pass

        log.debug("Toast backend: console (install 'plyer' for OS notifications)")
        return "console"

    # ------------------------------------------------------------------
    # Internal — fire notification via selected backend
    # ------------------------------------------------------------------

    def _fire(self, message: str) -> None:
        """
        Actually display the notification (called from daemon thread).
        Falls back gracefully if the primary backend fails.
        """
        if self._backend == "plyer":
            if self._try_plyer(message):
                return

        if self._backend in ("plyer", "win10toast"):
            if self._try_win10toast(message):
                return

        # Console fallback — always succeeds
        self._console(message)

    def _try_plyer(self, message: str) -> bool:
        try:
            from plyer import notification  # type: ignore[import]
            notification.notify(
                app_name=self._app_name,
                title=self._app_name,
                message=message,
                timeout=self._duration,
            )
            return True
        except Exception as exc:
            log.debug("plyer toast failed (%s) — trying next backend", exc)
            return False

    def _try_win10toast(self, message: str) -> bool:
        try:
            from win10toast import ToastNotifier  # type: ignore[import]
            notifier = ToastNotifier()
            notifier.show_toast(
                self._app_name,
                message,
                duration=self._duration,
                threaded=True,
            )
            return True
        except Exception as exc:
            log.debug("win10toast failed (%s) — falling back to console", exc)
            return False

    def _console(self, message: str) -> None:
        """Silent-print fallback (always works, never raises)."""
        try:
            print(f"\n  [{self._app_name}] {message}\n", flush=True)
        except Exception:
            pass   # last resort — swallow silently
