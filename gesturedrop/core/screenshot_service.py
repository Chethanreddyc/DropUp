"""
GestureDrop — Screenshot Service (Step 5A helper)
==================================================

Thin wrapper around `mss` that grabs the primary monitor and saves
the result as a PNG.  ContentManager calls this for the screenshot
fallback path.

No state — every call is stateless.  Thread-safe.

Usage
-----
    path = ScreenshotService.capture(dest_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("GestureDrop.Screenshot")


class ScreenshotCaptureError(Exception):
    """Raised when mss fails to capture or save the screenshot."""


class ScreenshotService:
    """
    Stateless screenshot utility.

    Only the primary monitor (monitor index 1 in mss) is captured.
    Designed to be imported and called directly — no instantiation required.
    """

    @staticmethod
    def capture(dest_path: Path) -> Path:
        """
        Capture the primary monitor and save it as a PNG.

        Parameters
        ----------
        dest_path : Path
            Full path (including filename) where the PNG will be written.
            The parent directory must already exist.

        Returns
        -------
        Path
            The same *dest_path*, confirming the file was written.

        Raises
        ------
        ScreenshotCaptureError
            If mss is unavailable or the capture fails for any reason.
        """
        try:
            import mss                   # type: ignore[import]
            import mss.tools             # type: ignore[import]
        except ImportError as exc:
            raise ScreenshotCaptureError(
                "mss is not installed — run: pip install mss>=9.0.1"
            ) from exc

        try:
            with mss.mss() as sct:
                # monitor[0] is the virtual "all monitors" frame;
                # monitor[1] is the primary physical monitor.
                monitor = sct.monitors[1]
                shot    = sct.grab(monitor)
                mss.tools.to_png(shot.rgb, shot.size, output=str(dest_path))

            log.debug(
                "Screenshot saved  size=%dx%d  path=%s",
                shot.size[0], shot.size[1], dest_path,
            )
            return dest_path

        except Exception as exc:
            raise ScreenshotCaptureError(
                f"Screenshot capture failed: {exc}"
            ) from exc
