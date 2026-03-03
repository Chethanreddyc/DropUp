"""
GestureDrop — Content Manager (Step 5A)
========================================

Responsibilities
----------------
  • Inspect the system clipboard (priority-ordered)
  • Normalise whatever is there to a single file on disk
  • Return a PreparedContent describing that file
  • Support one-shot cleanup of any temp files it created

Priority order (strict, never mixed)
--------------------------------------
  1. File path   → use as-is          (is_temp=False)
  2. Image       → PNG in temp dir    (is_temp=True)
  3. Text        → .txt in temp dir   (is_temp=True)
  4. Screenshot  → PNG fallback       (is_temp=True)

Wire contract with SenderService
---------------------------------
  ContentManager returns a PreparedContent whose .path is always a
  real file on disk.  SenderService never needs to know what came
  from the clipboard — it just sends the file.

Temp file location
------------------
  All ephemeral files land under  <project_root>/temp/outbound/
  The folder is created automatically on first use.

Dependencies
------------
  pyperclip >= 1.8.2   (cross-platform text/file-path clipboard)
  Pillow    >= 10.0.0  (image clipboard via ImageGrab)
  mss       >= 9.0.1   (screenshot fallback, via ScreenshotService)

Edge cases handled
------------------
  • Clipboard file no longer exists on disk      → skip, try next type
  • Clipboard image is None (empty grab)         → skip
  • Text is empty after stripping                → skip
  • Text > 64 KiB                               → raise TextTooLargeError
  • Screenshot capture fails                     → raise ScreenshotError (re-wrapped)
  • temp/outbound missing                        → auto-created
  • cleanup() on is_temp=False                   → no-op (never deletes originals)
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from gesturedrop.core.screenshot_service import (
    ScreenshotCaptureError,
    ScreenshotService,
)

log = logging.getLogger("GestureDrop.ContentManager")


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Maximum text size accepted from the clipboard (64 KiB)
MAX_TEXT_BYTES: int = 64 * 1024   # 64 KiB

# Default temp output directory (relative to the project root that contains
# the gesturedrop package).  Can be overridden via ContentManager(outbound_dir=…)
_DEFAULT_OUTBOUND_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "temp" / "outbound"
)


# ---------------------------------------------------------------------------
# Content-type enumeration
# ---------------------------------------------------------------------------

class ContentType(Enum):
    FILE       = auto()   # Original file from clipboard
    IMAGE      = auto()   # Bitmap/image extracted from clipboard → PNG
    TEXT       = auto()   # Text snippet saved as .txt
    SCREENSHOT = auto()   # Whole-screen fallback


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ClipboardEmptyError(Exception):
    """Clipboard has nothing useful (all priority levels exhausted)."""

class TextTooLargeError(Exception):
    """Clipboard text exceeds the MAX_TEXT_BYTES limit."""

class ScreenshotError(Exception):
    """Screenshot capture failed (wraps ScreenshotCaptureError)."""


# ---------------------------------------------------------------------------
# PreparedContent result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedContent:
    """
    Everything SenderService (and GestureController) needs to know
    about the content that is ready to transfer.

    Attributes
    ----------
    path         : Path to the file to send (always exists at creation time)
    content_type : ContentType enum member
    is_temp      : True  → ContentManager owns this file; call cleanup() after use
                   False → Original file; cleanup() is a no-op
    size         : File size in bytes at preparation time
    """
    path:         Path
    content_type: ContentType
    is_temp:      bool
    size:         int

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Human-readable one-liner for toasts / logs.

        Examples
        --------
        "report.pdf (1.2 MB)"
        "Copied image"
        "Code snippet (2.4 KB)"
        "Screenshot"
        """
        if self.content_type == ContentType.FILE:
            return f"{self.path.name} ({_fmt_size(self.size)})"
        if self.content_type == ContentType.IMAGE:
            return "Copied image"
        if self.content_type == ContentType.TEXT:
            return f"Code snippet ({_fmt_size(self.size)})"
        if self.content_type == ContentType.SCREENSHOT:
            return "Screenshot"
        return self.path.name   # fallback


# ---------------------------------------------------------------------------
# ContentManager
# ---------------------------------------------------------------------------

class ContentManager:
    """
    Inspects the clipboard and normalises its content to a file ready
    for transfer via SenderService.

    Parameters
    ----------
    outbound_dir : directory where temp files are written.
                   Defaults to <project_root>/temp/outbound/.
                   Created automatically if absent.

    Usage
    -----
    cm = ContentManager()
    content = cm.prepare_content()   # returns PreparedContent or None
    if content:
        sender.transfer(peer, content.path)
        cm.cleanup(content)          # removes temp file if is_temp=True
    """

    def __init__(self, outbound_dir: Optional[Path] = None) -> None:
        self._outbound_dir: Path = (
            Path(outbound_dir) if outbound_dir is not None else _DEFAULT_OUTBOUND_DIR
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_content(self) -> Optional[PreparedContent]:
        """
        Inspect the clipboard and return a PreparedContent.

        Priority
        --------
        1. File path in clipboard
        2. Image in clipboard
        3. Text in clipboard
        4. Screenshot fallback

        Returns
        -------
        PreparedContent on success, None if absolutely nothing is available.
        Callers should rarely see None because the screenshot fallback is
        always attempted last.

        Raises
        ------
        TextTooLargeError   — clipboard text exceeded MAX_TEXT_BYTES
        ScreenshotError     — screenshot capture failed
        """
        # ── Priority 1: File ────────────────────────────────────────────
        result = self._try_file()
        if result is not None:
            return result

        # ── Priority 2: Image ───────────────────────────────────────────
        result = self._try_image()
        if result is not None:
            return result

        # ── Priority 3: Text ────────────────────────────────────────────
        result = self._try_text()
        if result is not None:
            return result

        # ── Priority 4: Screenshot fallback ─────────────────────────────
        return self._do_screenshot()

    def cleanup(self, content: PreparedContent) -> None:
        """
        Delete *content*'s file if it is a temp file (is_temp=True).

        Never deletes original files.  Safe to call multiple times.
        """
        if not content.is_temp:
            return
        with contextlib.suppress(OSError):
            content.path.unlink(missing_ok=True)
            log.debug("Temp file deleted: %s", content.path)

    # ------------------------------------------------------------------
    # Internal — Priority implementations
    # ------------------------------------------------------------------

    def _try_file(self) -> Optional[PreparedContent]:
        """
        Check if the clipboard holds one or more file paths.
        Uses the first path that actually exists and is a regular file.
        Returns None if clipboard contains no file paths.
        """
        try:
            import pyperclip              # type: ignore[import]
        except ImportError:
            log.debug("pyperclip not installed — skipping file detection")
            return None

        try:
            raw = pyperclip.paste()
        except Exception as exc:
            log.debug("pyperclip.paste() raised %s — skipping file detection", exc)
            return None

        if not raw or not raw.strip():
            return None

        # Clipboard may hold multiple paths separated by newlines (Explorer copy)
        candidates = [line.strip().strip('"') for line in raw.splitlines()]
        candidates = [c for c in candidates if c]

        for raw_path in candidates:
            p = Path(raw_path)
            if p.exists() and p.is_file():
                log.info("Clipboard → FILE  path=%s  size=%d", p, p.stat().st_size)
                return PreparedContent(
                    path=p,
                    content_type=ContentType.FILE,
                    is_temp=False,
                    size=p.stat().st_size,
                )
            else:
                log.debug("Clipboard path does not exist or is not a file: %s", raw_path)

        return None

    def _try_image(self) -> Optional[PreparedContent]:
        """
        Check if the clipboard holds a bitmap image.
        Uses Pillow's ImageGrab.  Saves to a PNG temp file.
        Returns None if clipboard has no image.
        """
        try:
            from PIL import ImageGrab     # type: ignore[import]
        except ImportError:
            log.debug("Pillow not installed — skipping image detection")
            return None

        try:
            img = ImageGrab.grabclipboard()
        except Exception as exc:
            log.debug("ImageGrab.grabclipboard() raised %s — skipping", exc)
            return None

        if img is None:
            return None

        # ImageGrab can return a list of file paths on Windows when files are
        # copied in Explorer.  We only want an actual PIL Image object here —
        # file paths are handled by _try_file() with higher priority.
        try:
            from PIL import Image         # type: ignore[import]
            if not isinstance(img, Image.Image):
                return None
        except ImportError:
            return None

        # Convert to RGBA → RGB before saving as PNG (avoids mode conflicts)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")

        dest = self._make_temp_path(f"image_{_timestamp()}.png")
        try:
            img.save(dest, format="PNG")
        except Exception as exc:
            log.warning("Failed to save clipboard image to %s: %s", dest, exc)
            return None

        size = dest.stat().st_size
        log.info("Clipboard → IMAGE  path=%s  size=%d", dest, size)
        return PreparedContent(
            path=dest,
            content_type=ContentType.IMAGE,
            is_temp=True,
            size=size,
        )

    def _try_text(self) -> Optional[PreparedContent]:
        """
        Check if the clipboard holds plain text.
        Returns None if clipboard is empty or contains only whitespace.
        Raises TextTooLargeError if text exceeds MAX_TEXT_BYTES.
        """
        try:
            import pyperclip              # type: ignore[import]
        except ImportError:
            log.debug("pyperclip not installed — skipping text detection")
            return None

        try:
            raw = pyperclip.paste()
        except Exception as exc:
            log.debug("pyperclip.paste() raised %s — skipping text detection", exc)
            return None

        if not raw:
            return None

        text = raw.strip()
        if not text:
            return None

        encoded = text.encode("utf-8")
        if len(encoded) > MAX_TEXT_BYTES:
            raise TextTooLargeError(
                f"Clipboard text is {len(encoded):,} bytes — "
                f"exceeds the {MAX_TEXT_BYTES:,}-byte limit."
            )

        dest = self._make_temp_path(f"snippet_{_timestamp()}.txt")
        dest.write_bytes(encoded)

        size = dest.stat().st_size
        log.info("Clipboard → TEXT  path=%s  size=%d", dest, size)
        return PreparedContent(
            path=dest,
            content_type=ContentType.TEXT,
            is_temp=True,
            size=size,
        )

    def _do_screenshot(self) -> Optional[PreparedContent]:
        """
        Screenshot fallback — capture the primary monitor.
        Raises ScreenshotError if capture fails.
        Returns the PreparedContent on success.
        """
        dest = self._make_temp_path(f"screenshot_{_timestamp()}.png")
        try:
            ScreenshotService.capture(dest)
        except ScreenshotCaptureError as exc:
            raise ScreenshotError(str(exc)) from exc

        size = dest.stat().st_size
        log.info("Clipboard → SCREENSHOT  path=%s  size=%d", dest, size)
        return PreparedContent(
            path=dest,
            content_type=ContentType.SCREENSHOT,
            is_temp=True,
            size=size,
        )

    # ------------------------------------------------------------------
    # Internal — Helpers
    # ------------------------------------------------------------------

    def _make_temp_path(self, filename: str) -> Path:
        """
        Return a Path inside outbound_dir with the given filename.
        Creates outbound_dir if it does not yet exist.
        """
        self._outbound_dir.mkdir(parents=True, exist_ok=True)
        return self._outbound_dir / filename


# ---------------------------------------------------------------------------
# Private module helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """Microsecond-precision timestamp string safe for filenames."""
    return str(int(time.monotonic_ns() // 1000))


def _fmt_size(n: int) -> str:
    """Human-readable file size: B / KB / MB."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"
