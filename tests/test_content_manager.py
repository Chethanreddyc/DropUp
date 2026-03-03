"""
Unit tests for gesturedrop.core.content_manager (Step 5A)
==========================================================

Run with:
    pytest tests/test_content_manager.py -v

Coverage
--------
  PreparedContent
    - All fields stored correctly
    - summary() for each ContentType
    - Frozen dataclass raises on assignment

  ContentManager._try_file
    - Detects a real file path in clipboard
    - Single path only (MVP rule)
    - First valid path wins on multi-line clipboard
    - Non-existent path skipped, returns None
    - Empty clipboard text returns None
    - Whitespace-only clipboard returns None

  ContentManager._try_image
    - Returns PreparedContent(IMAGE) with PNG for real PIL Image
    - Returns None when ImageGrab returns None
    - Returns None when ImageGrab returns a list (file copy from Explorer)
    - PNG is written and is non-empty

  ContentManager._try_text
    - Returns PreparedContent(TEXT) for plain text
    - Written .txt content matches original clipboard text
    - Empty string → None
    - Whitespace-only string → None
    - Text at exactly MAX_TEXT_BYTES → accepted
    - Text over MAX_TEXT_BYTES → TextTooLargeError

  ContentManager._do_screenshot
    - Returns PreparedContent(SCREENSHOT) when capture succeeds
    - Raises ScreenshotError when ScreenshotService.capture raises

  ContentManager.prepare_content (priority integration)
    - FILE wins over IMAGE when both are available
    - IMAGE wins over TEXT when both are available
    - TEXT wins when no file/image
    - Falls through to screenshot when nothing else
    - Returns None if screenshot also fails (should raise ScreenshotError)

  ContentManager.cleanup
    - Deletes is_temp=True file
    - Does NOT delete is_temp=False file (original protection)
    - cleanup() on already-deleted file is idempotent (no crash)

  Helpers
    - _fmt_size: B / KB / MB formatting
    - outbound_dir is created automatically when missing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from gesturedrop.core.content_manager import (
    MAX_TEXT_BYTES,
    ClipboardEmptyError,
    ContentManager,
    ContentType,
    PreparedContent,
    ScreenshotError,
    TextTooLargeError,
    _fmt_size,
)
from gesturedrop.core.screenshot_service import ScreenshotCaptureError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_cm(tmp_path: Path) -> ContentManager:
    """Return a ContentManager pointing its outbound dir into tmp_path."""
    return ContentManager(outbound_dir=tmp_path / "outbound")


def _write_clipboard_file(tmp_path: Path, name: str, content: bytes) -> Path:
    """Create a real file and return its Path (to be put in the clipboard)."""
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# PreparedContent
# ---------------------------------------------------------------------------

class TestPreparedContent:

    def _make(self, tmp_path, content_type, is_temp=False, size=100):
        p = tmp_path / "dummy.bin"
        p.write_bytes(b"x" * size)
        return PreparedContent(
            path=p,
            content_type=content_type,
            is_temp=is_temp,
            size=size,
        )

    def test_fields_stored(self, tmp_path):
        pc = self._make(tmp_path, ContentType.FILE, is_temp=False, size=42)
        assert pc.content_type == ContentType.FILE
        assert pc.is_temp is False
        assert pc.size == 42

    def test_summary_file(self, tmp_path):
        pc = self._make(tmp_path, ContentType.FILE, size=1024)
        # size=1024 → "1.0 KB"
        summary = pc.summary()
        assert "dummy.bin" in summary
        assert "KB" in summary or "B" in summary

    def test_summary_image(self, tmp_path):
        pc = self._make(tmp_path, ContentType.IMAGE)
        assert pc.summary() == "Copied image"

    def test_summary_text(self, tmp_path):
        pc = self._make(tmp_path, ContentType.TEXT, size=2456)
        summary = pc.summary()
        assert "snippet" in summary.lower() or "KB" in summary or "B" in summary

    def test_summary_screenshot(self, tmp_path):
        pc = self._make(tmp_path, ContentType.SCREENSHOT)
        assert pc.summary() == "Screenshot"

    def test_frozen_raises_on_assignment(self, tmp_path):
        pc = self._make(tmp_path, ContentType.FILE)
        with pytest.raises(Exception):
            pc.size = 999   # frozen dataclass


# ---------------------------------------------------------------------------
# _fmt_size helper
# ---------------------------------------------------------------------------

class TestFmtSize:

    def test_bytes(self):
        assert _fmt_size(0)    == "0 B"
        assert _fmt_size(512)  == "512 B"
        assert _fmt_size(1023) == "1023 B"

    def test_kilobytes(self):
        result = _fmt_size(1024)
        assert "KB" in result
        result2 = _fmt_size(2048)
        assert "2.0 KB" in result2

    def test_megabytes(self):
        result = _fmt_size(1024 ** 2)
        assert "MB" in result
        result2 = _fmt_size(2 * 1024 ** 2)
        assert "2.0 MB" in result2


# ---------------------------------------------------------------------------
# ContentManager.outbound_dir auto-creation
# ---------------------------------------------------------------------------

class TestOutboundDirCreation:

    def test_dir_created_on_first_use(self, tmp_path):
        out_dir = tmp_path / "does" / "not" / "exist"
        assert not out_dir.exists()
        cm = ContentManager(outbound_dir=out_dir)
        # Trigger auto-creation via _make_temp_path
        path = cm._make_temp_path("test.txt")
        assert out_dir.exists()

    def test_dir_reused_if_already_exists(self, tmp_path):
        out_dir = tmp_path / "existing"
        out_dir.mkdir()
        (out_dir / "prior.txt").write_bytes(b"hello")
        cm = ContentManager(outbound_dir=out_dir)
        cm._make_temp_path("new.txt")
        # Prior file must not be deleted
        assert (out_dir / "prior.txt").exists()


# ---------------------------------------------------------------------------
# _try_file
# ---------------------------------------------------------------------------

class TestTryFile:

    def test_detects_valid_file_path(self, tmp_path):
        p = _write_clipboard_file(tmp_path, "report.pdf", b"PDF data")
        cm = _make_cm(tmp_path)
        with patch("pyperclip.paste", return_value=str(p)):
            result = cm._try_file()
        assert result is not None
        assert result.content_type == ContentType.FILE
        assert result.path == p
        assert result.is_temp is False
        assert result.size == len(b"PDF data")

    def test_first_valid_path_wins(self, tmp_path):
        """With multiple paths, the first existing file wins."""
        p1 = _write_clipboard_file(tmp_path, "first.txt", b"first")
        p2 = _write_clipboard_file(tmp_path, "second.txt", b"second")
        cm = _make_cm(tmp_path)
        clipboard_text = f"{p1}\n{p2}"
        with patch("pyperclip.paste", return_value=clipboard_text):
            result = cm._try_file()
        assert result is not None
        assert result.path == p1

    def test_nonexistent_path_skipped(self, tmp_path):
        cm = _make_cm(tmp_path)
        missing = str(tmp_path / "ghost.pdf")
        with patch("pyperclip.paste", return_value=missing):
            result = cm._try_file()
        assert result is None

    def test_empty_clipboard_returns_none(self, tmp_path):
        cm = _make_cm(tmp_path)
        with patch("pyperclip.paste", return_value=""):
            result = cm._try_file()
        assert result is None

    def test_whitespace_only_clipboard_returns_none(self, tmp_path):
        cm = _make_cm(tmp_path)
        with patch("pyperclip.paste", return_value="   \n\t  "):
            result = cm._try_file()
        assert result is None

    def test_quoted_path_is_accepted(self, tmp_path):
        """Windows copies paths with quotes: '"C:\\foo\\bar.txt"'."""
        p = _write_clipboard_file(tmp_path, "bar.txt", b"data")
        cm = _make_cm(tmp_path)
        with patch("pyperclip.paste", return_value=f'"{p}"'):
            result = cm._try_file()
        assert result is not None
        assert result.path == p

    def test_directory_path_returns_none(self, tmp_path):
        """A directory path must not be returned as a file."""
        cm = _make_cm(tmp_path)
        with patch("pyperclip.paste", return_value=str(tmp_path)):
            result = cm._try_file()
        assert result is None


# ---------------------------------------------------------------------------
# _try_image
# ---------------------------------------------------------------------------

class TestTryImage:

    def _fake_pil_image(self, width=4, height=4, mode="RGB"):
        """Create a tiny real PIL Image (no clipboard needed)."""
        from PIL import Image
        return Image.new(mode, (width, height), color=(128, 64, 32))

    def test_returns_image_content_for_pil_image(self, tmp_path):
        cm = _make_cm(tmp_path)
        fake_img = self._fake_pil_image()
        with patch(
            "PIL.ImageGrab.grabclipboard",
            return_value=fake_img,
        ):
            result = cm._try_image()
        assert result is not None
        assert result.content_type == ContentType.IMAGE
        assert result.is_temp is True
        assert result.path.suffix == ".png"
        assert result.path.exists()
        assert result.size > 0

    def test_png_file_is_non_empty(self, tmp_path):
        cm = _make_cm(tmp_path)
        fake_img = self._fake_pil_image()
        with patch("PIL.ImageGrab.grabclipboard", return_value=fake_img):
            result = cm._try_image()
        assert result.path.read_bytes()[:4] == b"\x89PNG"  # PNG magic bytes

    def test_none_clipboard_returns_none(self, tmp_path):
        cm = _make_cm(tmp_path)
        with patch("PIL.ImageGrab.grabclipboard", return_value=None):
            result = cm._try_image()
        assert result is None

    def test_list_clipboard_returns_none(self, tmp_path):
        """ImageGrab returns list of strings on Windows file copy — must be ignored."""
        cm = _make_cm(tmp_path)
        with patch(
            "PIL.ImageGrab.grabclipboard",
            return_value=[str(tmp_path / "file.txt")],
        ):
            result = cm._try_image()
        assert result is None

    def test_rgba_image_converted_without_error(self, tmp_path):
        cm = _make_cm(tmp_path)
        fake_img = self._fake_pil_image(mode="RGBA")
        with patch("PIL.ImageGrab.grabclipboard", return_value=fake_img):
            result = cm._try_image()
        assert result is not None

    def test_non_standard_mode_image(self, tmp_path):
        """LA mode (grayscale+alpha) should also be handled via convert."""
        cm = _make_cm(tmp_path)
        from PIL import Image
        img = Image.new("L", (4, 4), color=128)   # grayscale
        with patch("PIL.ImageGrab.grabclipboard", return_value=img):
            result = cm._try_image()
        assert result is not None


# ---------------------------------------------------------------------------
# _try_text
# ---------------------------------------------------------------------------

class TestTryText:

    def test_returns_text_content(self, tmp_path):
        cm = _make_cm(tmp_path)
        text = "Hello, GestureDrop!"
        with patch("pyperclip.paste", return_value=text):
            result = cm._try_text()
        assert result is not None
        assert result.content_type == ContentType.TEXT
        assert result.is_temp is True
        assert result.path.suffix == ".txt"

    def test_written_content_matches_clipboard(self, tmp_path):
        cm = _make_cm(tmp_path)
        text = "def hello():\n    print('world')"
        with patch("pyperclip.paste", return_value=text):
            result = cm._try_text()
        # strip() is applied before writing
        assert result.path.read_text(encoding="utf-8") == text.strip()

    def test_empty_string_returns_none(self, tmp_path):
        cm = _make_cm(tmp_path)
        with patch("pyperclip.paste", return_value=""):
            result = cm._try_text()
        assert result is None

    def test_whitespace_only_returns_none(self, tmp_path):
        cm = _make_cm(tmp_path)
        with patch("pyperclip.paste", return_value="   \t\n   "):
            result = cm._try_text()
        assert result is None

    def test_text_at_exact_limit_is_accepted(self, tmp_path):
        cm = _make_cm(tmp_path)
        # Exactly MAX_TEXT_BYTES bytes of ASCII (1 byte each)
        text = "A" * MAX_TEXT_BYTES
        with patch("pyperclip.paste", return_value=text):
            result = cm._try_text()
        assert result is not None

    def test_text_over_limit_raises(self, tmp_path):
        cm = _make_cm(tmp_path)
        text = "A" * (MAX_TEXT_BYTES + 1)
        with patch("pyperclip.paste", return_value=text):
            with pytest.raises(TextTooLargeError):
                cm._try_text()

    def test_unicode_text(self, tmp_path):
        cm = _make_cm(tmp_path)
        text = "日本語テスト 🎯"
        with patch("pyperclip.paste", return_value=text):
            result = cm._try_text()
        assert result is not None
        assert result.path.read_text(encoding="utf-8") == text.strip()


# ---------------------------------------------------------------------------
# _do_screenshot
# ---------------------------------------------------------------------------

class TestDoScreenshot:

    def test_returns_screenshot_content(self, tmp_path):
        cm = _make_cm(tmp_path)

        def fake_capture(dest_path: Path) -> Path:
            # Create a dummy PNG file at the destination
            dest_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            return dest_path

        with patch(
            "gesturedrop.core.content_manager.ScreenshotService.capture",
            side_effect=fake_capture,
        ):
            result = cm._do_screenshot()

        assert result is not None
        assert result.content_type == ContentType.SCREENSHOT
        assert result.is_temp is True
        assert result.path.suffix == ".png"
        assert result.size > 0

    def test_raises_screenshot_error_on_failure(self, tmp_path):
        cm = _make_cm(tmp_path)
        with patch(
            "gesturedrop.core.content_manager.ScreenshotService.capture",
            side_effect=ScreenshotCaptureError("mss failed"),
        ):
            with pytest.raises(ScreenshotError):
                cm._do_screenshot()


# ---------------------------------------------------------------------------
# prepare_content — priority integration
# ---------------------------------------------------------------------------

class TestPrepareContentPriority:
    """
    Verify strict priority ordering without touching real clipboard/system.
    We stub out the individual _try_* methods instead.
    """

    def _pc(self, tmp_path: Path, ct: ContentType, is_temp=False) -> PreparedContent:
        p = tmp_path / f"dummy_{ct.name}.bin"
        p.write_bytes(b"x")
        return PreparedContent(path=p, content_type=ct, is_temp=is_temp, size=1)

    def test_file_wins_over_everything(self, tmp_path):
        cm = _make_cm(tmp_path)
        file_content = self._pc(tmp_path, ContentType.FILE)
        with (
            patch.object(cm, "_try_file", return_value=file_content),
            patch.object(cm, "_try_image", return_value=self._pc(tmp_path, ContentType.IMAGE)),
            patch.object(cm, "_try_text",  return_value=self._pc(tmp_path, ContentType.TEXT)),
        ):
            result = cm.prepare_content()
        assert result.content_type == ContentType.FILE

    def test_image_wins_over_text_and_screenshot(self, tmp_path):
        cm = _make_cm(tmp_path)
        image_content = self._pc(tmp_path, ContentType.IMAGE, is_temp=True)
        with (
            patch.object(cm, "_try_file",  return_value=None),
            patch.object(cm, "_try_image", return_value=image_content),
            patch.object(cm, "_try_text",  return_value=self._pc(tmp_path, ContentType.TEXT)),
        ):
            result = cm.prepare_content()
        assert result.content_type == ContentType.IMAGE

    def test_text_wins_before_screenshot(self, tmp_path):
        cm = _make_cm(tmp_path)
        text_content = self._pc(tmp_path, ContentType.TEXT, is_temp=True)
        with (
            patch.object(cm, "_try_file",  return_value=None),
            patch.object(cm, "_try_image", return_value=None),
            patch.object(cm, "_try_text",  return_value=text_content),
            patch.object(cm, "_do_screenshot") as mock_ss,
        ):
            result = cm.prepare_content()
        assert result.content_type == ContentType.TEXT
        mock_ss.assert_not_called()

    def test_screenshot_fallback_used_when_all_none(self, tmp_path):
        cm = _make_cm(tmp_path)
        ss_content = self._pc(tmp_path, ContentType.SCREENSHOT, is_temp=True)
        with (
            patch.object(cm, "_try_file",    return_value=None),
            patch.object(cm, "_try_image",   return_value=None),
            patch.object(cm, "_try_text",    return_value=None),
            patch.object(cm, "_do_screenshot", return_value=ss_content),
        ):
            result = cm.prepare_content()
        assert result.content_type == ContentType.SCREENSHOT

    def test_screenshot_failure_propagates(self, tmp_path):
        cm = _make_cm(tmp_path)
        with (
            patch.object(cm, "_try_file",    return_value=None),
            patch.object(cm, "_try_image",   return_value=None),
            patch.object(cm, "_try_text",    return_value=None),
            patch.object(cm, "_do_screenshot",
                         side_effect=ScreenshotError("capture failed")),
        ):
            with pytest.raises(ScreenshotError):
                cm.prepare_content()

    def test_text_too_large_propagates(self, tmp_path):
        cm = _make_cm(tmp_path)
        with (
            patch.object(cm, "_try_file",  return_value=None),
            patch.object(cm, "_try_image", return_value=None),
            patch.object(cm, "_try_text",
                         side_effect=TextTooLargeError("too big")),
        ):
            with pytest.raises(TextTooLargeError):
                cm.prepare_content()


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class TestCleanup:

    def test_deletes_temp_file(self, tmp_path):
        out = tmp_path / "outbound"
        out.mkdir()
        p = out / "snippet_123.txt"
        p.write_bytes(b"temp content")

        cm = ContentManager(outbound_dir=out)
        pc = PreparedContent(path=p, content_type=ContentType.TEXT, is_temp=True, size=12)
        cm.cleanup(pc)
        assert not p.exists()

    def test_does_not_delete_original_file(self, tmp_path):
        p = tmp_path / "original.pdf"
        p.write_bytes(b"precious data")

        cm = _make_cm(tmp_path)
        pc = PreparedContent(path=p, content_type=ContentType.FILE, is_temp=False, size=13)
        cm.cleanup(pc)
        assert p.exists()   # must NOT be deleted

    def test_cleanup_idempotent(self, tmp_path):
        out = tmp_path / "outbound"
        out.mkdir()
        p = out / "image_123.png"
        p.write_bytes(b"fake png")

        cm = ContentManager(outbound_dir=out)
        pc = PreparedContent(path=p, content_type=ContentType.IMAGE, is_temp=True, size=8)
        cm.cleanup(pc)  # deletes file
        cm.cleanup(pc)  # second call — file is gone, must not raise
