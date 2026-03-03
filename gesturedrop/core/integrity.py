"""
GestureDrop — Integrity Utilities (Step 6)
==========================================

Shared hash primitives used by both SenderService and ReceiverService.

Provides
--------
  compute_file_sha256(path)     → bytes (32 raw bytes)
  hash_as_hex(raw_hash)         → str   (64-char lowercase hex)
  verify_hash(expected, actual) → bool
  IntegrityError                → raised on mismatch

Design
------
  Uses only hashlib from the standard library.
  Reads in CHUNK_SIZE pieces to keep RAM usage constant for large files.
  The 32 raw bytes are embedded directly in the wire header —
  no hex encoding is needed on the wire (saves 32 bytes vs hex string).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger("GestureDrop.Integrity")

# Must match CHUNK_SIZE in sender_service.py / receiver_service.py
_CHUNK_SIZE: int = 65_536     # 64 KiB

# SHA-256 digest is always exactly 32 bytes
SHA256_BYTES: int = 32


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class IntegrityError(Exception):
    """
    Raised when the SHA-256 digest received over the wire does not match
    the digest computed from the received file bytes.

    Attributes
    ----------
    expected : 64-char lowercase hex string (sender's digest)
    actual   : 64-char lowercase hex string (receiver's digest)
    """

    def __init__(self, expected: bytes, actual: bytes) -> None:
        self.expected = expected.hex()
        self.actual   = actual.hex()
        super().__init__(
            f"Integrity check FAILED\n"
            f"  expected : {self.expected}\n"
            f"  actual   : {self.actual}"
        )


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------

def compute_file_sha256(path: Path) -> bytes:
    """
    Compute the SHA-256 digest of a file and return 32 raw bytes.

    Parameters
    ----------
    path : Path to the file (must exist and be readable)

    Returns
    -------
    bytes — 32-byte raw digest

    Raises
    ------
    OSError — if the file cannot be read
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    digest = h.digest()
    log.debug("SHA-256(%s) = %s", path.name, digest.hex())
    return digest


def compute_stream_sha256(data: bytes) -> bytes:
    """
    Compute the SHA-256 digest of an in-memory bytes object.
    Convenience wrapper for test code and small payloads.

    Returns 32 raw bytes.
    """
    return hashlib.sha256(data).digest()


def hash_as_hex(raw: bytes) -> str:
    """Convert 32 raw digest bytes to a 64-character lowercase hex string."""
    return raw.hex()


def verify_hash(expected: bytes, actual: bytes) -> None:
    """
    Compare two 32-byte SHA-256 digests.

    Raises IntegrityError if they differ.
    Uses a constant-time comparison to be safe against timing attacks
    (even though this is not a security-critical path in the current design).
    """
    import hmac
    if not hmac.compare_digest(expected, actual):
        raise IntegrityError(expected, actual)
