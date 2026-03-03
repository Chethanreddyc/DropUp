"""
Unit & integration tests for gesturedrop.core.integrity (Step 6)
=================================================================

Run with:
    pytest tests/test_integrity.py -v

Covers
------
  compute_file_sha256
    - Returns 32 raw bytes
    - Matches hashlib.sha256 reference computation
    - Works for empty file
    - Works for multi-chunk file (> 64 KiB)
    - OS error on missing file

  compute_stream_sha256
    - Returns 32 raw bytes
    - Matches reference

  hash_as_hex
    - Returns 64-char lowercase hex string
    - All-zeros digest → 64 zeros
    - Roundtrip: hex → back

  verify_hash
    - Equal digests → no exception
    - Different digests → raises IntegrityError
    - IntegrityError carries readable hex strings in .expected / .actual
    - IntegrityError message contains both hex strings

  IntegrityError
    - Inherits from Exception
    - .expected/.actual are hex strings
    - str() is human-readable (contains "FAILED")

  Protocol field integration
    - SHA256_BYTES constant is exactly 32
    - hashlib.sha256().digest_size agrees with SHA256_BYTES
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gesturedrop.core.integrity import (
    SHA256_BYTES,
    IntegrityError,
    compute_file_sha256,
    compute_stream_sha256,
    hash_as_hex,
    verify_hash,
)


# ---------------------------------------------------------------------------
# SHA256_BYTES constant
# ---------------------------------------------------------------------------

class TestSha256BytesConstant:

    def test_sha256_bytes_is_32(self):
        assert SHA256_BYTES == 32

    def test_matches_hashlib_digest_size(self):
        assert hashlib.sha256().digest_size == SHA256_BYTES


# ---------------------------------------------------------------------------
# compute_file_sha256
# ---------------------------------------------------------------------------

class TestComputeFileSha256:

    def test_returns_32_bytes(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello world")
        digest = compute_file_sha256(f)
        assert isinstance(digest, bytes)
        assert len(digest) == SHA256_BYTES

    def test_matches_reference(self, tmp_path):
        data = b"GestureDrop test payload"
        f = tmp_path / "ref.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).digest()
        assert compute_file_sha256(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").digest()
        assert compute_file_sha256(f) == expected

    def test_multi_chunk_file(self, tmp_path):
        """File larger than the internal 64 KiB chunk size must hash correctly."""
        # 200 KiB — forces multiple internal reads
        data = bytes(range(256)) * 800   # 204,800 bytes
        f = tmp_path / "big.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).digest()
        assert compute_file_sha256(f) == expected

    def test_different_content_gives_different_digest(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert compute_file_sha256(f1) != compute_file_sha256(f2)

    def test_missing_file_raises_os_error(self, tmp_path):
        with pytest.raises(OSError):
            compute_file_sha256(tmp_path / "does_not_exist.bin")

    def test_deterministic(self, tmp_path):
        """Same file → same digest on repeated calls."""
        f = tmp_path / "det.bin"
        f.write_bytes(b"deterministic")
        d1 = compute_file_sha256(f)
        d2 = compute_file_sha256(f)
        assert d1 == d2


# ---------------------------------------------------------------------------
# compute_stream_sha256
# ---------------------------------------------------------------------------

class TestComputeStreamSha256:

    def test_returns_32_bytes(self):
        assert len(compute_stream_sha256(b"")) == SHA256_BYTES

    def test_matches_reference(self):
        data = b"stream test"
        assert compute_stream_sha256(data) == hashlib.sha256(data).digest()

    def test_empty_bytes(self):
        assert compute_stream_sha256(b"") == hashlib.sha256(b"").digest()


# ---------------------------------------------------------------------------
# hash_as_hex
# ---------------------------------------------------------------------------

class TestHashAsHex:

    def test_returns_64_char_string(self):
        raw = bytes(SHA256_BYTES)
        h   = hash_as_hex(raw)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_all_zeros(self):
        assert hash_as_hex(bytes(SHA256_BYTES)) == "0" * 64

    def test_all_ones(self):
        assert hash_as_hex(b"\xff" * SHA256_BYTES) == "ff" * SHA256_BYTES

    def test_lowercase(self):
        raw = bytes(range(SHA256_BYTES))
        assert hash_as_hex(raw) == hash_as_hex(raw).lower()

    def test_roundtrip(self):
        """hex → bytes → hex must be identical."""
        raw = hashlib.sha256(b"roundtrip").digest()
        assert bytes.fromhex(hash_as_hex(raw)) == raw


# ---------------------------------------------------------------------------
# verify_hash
# ---------------------------------------------------------------------------

class TestVerifyHash:

    def test_equal_digests_no_exception(self):
        d = hashlib.sha256(b"ok").digest()
        verify_hash(d, d)   # must not raise

    def test_equal_all_zeros_no_exception(self):
        d = bytes(SHA256_BYTES)
        verify_hash(d, d)

    def test_differing_digests_raise_integrity_error(self):
        d1 = hashlib.sha256(b"aaa").digest()
        d2 = hashlib.sha256(b"bbb").digest()
        with pytest.raises(IntegrityError):
            verify_hash(d1, d2)

    def test_one_byte_difference_raises(self):
        base = bytearray(hashlib.sha256(b"base").digest())
        alt  = bytearray(base)
        alt[0] ^= 0x01
        with pytest.raises(IntegrityError):
            verify_hash(bytes(base), bytes(alt))


# ---------------------------------------------------------------------------
# IntegrityError
# ---------------------------------------------------------------------------

class TestIntegrityError:

    def _make(self) -> IntegrityError:
        expected = hashlib.sha256(b"expected").digest()
        actual   = hashlib.sha256(b"actual"  ).digest()
        return IntegrityError(expected, actual)

    def test_is_exception(self):
        assert isinstance(self._make(), Exception)

    def test_expected_is_hex_string(self):
        exc = self._make()
        assert isinstance(exc.expected, str)
        assert len(exc.expected) == 64

    def test_actual_is_hex_string(self):
        exc = self._make()
        assert isinstance(exc.actual, str)
        assert len(exc.actual) == 64

    def test_expected_and_actual_differ(self):
        exc = self._make()
        assert exc.expected != exc.actual

    def test_str_contains_failed(self):
        msg = str(self._make())
        assert "FAILED" in msg

    def test_str_contains_both_hashes(self):
        exc = self._make()
        msg = str(exc)
        assert exc.expected in msg
        assert exc.actual   in msg

    def test_raised_by_verify_hash(self):
        d1 = hashlib.sha256(b"x").digest()
        d2 = hashlib.sha256(b"y").digest()
        with pytest.raises(IntegrityError) as exc_info:
            verify_hash(d1, d2)
        err = exc_info.value
        assert err.expected == d1.hex()
        assert err.actual   == d2.hex()


# ---------------------------------------------------------------------------
# End-to-end: compute → embed → receive → verify (simulates wire)
# ---------------------------------------------------------------------------

class TestIntegrityPipeline:
    """
    Simulate the full send-receive-verify pipeline without real sockets.

    Sender side  : compute digest → embed in header
    Receiver side: accumulate running hash → compare via verify_hash
    """

    def _simulate_send(self, data: bytes) -> tuple:
        """Returns (sender_digest, raw_data)."""
        return hashlib.sha256(data).digest(), data

    def _simulate_receive(self, data: bytes) -> bytes:
        """Simulates streaming receive with running hash."""
        h = hashlib.sha256()
        chunk_size = 65_536
        for i in range(0, len(data), chunk_size):
            h.update(data[i:i + chunk_size])
        return h.digest()

    def test_intact_transfer_passes(self):
        data             = b"GestureDrop transfer payload " * 1000
        sender_digest, _ = self._simulate_send(data)
        receiver_digest  = self._simulate_receive(data)
        verify_hash(sender_digest, receiver_digest)   # must not raise

    def test_corrupted_transfer_fails(self):
        data             = b"original payload " * 1000
        sender_digest, _ = self._simulate_send(data)
        # Simulate single-byte corruption
        corrupted        = bytearray(data)
        corrupted[42]   ^= 0xFF
        receiver_digest  = self._simulate_receive(bytes(corrupted))
        with pytest.raises(IntegrityError):
            verify_hash(sender_digest, receiver_digest)

    def test_truncated_transfer_fails(self):
        data             = b"truncated" * 500
        sender_digest, _ = self._simulate_send(data)
        receiver_digest  = self._simulate_receive(data[:len(data)//2])
        with pytest.raises(IntegrityError):
            verify_hash(sender_digest, receiver_digest)

    def test_empty_file_passes(self):
        data             = b""
        sender_digest, _ = self._simulate_send(data)
        receiver_digest  = self._simulate_receive(data)
        verify_hash(sender_digest, receiver_digest)
