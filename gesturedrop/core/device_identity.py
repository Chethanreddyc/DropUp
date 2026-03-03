"""
GestureDrop — Device Identity
==============================
Every device participating in GestureDrop has a stable, unique identity that
is generated once and persisted to disk.  Identity is used by the Discovery
layer to:

  • Avoid self-connection (incoming.device_id != my_device_id)
  • Identify peers in log messages
  • Carry app-version and capability information in discovery packets

Identity file location
  <project-root>/gesturedrop/device_identity.json
  (created automatically on first run)
"""

from __future__ import annotations

import json
import logging
import platform
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

log = logging.getLogger("GestureDrop.Identity")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_VERSION: str = "0.3.0"              # bumped at each step

# Capabilities advertised in every discovery packet (extensible list)
DEFAULT_CAPABILITIES: List[str] = ["file_transfer_v1"]

# Default storage location — sits next to the package root
_DEFAULT_IDENTITY_PATH: Path = (
    Path(__file__).resolve().parent.parent / "device_identity.json"
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeviceIdentity:
    """
    Stable, serialisable identity for one GestureDrop installation.

    Fields
    ------
    device_id   : str   — UUID4, generated once, never changes
    device_name : str   — human-readable hostname (editable)
    app_version : str   — semver string, updated with code
    capabilities: list  — feature tags for future negotiation
    """
    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_name: str = field(default_factory=platform.node)
    app_version: str = APP_VERSION
    capabilities: List[str] = field(default_factory=lambda: list(DEFAULT_CAPABILITIES))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (all str/list fields)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DeviceIdentity":
        return cls(
            device_id=str(data["device_id"]),
            device_name=str(data.get("device_name", platform.node())),
            app_version=str(data.get("app_version", APP_VERSION)),
            capabilities=list(data.get("capabilities", DEFAULT_CAPABILITIES)),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = _DEFAULT_IDENTITY_PATH) -> None:
        """Persist identity to *path* as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        log.debug("Identity saved → %s", path)

    @classmethod
    def load_or_create(
        cls,
        path: Path = _DEFAULT_IDENTITY_PATH,
    ) -> "DeviceIdentity":
        """
        Load identity from *path* if it exists, otherwise generate a new one
        and persist it so the same UUID is reused on the next startup.
        """
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                identity = cls.from_dict(data)
                # Upgrade app_version field in case code was updated
                if identity.app_version != APP_VERSION:
                    identity.app_version = APP_VERSION
                    identity.save(path)
                log.info(
                    "Identity loaded: id=%s  name=%r  version=%s",
                    identity.device_id, identity.device_name, identity.app_version,
                )
                return identity
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning(
                    "Corrupt identity file %s (%s) — regenerating.", path, exc
                )

        identity = cls()
        identity.save(path)
        log.info(
            "New identity created: id=%s  name=%r",
            identity.device_id, identity.device_name,
        )
        return identity

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return f"DeviceIdentity(id={self.device_id[:8]}…, name={self.device_name!r})"
