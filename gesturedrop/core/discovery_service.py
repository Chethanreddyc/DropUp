"""
GestureDrop — Discovery Service (Step 3)
=========================================

Responsibilities
----------------
  • Broadcast SEND_INTENT packets over UDP when the local device wants to send
  • Listen for SEND_INTENT packets and reply with READY_ACK when receiver is ready
  • Validate every packet: JSON schema, timestamp freshness, self-filter
  • Collect READY_ACK responses during a bounded matching window
  • Never block the calling thread — all I/O runs in daemon threads
  • Never crash on malformed input

Network Design
--------------
  Always-on listener  : 0.0.0.0 : DISCOVERY_PORT (50000)   — broadcast listener
  Per-broadcast reply : ephemeral OS port                   — for READY_ACK replies
  TCP transfer        : peer IP  : TRANSFER_PORT (54321)    — file data only

  Separation matters: on Windows, SO_REUSEADDR for UDP does NOT fan out unicast
  datagrams to all sockets sharing the same port — only one socket receives each
  datagram.  Embedding a per-broadcast reply_port in SEND_INTENT and directing
  READY_ACK to that ephemeral port ensures only the sender's reply socket
  receives the reply, regardless of how many GestureDrop instances run on the
  same host.

Packet Schemas  (JSON, one object per UDP datagram)
---------------------------------------------------
  SEND_INTENT  (broadcaster → all devices on DISCOVERY_PORT)
  {
    "type":          "SEND_INTENT",
    "device_id":     "<uuid4>",
    "device_name":   "Alice-PC",
    "app_version":   "0.3.0",
    "capabilities":  ["file_transfer_v1"],
    "timestamp":     1709295432.841,      ← Unix time float
    "transfer_port": 54321,               ← TCP port for the actual file transfer
    "reply_port":    49152                ← Ephemeral UDP port to send READY_ACK to
  }

  READY_ACK  (receiver → sender's reply_port, unicast)
  {
    "type":          "READY_ACK",
    "device_id":     "<uuid4>",
    "device_name":   "Bob-PC",
    "app_version":   "0.3.0",
    "capabilities":  ["file_transfer_v1"],
    "timestamp":     1709295432.900,
    "accept_port":   54321
  }

State Integration
-----------------
  Receiver state     Reply to SEND_INTENT?
  ─────────────────  ─────────────────────
  IDLE               NO  (silent ignore)
  READY_TO_RECEIVE   YES → send READY_ACK to sender's reply_port
  RECEIVING          NO
  BUSY               NO

Failure Handling
----------------
  • Malformed / non-JSON datagrams → log warning, continue
  • Unknown packet type            → log debug, ignore
  • Timestamp too old/new          → log warning, ignore (replay guard)
  • Own device_id received         → log debug, ignore (self-filter)
  • Socket errors in listener      → log error, keep loop alive
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from gesturedrop.core.device_identity import DeviceIdentity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCOVERY_PORT: int = 50_000          # UDP port for inbound SEND_INTENT broadcasts
BROADCAST_ADDR: str = "255.255.255.255"
LISTEN_ADDR: str = "0.0.0.0"

# Message type tags
MSG_SEND_INTENT: str = "SEND_INTENT"
MSG_READY_ACK:   str = "READY_ACK"

# Timing
SEND_INTENT_WAIT: float    = 5.0    # Sender waits this long for READY_ACK responses
TIMESTAMP_TOLERANCE: float = 10.0   # Max age (seconds) of an accepted discovery packet

# Packet size cap
MAX_DATAGRAM: int = 4096

# Default TCP transfer port (matches receiver_service.LISTEN_PORT)
DEFAULT_TRANSFER_PORT: int = 54321

log = logging.getLogger("GestureDrop.Discovery")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeerInfo:
    """
    A discovered peer that replied READY_ACK.

    Attributes
    ----------
    device_id    : str   — UUID of the remote device
    device_name  : str   — human label (e.g. "Bob-PC")
    host         : str   — IP address the ACK arrived from
    accept_port  : int   — TCP port on which the remote receiver is listening
    capabilities : tuple — immutable set of capability strings
    """
    device_id: str
    device_name: str
    host: str
    accept_port: int
    capabilities: tuple


# ---------------------------------------------------------------------------
# Internal packet helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _build_send_intent(
    identity: DeviceIdentity,
    transfer_port: int,
    reply_port: int,
) -> bytes:
    """Serialise a SEND_INTENT datagram.

    Parameters
    ----------
    reply_port : ephemeral UDP port the sender is listening on for READY_ACK
                 replies.  Receivers must send their READY_ACK to this port so
                 that the reply is delivered exclusively to the waiting sender
                 socket (not to the shared discovery-port socket).
    """
    payload = {
        "type":          MSG_SEND_INTENT,
        "device_id":     identity.device_id,
        "device_name":   identity.device_name,
        "app_version":   identity.app_version,
        "capabilities":  identity.capabilities,
        "timestamp":     _now(),
        "transfer_port": transfer_port,
        "reply_port":    reply_port,
    }
    return json.dumps(payload).encode("utf-8")


def _build_ready_ack(
    identity: DeviceIdentity,
    accept_port: int,
) -> bytes:
    """Serialise a READY_ACK datagram."""
    payload = {
        "type":        MSG_READY_ACK,
        "device_id":   identity.device_id,
        "device_name": identity.device_name,
        "app_version": identity.app_version,
        "capabilities": identity.capabilities,
        "timestamp":   _now(),
        "accept_port": accept_port,
    }
    return json.dumps(payload).encode("utf-8")


def _parse_packet(raw: bytes) -> Optional[dict]:
    """
    Decode a raw UDP payload.

    Returns the parsed dict, or None if the packet is invalid.

    Validation rules
    ----------------
    - Must be valid UTF-8
    - Must be valid JSON object (not array / scalar)
    - Must contain "type", "device_id", and "timestamp"
    - Timestamp must be within ±TIMESTAMP_TOLERANCE of now (replay guard)
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("Discovery: received non-UTF-8 datagram — ignored.")
        return None

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Discovery: JSON parse error (%s) — ignored.", exc)
        return None

    if not isinstance(obj, dict):
        log.warning("Discovery: packet is not a JSON object — ignored.")
        return None

    for required in ("type", "device_id", "timestamp"):
        if required not in obj:
            log.warning(
                "Discovery: missing required field %r — ignored.", required
            )
            return None

    try:
        ts = float(obj["timestamp"])
    except (TypeError, ValueError):
        log.warning("Discovery: invalid timestamp value — ignored.")
        return None

    age = abs(_now() - ts)
    if age > TIMESTAMP_TOLERANCE:
        log.warning(
            "Discovery: stale packet (age=%.1fs, tolerance=%.1fs) — ignored.",
            age, TIMESTAMP_TOLERANCE,
        )
        return None

    return obj


# ---------------------------------------------------------------------------
# Discovery Service
# ---------------------------------------------------------------------------

class DiscoveryService:
    """
    UDP-based peer discovery for GestureDrop.

    Sender usage
    ------------
    disc = DiscoveryService(identity, receiver_svc)
    disc.start()
    peers = disc.broadcast_send_intent(transfer_port=54321)
    peer  = DiscoveryService.select_peer(peers)
    # … initiate TCP transfer to peer.host:peer.accept_port …

    Receiver usage
    --------------
    disc = DiscoveryService(identity, receiver_svc)
    disc.start()
    # The service automatically replies to SEND_INTENT packets when
    # receiver_svc.can_reply_to_discovery() returns True.
    # No further calls needed.

    disc.stop()   # clean shutdown
    """

    def __init__(
        self,
        identity: DeviceIdentity,
        receiver_svc,                           # ReceiverService (duck-typed)
        discovery_port: int = DISCOVERY_PORT,
        transfer_port:  int = DEFAULT_TRANSFER_PORT,
        send_intent_wait: float = SEND_INTENT_WAIT,
        on_peer_found: Optional[Callable[[PeerInfo], None]] = None,
        on_send_intent_received: Optional[Callable[[dict, str], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        identity         : DeviceIdentity — local device's stable identity
        receiver_svc     : ReceiverService — controls the discovery reply gate
        discovery_port   : int   — UDP port for inbound SEND_INTENT broadcasts
        transfer_port    : int   — TCP port advertised in READY_ACK
        send_intent_wait : float — seconds to collect READY_ACK replies
        on_peer_found    : optional callback(PeerInfo) — fires when a READY_ACK
                           arrives during an active broadcast_send_intent() window
        on_send_intent_received : optional callback(packet, sender_ip) — fires on
                           every received (and filtered) SEND_INTENT — useful for
                           UI / debug
        """
        self._identity          = identity
        self._receiver_svc      = receiver_svc
        self._discovery_port    = discovery_port
        self._transfer_port     = transfer_port
        self._send_intent_wait  = send_intent_wait
        self._on_peer_found     = on_peer_found
        self._on_send_intent_received = on_send_intent_received

        # ── Always-on broadcast listener ────────────────────────────────
        self._running: bool = False
        self._listener_thread: Optional[threading.Thread] = None
        self._udp_sock: Optional[socket.socket] = None

        # ── Sender-side: per-broadcast ephemeral reply socket ───────────
        # While broadcast_send_intent() is active this socket is open and
        # _collecting is True.  READY_ACK datagrams arrive here exclusively.
        self._reply_sock: Optional[socket.socket] = None
        self._collecting: bool = False
        self._collecting_lock: threading.Lock = threading.Lock()
        self._ready_acks: List[PeerInfo] = []
        self._ack_event: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Public API — Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the always-on UDP listener thread (receive SEND_INTENT). Safe to call once."""
        if self._running:
            log.warning("DiscoveryService.start() called while already running.")
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name="GD-DiscoveryListener",
            daemon=True,
        )
        self._listener_thread.start()
        log.info(
            "DiscoveryService started  port=%d  id=%s  name=%r",
            self._discovery_port,
            self._identity.device_id,
            self._identity.device_name,
        )

    def stop(self) -> None:
        """Signal the listener to stop and join its thread."""
        if not self._running:
            return
        self._running = False
        if self._udp_sock is not None:
            with contextlib.suppress(OSError):
                self._udp_sock.close()
        if self._reply_sock is not None:
            with contextlib.suppress(OSError):
                self._reply_sock.close()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=3.0)
        log.info("DiscoveryService stopped.")

    # ------------------------------------------------------------------
    # Public API — Sender: broadcast SEND_INTENT and collect replies
    # ------------------------------------------------------------------

    def broadcast_send_intent(
        self,
        transfer_port: Optional[int] = None,
        wait: Optional[float] = None,
    ) -> List[PeerInfo]:
        """
        Broadcast a SEND_INTENT packet and wait for READY_ACK replies.

        Opens a short-lived ephemeral UDP socket exclusively for READY_ACK
        collection (the reply_port).  Receivers directed there by the
        reply_port field embedded in the SEND_INTENT.

        This call BLOCKS the calling thread for up to *wait* seconds.
        Call it from a background / worker thread — never from the listener thread.

        Parameters
        ----------
        transfer_port : TCP port to advertise (default: self._transfer_port)
        wait          : seconds to collect replies (default: send_intent_wait)

        Returns
        -------
        List[PeerInfo] — discovered peers that replied READY_ACK, in arrival order.
        Empty list means no receiver responded inside the time window.
        """
        tcp_port = transfer_port if transfer_port is not None else self._transfer_port
        window   = wait if wait is not None else self._send_intent_wait

        # ── Open ephemeral reply socket ─────────────────────────────────
        reply_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reply_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reply_sock.settimeout(0.1)           # short poll interval inside the loop
        reply_sock.bind((LISTEN_ADDR, 0))    # OS picks an ephemeral port
        reply_port = reply_sock.getsockname()[1]
        self._reply_sock = reply_sock

        log.info(
            "SEND_INTENT broadcast  transfer_port=%d  reply_port=%d  wait=%.1fs",
            tcp_port, reply_port, window,
        )

        # ── Open collection window ──────────────────────────────────────
        with self._collecting_lock:
            self._collecting = True
            self._ready_acks = []
            self._ack_event.clear()

        try:
            pkt = _build_send_intent(self._identity, tcp_port, reply_port)
            self._send_broadcast(pkt)

            # ── Poll the reply socket for READY_ACK datagrams ───────────
            deadline = time.monotonic() + window
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw, (sender_ip, _) = reply_sock.recvfrom(MAX_DATAGRAM)
                    self._dispatch_ack(raw, sender_ip)
                except socket.timeout:
                    continue
                except OSError:
                    break   # closed early via stop()
        finally:
            with contextlib.suppress(OSError):
                reply_sock.close()
            self._reply_sock = None

            with self._collecting_lock:
                self._collecting = False
                results = list(self._ready_acks)

        if results:
            log.info(
                "Matching window closed — %d peer(s) replied: %s",
                len(results),
                [f"{p.device_name}@{p.host}" for p in results],
            )
        else:
            log.info("Matching window closed — no peers replied.")

        return results

    # ------------------------------------------------------------------
    # Public API — Peer selection strategy
    # ------------------------------------------------------------------

    @staticmethod
    def select_peer(peers: List[PeerInfo]) -> Optional[PeerInfo]:
        """
        First-valid-READY-wins peer selection (simplest correct policy).
        Future: can be swapped for signal-strength or manual-UI selection.
        """
        return peers[0] if peers else None

    # ------------------------------------------------------------------
    # Internal — Always-on UDP listener (receive SEND_INTENT, dispatch)
    # ------------------------------------------------------------------

    def _listener_loop(self) -> None:
        """
        Binds to DISCOVERY_PORT and processes inbound datagrams.
        Handles only SEND_INTENT here; READY_ACKs are received on the
        ephemeral reply socket inside broadcast_send_intent().
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1.0)
            sock.bind((LISTEN_ADDR, self._discovery_port))
            self._udp_sock = sock
            log.debug("UDP listener bound on %s:%d", LISTEN_ADDR, self._discovery_port)

            while self._running:
                try:
                    raw, (sender_ip, _) = sock.recvfrom(MAX_DATAGRAM)
                except socket.timeout:
                    continue
                except OSError:
                    break

                self._dispatch(raw, sender_ip)

        except Exception as exc:           # pragma: no cover
            log.exception("Fatal error in discovery listener: %s", exc)
        finally:
            self._udp_sock = None
            log.debug("Discovery listener loop exited.")

    # ------------------------------------------------------------------
    # Internal — Dispatch inbound SEND_INTENT (listener thread path)
    # ------------------------------------------------------------------

    def _dispatch(self, raw: bytes, sender_ip: str) -> None:
        """Parse and route one datagram arriving on the discovery port."""
        packet = _parse_packet(raw)
        if packet is None:
            return

        pkt_type      = packet.get("type", "UNKNOWN")
        pkt_device_id = str(packet.get("device_id", ""))

        # ── Self-filter ────────────────────────────────────────────────
        if pkt_device_id == self._identity.device_id:
            log.debug("Ignored own broadcast (%s) from %s", pkt_type, sender_ip)
            return

        if pkt_type == MSG_SEND_INTENT:
            self._handle_send_intent(packet, sender_ip)
        elif pkt_type == MSG_READY_ACK:
            # A stray READY_ACK on the discovery port — try to collect it
            # (may happen if the receiver doesn't yet know the reply_port)
            self._handle_ready_ack(packet, sender_ip)
        else:
            log.debug("Unknown packet type %r from %s — ignored.", pkt_type, sender_ip)

    # ------------------------------------------------------------------
    # Internal — Dispatch READY_ACK (reply-socket path)
    # ------------------------------------------------------------------

    def _dispatch_ack(self, raw: bytes, sender_ip: str) -> None:
        """
        Parse one datagram that arrived on the ephemeral reply socket.
        Expected to be a READY_ACK; anything else is filtered out.
        """
        packet = _parse_packet(raw)
        if packet is None:
            return

        pkt_type      = packet.get("type", "UNKNOWN")
        pkt_device_id = str(packet.get("device_id", ""))

        if pkt_device_id == self._identity.device_id:
            log.debug("Ignored own READY_ACK echo from %s", sender_ip)
            return

        if pkt_type == MSG_READY_ACK:
            self._handle_ready_ack(packet, sender_ip)
        else:
            log.debug(
                "Unexpected packet type %r on reply socket from %s — ignored.",
                pkt_type, sender_ip,
            )

    # ------------------------------------------------------------------
    # Internal — Handle SEND_INTENT (receiver side)
    # ------------------------------------------------------------------

    def _handle_send_intent(self, packet: dict, sender_ip: str) -> None:
        """
        A remote device wants to send.  Reply with READY_ACK only when
        the local receiver's state machine allows it.
        """
        sender_name   = packet.get("device_name", "unknown")
        transfer_port = packet.get("transfer_port", DEFAULT_TRANSFER_PORT)
        reply_port    = int(packet.get("reply_port", self._discovery_port))

        log.debug(
            "SEND_INTENT from %s (%s)  transfer_port=%s  reply_port=%d",
            sender_ip, sender_name, transfer_port, reply_port,
        )

        # Fire the optional raw-packet callback (for UI / debug)
        if self._on_send_intent_received is not None:
            threading.Thread(
                target=self._on_send_intent_received,
                args=(packet, sender_ip),
                daemon=True,
                name="GD-DiscCb",
            ).start()

        # ── Discovery Gate — Step 2 → Step 3 link ─────────────────────
        if not self._receiver_svc.can_reply_to_discovery():
            log.debug(
                "SEND_INTENT from %s ignored — receiver state=%s.",
                sender_ip, self._receiver_svc.state.name,
            )
            return

        # Reply unicast to the sender's ephemeral reply_port
        ack = _build_ready_ack(self._identity, self._transfer_port)
        try:
            self._send_unicast(ack, sender_ip, reply_port)
            log.info(
                "READY_ACK sent → %s:%d (%s)  accept_port=%d",
                sender_ip, reply_port, sender_name, self._transfer_port,
            )
        except OSError as exc:
            log.error("Failed to send READY_ACK to %s:%d: %s", sender_ip, reply_port, exc)

    # ------------------------------------------------------------------
    # Internal — Handle READY_ACK (sender side)
    # ------------------------------------------------------------------

    def _handle_ready_ack(self, packet: dict, sender_ip: str) -> None:
        """
        A remote device replied READY.  Record it in the collection window
        if one is currently active.
        """
        device_id    = str(packet.get("device_id", ""))
        device_name  = str(packet.get("device_name", "unknown"))
        accept_port  = int(packet.get("accept_port", DEFAULT_TRANSFER_PORT))
        capabilities = tuple(packet.get("capabilities", []))

        peer = PeerInfo(
            device_id=device_id,
            device_name=device_name,
            host=sender_ip,
            accept_port=accept_port,
            capabilities=capabilities,
        )

        log.info(
            "READY_ACK from %s (%s)  accept_port=%d",
            sender_ip, device_name, accept_port,
        )

        with self._collecting_lock:
            if self._collecting:
                self._ready_acks.append(peer)
                self._ack_event.set()
                if self._on_peer_found is not None:
                    threading.Thread(
                        target=self._on_peer_found,
                        args=(peer,),
                        daemon=True,
                        name="GD-PeerCb",
                    ).start()
            else:
                log.debug(
                    "READY_ACK from %s arrived outside collection window — dropped.",
                    sender_ip,
                )

    # ------------------------------------------------------------------
    # Internal — Socket send helpers
    # ------------------------------------------------------------------

    def _send_broadcast(self, data: bytes) -> None:
        """UDP-broadcast *data* to BROADCAST_ADDR on DISCOVERY_PORT."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.sendto(data, (BROADCAST_ADDR, self._discovery_port))

    def _send_unicast(self, data: bytes, host: str, port: int) -> None:
        """UDP-unicast *data* to (host, port)."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(data, (host, port))
