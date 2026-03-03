"""
Unit & integration tests for gesturedrop discovery layer (Step 3)
=================================================================

Run with:
  pytest tests/test_discovery_service.py -v

Covers
------
  DeviceIdentity
    - Default field generation (UUID, hostname, version)
    - Serialisation round-trip (to_dict / from_dict)
    - Persistence (save / load_or_create)
    - Corrupt file → regenerate
    - App version upgrade on load

  Packet codec  (_build_send_intent, _build_ready_ack, _parse_packet)
    - Well-formed packets parse correctly
    - reply_port field present in SEND_INTENT
    - Missing required fields → None
    - Non-JSON → None; Non-UTF-8 → None
    - Stale / future / boundary timestamps
    - Invalid timestamp type → None
    - Capabilities included

  DiscoveryService — unit (no real sockets)
    - Self-filter: own device_id ignored via _dispatch and _dispatch_ack
    - IDLE receiver does not send READY_ACK
    - READY_TO_RECEIVE receiver sends READY_ACK to reply_port
    - Unknown packet type ignored silently
    - READY_ACK outside collection window dropped
    - READY_ACK inside window collected
    - on_send_intent_received callback fires
    - on_peer_found callback fires

  DiscoveryService — integration (real UDP loopback)
    - Sender broadcasts and READY receiver replies
    - IDLE receiver not included
    - Self not in peers
    - on_peer_found callback fires with correct PeerInfo
    - on_send_intent_received fires on receiver
    - Multiple receivers all collected
    - select_peer first-wins
    - start/stop lifecycle
    - Double start does not duplicate threads
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from gesturedrop.core.device_identity import DeviceIdentity, APP_VERSION
from gesturedrop.core.discovery_service import (
    DISCOVERY_PORT,
    MSG_READY_ACK,
    MSG_SEND_INTENT,
    TIMESTAMP_TOLERANCE,
    DiscoveryService,
    PeerInfo,
    _build_ready_ack,
    _build_send_intent,
    _parse_packet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_udp_port() -> int:
    """Ask the OS for a free UDP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_identity(name: str = "TestDevice") -> DeviceIdentity:
    return DeviceIdentity(device_id=str(uuid.uuid4()), device_name=name)


def _mock_receiver(ready: bool = False):
    svc = MagicMock()
    svc.can_reply_to_discovery.return_value = ready
    svc.state.name = "READY_TO_RECEIVE" if ready else "IDLE"
    return svc


def _make_disc(
    identity: DeviceIdentity,
    receiver_svc,
    port: int,
    transfer_port: int = 54321,
    wait: float = 5.0,
    **kwargs,
) -> DiscoveryService:
    return DiscoveryService(
        identity=identity,
        receiver_svc=receiver_svc,
        discovery_port=port,
        transfer_port=transfer_port,
        send_intent_wait=wait,
        **kwargs,
    )


# ===========================================================================
# DeviceIdentity Tests
# ===========================================================================

class TestDeviceIdentity:

    def test_default_fields_populated(self):
        identity = DeviceIdentity()
        assert identity.device_id
        assert len(identity.device_id) == 36
        assert identity.device_name
        assert identity.app_version == APP_VERSION
        assert isinstance(identity.capabilities, list)
        assert len(identity.capabilities) > 0

    def test_device_id_is_valid_uuid(self):
        identity = DeviceIdentity()
        parsed = uuid.UUID(identity.device_id)
        assert str(parsed) == identity.device_id

    def test_two_instances_have_different_ids(self):
        a = DeviceIdentity()
        b = DeviceIdentity()
        assert a.device_id != b.device_id

    def test_to_dict_roundtrip(self):
        original = _make_identity("Alice-PC")
        data = original.to_dict()
        restored = DeviceIdentity.from_dict(data)
        assert restored.device_id   == original.device_id
        assert restored.device_name == original.device_name
        assert restored.app_version == original.app_version
        assert restored.capabilities == original.capabilities

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "identity.json"
        original = _make_identity("SaveTest")
        original.save(path)
        assert path.exists()
        loaded = DeviceIdentity.load_or_create(path)
        assert loaded.device_id   == original.device_id
        assert loaded.device_name == original.device_name

    def test_load_creates_file_if_absent(self, tmp_path):
        path = tmp_path / "new_identity.json"
        assert not path.exists()
        identity = DeviceIdentity.load_or_create(path)
        assert path.exists()
        assert identity.device_id

    def test_corrupt_file_regenerates(self, tmp_path):
        path = tmp_path / "identity.json"
        path.write_text("!! not json !!", encoding="utf-8")
        identity = DeviceIdentity.load_or_create(path)
        assert identity.device_id

    def test_app_version_upgraded_on_load(self, tmp_path):
        path = tmp_path / "identity.json"
        old = _make_identity("TestDevice")
        old_dict = old.to_dict()
        old_dict["app_version"] = "0.0.1"
        path.write_text(json.dumps(old_dict), encoding="utf-8")
        loaded = DeviceIdentity.load_or_create(path)
        assert loaded.app_version == APP_VERSION

    def test_str_repr(self):
        identity = _make_identity("MyPC")
        s = str(identity)
        assert "MyPC" in s
        assert "…" in s


# ===========================================================================
# Packet Codec Tests
# ===========================================================================

class TestPacketCodec:

    def test_send_intent_is_valid_json(self):
        identity = _make_identity()
        raw = _build_send_intent(identity, 54321, reply_port=49152)
        obj = json.loads(raw.decode("utf-8"))
        assert obj["type"] == MSG_SEND_INTENT
        assert obj["device_id"] == identity.device_id
        assert obj["transfer_port"] == 54321
        assert obj["reply_port"] == 49152

    def test_ready_ack_is_valid_json(self):
        identity = _make_identity()
        raw = _build_ready_ack(identity, 54321)
        obj = json.loads(raw.decode("utf-8"))
        assert obj["type"] == MSG_READY_ACK
        assert obj["device_id"] == identity.device_id
        assert obj["accept_port"] == 54321

    def test_parse_valid_send_intent(self):
        identity = _make_identity()
        raw = _build_send_intent(identity, 54321, reply_port=49152)
        pkt = _parse_packet(raw)
        assert pkt is not None
        assert pkt["type"] == MSG_SEND_INTENT

    def test_parse_valid_ready_ack(self):
        identity = _make_identity()
        raw = _build_ready_ack(identity, 54321)
        pkt = _parse_packet(raw)
        assert pkt is not None
        assert pkt["type"] == MSG_READY_ACK

    def test_parse_non_utf8_returns_none(self):
        assert _parse_packet(b"\xff\xfe\xfd") is None

    def test_parse_non_json_returns_none(self):
        assert _parse_packet(b"hello world not json") is None

    def test_parse_json_array_returns_none(self):
        assert _parse_packet(b"[1, 2, 3]") is None

    def test_parse_missing_type_returns_none(self):
        obj = {"device_id": "abc", "timestamp": time.time()}
        assert _parse_packet(json.dumps(obj).encode()) is None

    def test_parse_missing_device_id_returns_none(self):
        obj = {"type": MSG_SEND_INTENT, "timestamp": time.time()}
        assert _parse_packet(json.dumps(obj).encode()) is None

    def test_parse_missing_timestamp_returns_none(self):
        obj = {"type": MSG_SEND_INTENT, "device_id": "abc"}
        assert _parse_packet(json.dumps(obj).encode()) is None

    def test_parse_stale_timestamp_returns_none(self):
        identity = _make_identity()
        raw = _build_send_intent(identity, 54321, reply_port=49152)
        pkt = json.loads(raw)
        pkt["timestamp"] = time.time() - (TIMESTAMP_TOLERANCE + 1)
        assert _parse_packet(json.dumps(pkt).encode()) is None

    def test_parse_future_timestamp_returns_none(self):
        identity = _make_identity()
        raw = _build_send_intent(identity, 54321, reply_port=49152)
        pkt = json.loads(raw)
        pkt["timestamp"] = time.time() + (TIMESTAMP_TOLERANCE + 1)
        assert _parse_packet(json.dumps(pkt).encode()) is None

    def test_parse_fresh_timestamp_at_boundary_is_accepted(self):
        identity = _make_identity()
        raw = _build_send_intent(identity, 54321, reply_port=49152)
        pkt = json.loads(raw)
        pkt["timestamp"] = time.time() - (TIMESTAMP_TOLERANCE - 0.5)
        assert _parse_packet(json.dumps(pkt).encode()) is not None

    def test_parse_invalid_timestamp_type_returns_none(self):
        obj = {
            "type": MSG_SEND_INTENT,
            "device_id": "abc",
            "timestamp": "not-a-number",
        }
        assert _parse_packet(json.dumps(obj).encode()) is None

    def test_capabilities_included_in_packets(self):
        identity = _make_identity()
        identity.capabilities = ["file_transfer_v1", "future_feature"]
        raw = _build_send_intent(identity, 54321, reply_port=49152)
        pkt = json.loads(raw)
        assert pkt["capabilities"] == ["file_transfer_v1", "future_feature"]


# ===========================================================================
# DiscoveryService Unit Tests (no real sockets — drives internal methods)
# ===========================================================================

class TestDiscoveryServiceUnit:

    def _send_intent_raw(self, identity: DeviceIdentity, reply_port: int = 49152) -> bytes:
        return _build_send_intent(identity, 54321, reply_port=reply_port)

    def _ready_ack_raw(self, identity: DeviceIdentity) -> bytes:
        return _build_ready_ack(identity, 54321)

    def test_self_filter_drops_own_send_intent(self):
        identity = _make_identity("Self")
        svc = _mock_receiver(ready=True)
        disc = _make_disc(identity, svc, port=_free_udp_port())
        sent: list = []
        disc._send_unicast = lambda d, h, p: sent.append((h, p, d))

        # Packet from our own device_id
        raw = self._send_intent_raw(identity)
        disc._dispatch(raw, "127.0.0.1")
        assert len(sent) == 0

    def test_self_filter_drops_own_ready_ack_on_reply_socket(self):
        identity = _make_identity("Self")
        svc = _mock_receiver(ready=False)
        disc = _make_disc(identity, svc, port=_free_udp_port())
        # Prime collection window
        with disc._collecting_lock:
            disc._collecting = True
            disc._ready_acks = []

        raw = self._ready_ack_raw(identity)  # own id
        disc._dispatch_ack(raw, "192.168.1.44")
        assert len(disc._ready_acks) == 0

    def test_idle_receiver_does_not_reply(self):
        my_id    = _make_identity("Receiver")
        their_id = _make_identity("Sender")
        svc = _mock_receiver(ready=False)
        disc = _make_disc(my_id, svc, port=_free_udp_port())
        sent: list = []
        disc._send_unicast = lambda d, h, p: sent.append((h, p))

        disc._dispatch(self._send_intent_raw(their_id), "192.168.1.50")
        assert len(sent) == 0

    def test_ready_receiver_replies_to_reply_port(self):
        my_id    = _make_identity("Receiver")
        their_id = _make_identity("Sender")
        svc = _mock_receiver(ready=True)
        disc = _make_disc(my_id, svc, port=_free_udp_port())
        sent: list = []
        disc._send_unicast = lambda d, h, p: sent.append((h, p, d))

        # reply_port=55555 embedded in the SEND_INTENT
        raw = _build_send_intent(their_id, 54321, reply_port=55555)
        disc._dispatch(raw, "192.168.1.50")

        assert len(sent) == 1
        host, port, data = sent[0]
        assert host == "192.168.1.50"
        assert port == 55555         # must honour the embedded reply_port
        pkt = json.loads(data.decode())
        assert pkt["type"] == MSG_READY_ACK

    def test_unknown_packet_type_ignored(self):
        my_id    = _make_identity("Me")
        their_id = _make_identity("Them")
        svc = _mock_receiver(ready=True)
        disc = _make_disc(my_id, svc, port=_free_udp_port())
        sent: list = []
        disc._send_unicast = lambda d, h, p: sent.append(1)

        payload = json.dumps({
            "type": "UNKNOWN_MSG_TYPE",
            "device_id": their_id.device_id,
            "timestamp": time.time(),
        }).encode()
        disc._dispatch(payload, "10.0.0.1")
        assert len(sent) == 0

    def test_ready_ack_outside_window_dropped(self):
        my_id    = _make_identity("Sender")
        their_id = _make_identity("Receiver")
        svc = _mock_receiver(ready=False)
        disc = _make_disc(my_id, svc, port=_free_udp_port())
        # _collecting is False by default
        disc._dispatch_ack(self._ready_ack_raw(their_id), "192.168.1.99")
        assert len(disc._ready_acks) == 0

    def test_ready_ack_inside_window_collected(self):
        my_id    = _make_identity("Sender")
        their_id = _make_identity("Receiver")
        svc = _mock_receiver(ready=False)
        disc = _make_disc(my_id, svc, port=_free_udp_port())
        with disc._collecting_lock:
            disc._collecting = True
            disc._ready_acks = []

        disc._dispatch_ack(self._ready_ack_raw(their_id), "192.168.1.99")
        assert len(disc._ready_acks) == 1
        assert disc._ready_acks[0].host == "192.168.1.99"

    def test_malformed_packet_does_not_crash(self):
        my_id = _make_identity("Me")
        svc   = _mock_receiver(ready=True)
        disc  = _make_disc(my_id, svc, port=_free_udp_port())
        disc._dispatch(b"\xff\xfe corrupt garbage \x00\x01", "10.0.0.2")
        disc._dispatch(b"", "10.0.0.2")
        disc._dispatch(b"null", "10.0.0.2")
        disc._dispatch_ack(b"\xff\xfe\xfd", "10.0.0.2")

    def test_on_send_intent_received_callback_fires(self):
        my_id    = _make_identity("Receiver")
        their_id = _make_identity("Sender")
        svc = _mock_receiver(ready=False)
        received: list = []
        ev = threading.Event()

        def cb(pkt, ip):
            received.append((pkt, ip))
            ev.set()

        disc = _make_disc(my_id, svc, port=_free_udp_port(), on_send_intent_received=cb)
        raw = _build_send_intent(their_id, 54321, reply_port=49152)
        disc._dispatch(raw, "10.0.0.5")
        ev.wait(timeout=1.0)
        assert len(received) == 1
        assert received[0][1] == "10.0.0.5"

    def test_on_peer_found_callback_fires(self):
        my_id    = _make_identity("Sender")
        their_id = _make_identity("Receiver")
        svc = _mock_receiver(ready=False)
        found: List[PeerInfo] = []
        ev = threading.Event()

        def cb(peer):
            found.append(peer)
            ev.set()

        disc = _make_disc(my_id, svc, port=_free_udp_port(), on_peer_found=cb)
        with disc._collecting_lock:
            disc._collecting = True
            disc._ready_acks = []

        disc._dispatch_ack(self._ready_ack_raw(their_id), "10.0.0.7")
        ev.wait(timeout=1.0)
        assert len(found) == 1
        assert found[0].device_name == "Receiver"

    def test_select_peer_returns_first(self):
        peers = [
            PeerInfo("id1", "Alpha", "10.0.0.1", 54321, ()),
            PeerInfo("id2", "Beta",  "10.0.0.2", 54321, ()),
        ]
        assert DiscoveryService.select_peer(peers) == peers[0]

    def test_select_peer_returns_none_for_empty(self):
        assert DiscoveryService.select_peer([]) is None


# ===========================================================================
# DiscoveryService Integration Tests (real UDP, loopback)
# ===========================================================================

class TestDiscoveryServiceIntegration:
    """
    Two DiscoveryService instances on different discovery ports.
    The sender uses a per-broadcast ephemeral reply socket; the receiver
    sends READY_ACK to that ephemeral port, so the OS delivers it
    exclusively to the sender's receive socket.
    """

    WAIT = 1.5   # short windows to keep the suite fast

    def setup_method(self):
        self.port = _free_udp_port()

    def test_receiver_ready_replies_to_sender(self):
        sender_id   = _make_identity("Sender")
        receiver_id = _make_identity("Receiver")

        receiver_svc  = _mock_receiver(ready=True)
        receiver_disc = _make_disc(receiver_id, receiver_svc, self.port,
                                   transfer_port=54322, wait=self.WAIT)

        sender_svc  = _mock_receiver(ready=False)
        sender_disc = _make_disc(sender_id, sender_svc, self.port,
                                 transfer_port=54321, wait=self.WAIT)

        receiver_disc.start()
        sender_disc.start()
        time.sleep(0.2)

        try:
            peers = sender_disc.broadcast_send_intent(transfer_port=54321, wait=self.WAIT)
        finally:
            sender_disc.stop()
            receiver_disc.stop()

        assert len(peers) >= 1
        match = next((p for p in peers if p.device_id == receiver_id.device_id), None)
        assert match is not None, f"Receiver not in peers: {peers}"
        assert match.device_name == "Receiver"
        assert match.accept_port == 54322

    def test_idle_receiver_not_included(self):
        sender_id   = _make_identity("Sender")
        receiver_id = _make_identity("IdleReceiver")

        receiver_svc  = _mock_receiver(ready=False)
        receiver_disc = _make_disc(receiver_id, receiver_svc, self.port, wait=self.WAIT)

        sender_svc  = _mock_receiver(ready=False)
        sender_disc = _make_disc(sender_id, sender_svc, self.port, wait=self.WAIT)

        receiver_disc.start()
        sender_disc.start()
        time.sleep(0.2)

        try:
            peers = sender_disc.broadcast_send_intent(wait=self.WAIT)
        finally:
            sender_disc.stop()
            receiver_disc.stop()

        assert peers == []

    def test_self_not_in_peers(self):
        sender_id  = _make_identity("AloneDevice")
        sender_svc = _mock_receiver(ready=True)   # even if READY, must not self-pair
        sender_disc = _make_disc(sender_id, sender_svc, self.port, wait=0.5)
        sender_disc.start()
        time.sleep(0.15)

        try:
            peers = sender_disc.broadcast_send_intent(wait=0.5)
        finally:
            sender_disc.stop()

        self_match = [p for p in peers if p.device_id == sender_id.device_id]
        assert self_match == [], "Device paired with itself!"

    def test_on_peer_found_callback_integration(self):
        sender_id   = _make_identity("SenderCb")
        receiver_id = _make_identity("ReceiverCb")

        found: List[PeerInfo] = []
        ev = threading.Event()

        def on_found(peer):
            found.append(peer)
            ev.set()

        receiver_svc  = _mock_receiver(ready=True)
        receiver_disc = _make_disc(receiver_id, receiver_svc, self.port, wait=self.WAIT)

        sender_svc  = _mock_receiver(ready=False)
        sender_disc = _make_disc(sender_id, sender_svc, self.port,
                                 wait=self.WAIT, on_peer_found=on_found)

        receiver_disc.start()
        sender_disc.start()
        time.sleep(0.2)

        try:
            peers = sender_disc.broadcast_send_intent(wait=self.WAIT)
        finally:
            sender_disc.stop()
            receiver_disc.stop()

        ev.wait(timeout=2.0)
        assert len(found) >= 1
        assert found[0].device_id == receiver_id.device_id

    def test_on_send_intent_received_integration(self):
        sender_id   = _make_identity("SenderSI")
        receiver_id = _make_identity("ReceiverSI")

        intents: list = []
        si_ev = threading.Event()

        def on_si(pkt, ip):
            intents.append((pkt, ip))
            si_ev.set()

        receiver_svc  = _mock_receiver(ready=False)
        receiver_disc = _make_disc(receiver_id, receiver_svc, self.port,
                                   wait=self.WAIT, on_send_intent_received=on_si)

        sender_svc  = _mock_receiver(ready=False)
        sender_disc = _make_disc(sender_id, sender_svc, self.port, wait=0.4)

        receiver_disc.start()
        sender_disc.start()
        time.sleep(0.2)

        try:
            sender_disc.broadcast_send_intent(wait=0.4)
        finally:
            sender_disc.stop()
            receiver_disc.stop()

        si_ev.wait(timeout=2.0)
        assert len(intents) >= 1
        pkt, ip = intents[0]
        assert pkt["type"] == MSG_SEND_INTENT
        assert pkt["device_id"] == sender_id.device_id

    def test_multiple_receivers_all_collected(self):
        """Both receivers READY on same port — both PeerInfos appear in result."""
        sender_id    = _make_identity("MultiSender")
        receiver1_id = _make_identity("Recv1")
        receiver2_id = _make_identity("Recv2")

        r1_svc  = _mock_receiver(ready=True)
        r2_svc  = _mock_receiver(ready=True)
        r1_disc = _make_disc(receiver1_id, r1_svc, self.port,
                             transfer_port=54401, wait=self.WAIT)
        r2_disc = _make_disc(receiver2_id, r2_svc, self.port,
                             transfer_port=54402, wait=self.WAIT)

        sender_svc  = _mock_receiver(ready=False)
        sender_disc = _make_disc(sender_id, sender_svc, self.port, wait=self.WAIT)

        r1_disc.start()
        r2_disc.start()
        sender_disc.start()
        time.sleep(0.25)

        try:
            peers = sender_disc.broadcast_send_intent(wait=self.WAIT)
        finally:
            sender_disc.stop()
            r1_disc.stop()
            r2_disc.stop()

        peer_ids = {p.device_id for p in peers}
        assert receiver1_id.device_id in peer_ids, "Receiver 1 not found"
        assert receiver2_id.device_id in peer_ids, "Receiver 2 not found"

    def test_service_stops_cleanly(self):
        identity = _make_identity("StopTest")
        svc = _mock_receiver(ready=False)
        disc = _make_disc(identity, svc, self.port, wait=self.WAIT)
        disc.start()
        time.sleep(0.1)
        disc.stop()
        if disc._listener_thread:
            assert not disc._listener_thread.is_alive()

    def test_double_start_does_not_duplicate_thread(self):
        identity = _make_identity("DoubleStart")
        svc = _mock_receiver(ready=False)
        disc = _make_disc(identity, svc, self.port, wait=self.WAIT)
        disc.start()
        disc.start()
        time.sleep(0.1)
        try:
            assert disc._listener_thread is not None
        finally:
            disc.stop()
