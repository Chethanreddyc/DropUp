# GestureDrop — Setup & Run Guide

Gesture-driven peer-to-peer file transfer between two PCs on the same network.

---

## Quick Start (Second PC)

### 1. Clone the repo

```bash
git clone https://github.com/Chethanreddyc/DropUp.git
cd DropUp
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** mediapipe is pinned to `0.10.8` — do NOT upgrade it.  
> Versions 0.10.14+ removed the `mp.solutions.hands` API that GestureDrop uses.

### 4. Run GestureDrop

```bash
# Camera mode (default) — requires webcam
python -m gesturedrop

# Headless mode — keyboard driven, no camera needed
python -m gesturedrop --headless

# One-shot send (reads clipboard and sends immediately)
python -m gesturedrop --send

# One-shot receive (opens receive window for 15 seconds)
python -m gesturedrop --receive

# Debug logging (verbose console output)
python -m gesturedrop --debug
```

---

## Two-PC File Transfer Demo

### On PC-B (receiver):
```bash
python -m gesturedrop --headless
# Type: r     (opens receive window for 10 seconds)
```

### On PC-A (sender):
```bash
# Copy a file/text/image to clipboard, then:
python -m gesturedrop --headless
# Type: s     (reads clipboard, discovers PC-B, sends file)
```

### With camera (both PCs):
```bash
python -m gesturedrop
# PC-B: Show RECEIVE gesture (fist -> open palm)
# PC-A: Show SEND gesture (open palm -> fist)
```

### What happens behind the scenes:
```
PC-A (sender)                          PC-B (receiver)
─────────────                          ────────────────
1. Reads clipboard                     1. Enters READY_TO_RECEIVE
2. Computes SHA-256 hash               2. Listens on TCP port 54321
3. Broadcasts SEND_INTENT (UDP 50000)  3. Replies READY_ACK (UDP)
4. Connects to PC-B:54321             4. Sends ACCEPT
5. Sends v2 header (name+size+hash)    5. Reads header
6. Streams file chunks                 6. Receives + hashes chunks
7. Waits for confirmation              7. Verifies SHA-256 → sends DONE
8. ✅ Transfer complete                8. ✅ File saved to received_files/
```

---

## Network Requirements

- Both PCs must be on the **same LAN / Wi-Fi network**
- UDP port `50000` must be open (discovery)
- TCP port `54321` must be open (file transfer)
- Windows Firewall may prompt — click **Allow**

---

## Project Structure

```
DropUp/
├── gesturedrop/
│   ├── __init__.py            # Package root
│   ├── __main__.py            # Entry point (python -m gesturedrop)
│   ├── core/
│   │   ├── device_identity.py # Persistent UUID per machine
│   │   ├── discovery_service.py # UDP peer discovery
│   │   ├── receiver_service.py  # TCP file receiver (v2 protocol)
│   │   ├── sender_service.py    # TCP file sender (v2 protocol)
│   │   ├── integrity.py         # SHA-256 hashing utilities
│   │   ├── content_manager.py   # Clipboard → file preparation
│   │   ├── gesture.py           # MediaPipe hand gesture detection
│   │   ├── gesture_controller.py # SEND/RECEIVE flow orchestration
│   │   └── screenshot_service.py # Screen capture fallback
│   └── ui/
│       └── toast_service.py     # Cross-platform notifications
├── tests/                       # pytest test suite (150+ tests)
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Dev + test dependencies
├── pyproject.toml               # Build config
└── README.md                    # This file
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
