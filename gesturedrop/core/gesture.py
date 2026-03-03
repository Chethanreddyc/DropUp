"""
GestureDrop — Gesture Detection (Step 5B rewrite)
==================================================

Architecture (layered, fully separated)
-----------------------------------------
  RawClassifier       — single-frame hand → OPEN / FIST / UNKNOWN
  Stabilizer          — requires gesture to hold for ≥ STABLE_DURATION ms
  TransitionDetector  — OPEN→FIST = SEND, FIST→OPEN = RECEIVE
                        with per-action cooldowns and a neutral lock
  GestureDetector     — composes all three + spatial guards

All 10 improvements over the original implementation
------------------------------------------------------
  1. Angle-based finger detection (rotation-invariant)
  2. Handedness-aware thumb logic (left vs right)
  3. UNKNOWN never enters stabiliser or transition history
  4. Neutral lock prevents back-to-back accidental triggers
  5. Time-based stabilisation (≥ 300 ms, FPS-independent)
  6. Center-zone constraint (middle 40% of frame)
  7. Minimum hand size guard (> 3% of frame area)
  8. Separate SEND / RECEIVE cooldowns
  9. Confidence filter (score < 0.75 → treat as UNKNOWN)
 10. Detector / Stabiliser / Transition fully separated

Gesture‑to‑Action mapping
--------------------------
  Palm → Fist  (OPEN → FIST)   :  SEND
  Fist → Palm  (FIST → OPEN)   :  RECEIVE
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp


# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

STABLE_DURATION:     float = 0.30   # seconds gesture must remain stable
SEND_COOLDOWN:       float = 2.0    # seconds between consecutive SEND actions
RECEIVE_COOLDOWN:    float = 2.0    # seconds between consecutive RECEIVE actions
NEUTRAL_HOLD_TIME:   float = 0.40   # seconds of no-hand / UNKNOWN to lift neutral lock
MIN_TRACK_CONF:      float = 0.75   # tracking confidence below this → UNKNOWN
EXTENDED_ANGLE:      float = 160.0  # degrees — finger counts as extended
CURLED_ANGLE:        float = 90.0   # degrees — finger counts as curled
CENTER_ZONE:  Tuple[float, float] = (0.30, 0.70)  # normalised x & y bounds
MIN_HAND_AREA_RATIO: float = 0.03   # fraction of frame area the hand box must cover


# ---------------------------------------------------------------------------
# Step 1 & 2: RawClassifier — single-frame, stateless
# ---------------------------------------------------------------------------

class RawClassifier:
    """
    Classifies one hand's landmark set as OPEN, FIST, or UNKNOWN.

    Uses angle-at-PIP for all four fingers and handedness-aware
    threshold logic for the thumb.  Entirely stateless.
    """

    # (MCP_idx, PIP_idx, TIP_idx) for each of the four fingers
    _FINGER_JOINTS = [
        (5,  6,  8),   # Index
        (9,  10, 12),  # Middle
        (13, 14, 16),  # Ring
        (17, 18, 20),  # Pinky
    ]

    # ── Geometry helpers ───────────────────────────────────────────────

    @staticmethod
    def _angle(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Angle in degrees between two 2-D vectors (clamped to [0°, 180°])."""
        dot  = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 * mag2 < 1e-9:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))

    @classmethod
    def _finger_extended(cls, lm, mcp: int, pip: int, tip: int) -> bool:
        """
        True if the PIP joint angle indicates extension.

        Vectors emanate from PIP:
          v_prox  = PIP → MCP  (towards palm)
          v_distal = PIP → TIP  (towards fingertip)
        A large angle (~180°) means the finger is straight.
        """
        v_prox   = (lm[mcp].x - lm[pip].x, lm[mcp].y - lm[pip].y)
        v_distal = (lm[tip].x - lm[pip].x, lm[tip].y - lm[pip].y)
        return cls._angle(v_prox, v_distal) > EXTENDED_ANGLE

    # ── Public interface ───────────────────────────────────────────────

    @classmethod
    def classify(cls, lm, is_right_hand: bool) -> str:
        """
        Parameters
        ----------
        lm           : hand_landmarks.landmark sequence
        is_right_hand: True for the user's right hand
                       (MediaPipe labels from the camera's perspective —
                        i.e. 'Right' in the API = user's left on a mirrored feed.
                        Callers that flip the frame should flip the label too.)

        Returns
        -------
        'OPEN', 'FIST', or 'UNKNOWN'
        """
        extended_flags = [
            cls._finger_extended(lm, mcp, pip, tip)
            for mcp, pip, tip in cls._FINGER_JOINTS
        ]

        # Thumb: use horizontal position relative to IP joint (lm[3])
        # Right hand (camera mirrored) → extended if tip.x < ip.x
        # Left  hand                   → extended if tip.x > ip.x
        if is_right_hand:
            thumb_extended = lm[4].x < lm[3].x
        else:
            thumb_extended = lm[4].x > lm[3].x
        extended_flags.append(thumb_extended)

        count = sum(extended_flags)
        if count >= 4:   return "OPEN"
        if count <= 1:   return "FIST"
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Step 3, 4 & 5: Stabilizer — time-based, UNKNOWN-blind
# ---------------------------------------------------------------------------

class Stabilizer:
    """
    Confirms a raw classification only after it remains the same for
    at least STABLE_DURATION seconds.

    Rules
    -----
    • UNKNOWN passes through as None — it is never "confirmed"
      and always resets the candidate timer.
    • Switching candidate resets the clock.
    • Returns the confirmed state once the window elapses.
    """

    def __init__(self, stable_duration: float = STABLE_DURATION) -> None:
        self._dur   = stable_duration
        self._cand: Optional[str] = None
        self._start: float        = 0.0

    def reset(self) -> None:
        """Hard reset — called when no hand is in the frame."""
        self._cand  = None
        self._start = 0.0

    def update(self, raw: str) -> Optional[str]:
        """
        Feed one raw classification.

        Returns the confirmed state string, or None if still unstable.
        """
        # Improvement 3: UNKNOWN never enters history; resets candidate
        if raw == "UNKNOWN":
            self.reset()
            return None

        now = time.monotonic()
        if raw != self._cand:
            self._cand  = raw
            self._start = now
            return None

        # Improvement 5: time-based, not frame-count-based
        if now - self._start >= self._dur:
            return self._cand

        return None


# ---------------------------------------------------------------------------
# Step 4 & 8: TransitionDetector — separate cooldowns + neutral lock
# ---------------------------------------------------------------------------

class TransitionDetector:
    """
    Converts confirmed stable state changes into named actions.

    State machine
    -------------
      OPEN  →  FIST  :  SEND
      FIST  →  OPEN  :  RECEIVE

    Safety mechanisms
    -----------------
    • Separate cooldowns for SEND and RECEIVE (Improvement 8)
    • Neutral lock after every trigger (Improvement 4):
        After firing, the detector ignores all input until the hand
        has been absent / UNKNOWN for at least NEUTRAL_HOLD_TIME seconds.
        This prevents unintended back-to-back gestures.
    """

    def __init__(
        self,
        send_cooldown:    float = SEND_COOLDOWN,
        receive_cooldown: float = RECEIVE_COOLDOWN,
        neutral_hold:     float = NEUTRAL_HOLD_TIME,
    ) -> None:
        self._send_cd    = send_cooldown
        self._recv_cd    = receive_cooldown
        self._neut_hold  = neutral_hold

        self._prev:          Optional[str] = None
        self._last_send:     float = 0.0
        self._last_recv:     float = 0.0
        self._wait_neutral:  bool  = False
        self._neutral_since: float = 0.0   # monotonic timestamp of neutral start

    # ── Called when no hand / UNKNOWN is seen ─────────────────────────

    def feed_neutral(self) -> None:
        """
        Inform the detector that the hand is absent or unclassifiable.
        Accumulates neutral time to release the post-trigger lock.
        """
        if not self._wait_neutral:
            return
        now = time.monotonic()
        if self._neutral_since == 0.0:
            self._neutral_since = now
        elif now - self._neutral_since >= self._neut_hold:
            # Neutral lock lifted
            self._wait_neutral  = False
            self._neutral_since = 0.0
            self._prev          = None

    # ── Called when a stable non-UNKNOWN state is confirmed ───────────

    def update(self, confirmed: str) -> Optional[str]:
        """
        Returns an action string ('SEND' or 'RECEIVE'), or None.
        """
        # Reset neutral-time counter: we have an active hand state
        self._neutral_since = 0.0

        if self._wait_neutral:
            return None

        now    = time.monotonic()
        action = None

        if self._prev == "OPEN" and confirmed == "FIST":
            if now - self._last_send >= self._send_cd:
                action = "SEND"
                self._last_send = now

        elif self._prev == "FIST" and confirmed == "OPEN":
            if now - self._last_recv >= self._recv_cd:
                action = "RECEIVE"
                self._last_recv = now

        if action:
            # Improvement 4: engage neutral lock immediately after trigger
            self._wait_neutral  = True
            self._neutral_since = 0.0
            self._prev          = None
        else:
            self._prev = confirmed

        return action


# ---------------------------------------------------------------------------
# Step 6–10: GestureDetector — full pipeline with spatial guards
# ---------------------------------------------------------------------------

class GestureDetector:
    """
    End-to-end gesture pipeline.

    Composes
    --------
    RawClassifier  +  Stabilizer  +  TransitionDetector

    Adds
    ----
    • Center-zone constraint  (Improvement 6)
    • Minimum hand-area guard (Improvement 7)
    • Tracking-confidence filter (Improvement 9)
    • Landmark drawing + OSD annotation

    Usage
    -----
    detector = GestureDetector()
    annotated_frame, action = detector.process_frame(frame)
    # action is 'SEND', 'RECEIVE', or None
    """

    def __init__(
        self,
        send_cooldown:    float         = SEND_COOLDOWN,
        receive_cooldown: float         = RECEIVE_COOLDOWN,
        stable_duration:  float         = STABLE_DURATION,
        center_zone:      Tuple[float, float] = CENTER_ZONE,
        min_hand_area:    float         = MIN_HAND_AREA_RATIO,
        min_track_conf:   float         = MIN_TRACK_CONF,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._stabilizer  = Stabilizer(stable_duration)
        self._transitions = TransitionDetector(send_cooldown, receive_cooldown)
        self._center_zone = center_zone
        self._min_area    = min_hand_area
        self._min_conf    = min_track_conf

    # ── Spatial guards ─────────────────────────────────────────────────

    def _in_center_zone(self, lm, w: int, h: int) -> bool:
        """True if the hand bounding-box centre falls in the centre 40% of frame."""
        xs  = [p.x for p in lm]
        ys  = [p.y for p in lm]
        cx  = (min(xs) + max(xs)) / 2
        cy  = (min(ys) + max(ys)) / 2
        lo, hi = self._center_zone
        return lo <= cx <= hi and lo <= cy <= hi

    def _large_enough(self, lm, w: int, h: int) -> bool:
        """True if the hand occupies at least MIN_HAND_AREA_RATIO of the frame."""
        xs_px = [p.x * w for p in lm]
        ys_px = [p.y * h for p in lm]
        box_area   = (max(xs_px) - min(xs_px)) * (max(ys_px) - min(ys_px))
        frame_area = w * h
        return frame_area > 0 and (box_area / frame_area) >= self._min_area

    # ── Main entry point ───────────────────────────────────────────────

    def process_frame(self, frame) -> Tuple:
        """
        Run the full detection pipeline on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray  — OpenCV BGR frame (may be pre-flipped)

        Returns
        -------
        (annotated_frame, action)
        action : 'SEND' | 'RECEIVE' | None
        """
        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result  = self.hands.process(rgb)

        action = None

        if result.multi_hand_landmarks:
            landmarks  = result.multi_hand_landmarks[0]
            lm         = landmarks.landmark

            # Always draw the skeleton regardless of guard results
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
            )

            # ── Improvement 6: center-zone guard ────────────────────────
            if not self._in_center_zone(lm, w, h):
                self._stabilizer.reset()
                self._transitions.feed_neutral()
                cv2.putText(frame, "ZONE: out of center",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (80, 80, 255), 1)
                return frame, None

            # ── Improvement 7: size guard ────────────────────────────────
            if not self._large_enough(lm, w, h):
                self._stabilizer.reset()
                self._transitions.feed_neutral()
                cv2.putText(frame, "ZONE: hand too small",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (80, 80, 255), 1)
                return frame, None

            # ── Improvement 9: confidence filter ────────────────────────
            if result.multi_handedness:
                conf     = result.multi_handedness[0].classification[0].score
                label    = result.multi_handedness[0].classification[0].label
                is_right = (label == "Right")
                if conf < self._min_conf:
                    self._stabilizer.reset()
                    self._transitions.feed_neutral()
                    cv2.putText(frame, f"CONF: low ({conf:.2f})",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (80, 80, 255), 1)
                    return frame, None
            else:
                is_right = True   # safe default

            # ── Improvement 1 & 2: classify using angles + handedness ───
            raw_state       = RawClassifier.classify(lm, is_right)

            # ── Improvements 3 & 5: time-based stabilisation ────────────
            confirmed_state = self._stabilizer.update(raw_state)

            # ── Improvements 4 & 8: transitions + neutral lock ──────────
            if confirmed_state:
                action = self._transitions.update(confirmed_state)

            # ── OSD ─────────────────────────────────────────────────────
            osd_state = confirmed_state or raw_state
            color     = (0, 255, 0) if confirmed_state else (200, 200, 0)
            cv2.putText(frame, f"STATE: {osd_state}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if action:
                cv2.putText(frame, f"ACTION: {action}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # No hand in frame — feed neutral to both sub-systems
            self._stabilizer.reset()
            self._transitions.feed_neutral()

        return frame, action

    # ── Standalone blocking loop ───────────────────────────────────────

    def run(self) -> None:
        """
        Blocking standalone loop for development / smoke-testing.
        Press 'q' to quit.
        """
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame         = cv2.flip(frame, 1)
                frame, action = self.process_frame(frame)
                if action:
                    print(f"[GestureDetector] ▶  {action}")
                cv2.imshow("GestureDrop — Gesture Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
