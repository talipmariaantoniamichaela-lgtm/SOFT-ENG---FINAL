"""
Movement classification engine for VISION AI.
Implements temporal state machines for action detection.
"""
import math
import time
from collections import deque
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field

from app.detectors.vision_detector import (
    PersonDetection, Landmark,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    NOSE,
)


class MovementState(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


@dataclass
class DetectedAction:
    category: str          # ACTION | POSTURE | GESTURE | FACE | HAND
    action: str            # snake_case identifier
    description: str       # Human-readable
    confidence: float
    source: str = "movement_classifier"
    person_id: int = 1
    severity: str = "info"


@dataclass
class MovementContext:
    """Rolling state for one person's movement analysis."""
    person_id: int = 1
    # Landmark history
    pose_history: deque = field(default_factory=lambda: deque(maxlen=60))
    ts_history: deque = field(default_factory=lambda: deque(maxlen=60))
    scale_history: deque = field(default_factory=lambda: deque(maxlen=60))

    # Arm raise state (hysteresis)
    left_arm_raised: bool = False
    right_arm_raised: bool = False

    # Waving
    l_wrist_x_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    r_wrist_x_hist: deque = field(default_factory=lambda: deque(maxlen=30))

    # Posture
    posture_buffer: deque = field(default_factory=lambda: deque(maxlen=7))
    current_posture: str = "Unknown"

    # Head
    head_state: str = "forward"     # forward | left | right | up | down
    prev_head_state: str = "forward"
    head_history: deque = field(default_factory=lambda: deque(maxlen=15))

    # Face yaw/pitch history
    yaw_history: deque = field(default_factory=lambda: deque(maxlen=20))
    pitch_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Eye / mouth state
    mouth_open: bool = False
    eyes_closed: bool = False

    # Cooldowns (timestamps of last emission per action)
    cooldowns: Dict[str, float] = field(default_factory=dict)

    # Step / walk detection
    ankle_y_history: deque = field(default_factory=lambda: deque(maxlen=40))

    def cooldown_ok(self, action: str, cooldown_s: float, now: float) -> bool:
        last = self.cooldowns.get(action, -99999.0)
        if now - last >= cooldown_s:
            self.cooldowns[action] = now
            return True
        return False


class MovementClassifier:
    """
    Stateful temporal movement classifier.
    Maintains per-person MovementContext and emits DetectedAction events.
    """

    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = max(0.5, min(2.0, sensitivity))
        self._contexts: Dict[int, MovementContext] = {}

    def reset(self):
        self._contexts.clear()

    def _ctx(self, person_id: int) -> MovementContext:
        if person_id not in self._contexts:
            self._contexts[person_id] = MovementContext(person_id=person_id)
        return self._contexts[person_id]

    def classify(
        self,
        person: PersonDetection,
        timestamp_s: float,
    ) -> List[DetectedAction]:
        """
        Analyse a PersonDetection at the current timestamp.
        Returns a list of newly detected actions (may be empty).
        """
        actions: List[DetectedAction] = []
        ctx = self._ctx(person.person_id)
        now = timestamp_s
        lms = person.pose_landmarks

        if len(lms) < 25:
            return actions

        # Extract key landmarks
        l_sh = lms[LEFT_SHOULDER]
        r_sh = lms[RIGHT_SHOULDER]
        l_el = lms[LEFT_ELBOW]
        r_el = lms[RIGHT_ELBOW]
        l_wr = lms[LEFT_WRIST]
        r_wr = lms[RIGHT_WRIST]
        l_hip = lms[LEFT_HIP]
        r_hip = lms[RIGHT_HIP]
        l_kn = lms[LEFT_KNEE]
        r_kn = lms[RIGHT_KNEE]
        l_ank = lms[LEFT_ANKLE]
        r_ank = lms[RIGHT_ANKLE]
        nose = lms[NOSE]

        # Geometry helpers
        mid_sh_x = (l_sh.x + r_sh.x) / 2
        mid_sh_y = (l_sh.y + r_sh.y) / 2
        mid_hip_x = (l_hip.x + r_hip.x) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        sh_width = math.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
        torso_h = math.hypot(mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y)
        body_scale = max(0.05, max(sh_width, torso_h))

        ctx.pose_history.append(lms)
        ctx.ts_history.append(now)
        ctx.scale_history.append(body_scale)

        pid = f"Person {person.person_id:02d}"

        # ── 1. Posture ─────────────────────────────────────────────────────
        posture = self._classify_posture(lms, torso_h)
        ctx.posture_buffer.append(posture)
        if len(ctx.posture_buffer) >= 5:
            counts: Dict[str, int] = {}
            for p in ctx.posture_buffer:
                counts[p] = counts.get(p, 0) + 1
            stable_posture = max(counts, key=lambda k: counts[k])
            if stable_posture != ctx.current_posture:
                ctx.current_posture = stable_posture
                if ctx.cooldown_ok("posture", 2.0, now):
                    conf = min(0.95, counts[stable_posture] / len(ctx.posture_buffer) + 0.3)
                    actions.append(DetectedAction(
                        category="POSTURE",
                        action=f"posture_{stable_posture.lower()}",
                        description=f"{pid} is {stable_posture.lower()}.",
                        confidence=round(conf, 2),
                        person_id=person.person_id,
                    ))

        # ── 2. Arm Raise ───────────────────────────────────────────────────
        raise_thr = 0.04 / self.sensitivity
        lower_thr = 0.05 * self.sensitivity

        # Left arm
        if l_wr.visibility > 0.4 and l_sh.visibility > 0.4:
            if not ctx.left_arm_raised:
                if l_wr.y < l_sh.y - raise_thr:
                    ctx.left_arm_raised = True
                    if ctx.cooldown_ok("left_arm_raise", 1.5, now):
                        actions.append(DetectedAction(
                            category="ACTION",
                            action="left_arm_raised",
                            description=f"{pid} raises their left arm.",
                            confidence=0.88,
                            person_id=person.person_id,
                        ))
            else:
                if l_wr.y > l_sh.y + lower_thr:
                    ctx.left_arm_raised = False
                    if ctx.cooldown_ok("left_arm_lower", 1.5, now):
                        actions.append(DetectedAction(
                            category="ACTION",
                            action="left_arm_lowered",
                            description=f"{pid} lowers their left arm.",
                            confidence=0.85,
                            person_id=person.person_id,
                        ))

        # Right arm
        if r_wr.visibility > 0.4 and r_sh.visibility > 0.4:
            if not ctx.right_arm_raised:
                if r_wr.y < r_sh.y - raise_thr:
                    ctx.right_arm_raised = True
                    if ctx.cooldown_ok("right_arm_raise", 1.5, now):
                        actions.append(DetectedAction(
                            category="ACTION",
                            action="right_arm_raised",
                            description=f"{pid} raises their right arm.",
                            confidence=0.88,
                            person_id=person.person_id,
                        ))
            else:
                if r_wr.y > r_sh.y + lower_thr:
                    ctx.right_arm_raised = False
                    if ctx.cooldown_ok("right_arm_lower", 1.5, now):
                        actions.append(DetectedAction(
                            category="ACTION",
                            action="right_arm_lowered",
                            description=f"{pid} lowers their right arm.",
                            confidence=0.85,
                            person_id=person.person_id,
                        ))

        # Both arms
        if ctx.left_arm_raised and ctx.right_arm_raised:
            if ctx.cooldown_ok("both_arms", 3.0, now):
                actions.append(DetectedAction(
                    category="ACTION",
                    action="both_arms_raised",
                    description=f"{pid} raises both arms.",
                    confidence=0.91,
                    person_id=person.person_id,
                ))

        # ── 3. Waving Detection ────────────────────────────────────────────
        ctx.r_wrist_x_hist.append(r_wr.x - r_sh.x)
        ctx.l_wrist_x_hist.append(l_wr.x - l_sh.x)

        wave, wave_conf = self._detect_wave(ctx.r_wrist_x_hist, ctx.right_arm_raised)
        if not wave:
            wave, wave_conf = self._detect_wave(ctx.l_wrist_x_hist, ctx.left_arm_raised)
        if wave and ctx.cooldown_ok("waving", 2.5, now):
            actions.append(DetectedAction(
                category="GESTURE",
                action="waving",
                description=f"{pid} appears to be waving.",
                confidence=round(wave_conf, 2),
                person_id=person.person_id,
            ))

        # ── 4. Lean Detection ──────────────────────────────────────────────
        spine_dx = mid_sh_x - mid_hip_x
        spine_dy = mid_sh_y - mid_hip_y
        lean_deg = math.degrees(math.atan2(spine_dx, -spine_dy))
        lean_thr = 12.0 / self.sensitivity

        if lean_deg > lean_thr and ctx.cooldown_ok("lean", 2.0, now):
            conf = min(0.93, 0.78 + (lean_deg - lean_thr) / 30)
            actions.append(DetectedAction(
                category="ACTION",
                action="leaning_right",
                description=f"{pid} leans to the right.",
                confidence=round(conf, 2),
                person_id=person.person_id,
            ))
        elif lean_deg < -lean_thr and ctx.cooldown_ok("lean", 2.0, now):
            conf = min(0.93, 0.78 + (abs(lean_deg) - lean_thr) / 30)
            actions.append(DetectedAction(
                category="ACTION",
                action="leaning_left",
                description=f"{pid} leans to the left.",
                confidence=round(conf, 2),
                person_id=person.person_id,
            ))

        # ── 5. Body Turn ───────────────────────────────────────────────────
        if sh_width > 0.04 and torso_h > 0.04:
            rel_nose = (nose.x - mid_sh_x) / sh_width
            turn_thr = 0.30 / self.sensitivity
            if rel_nose > turn_thr and ctx.cooldown_ok("body_turn", 2.0, now):
                conf = min(0.92, 0.80 + (rel_nose - turn_thr) * 0.5)
                actions.append(DetectedAction(
                    category="ACTION",
                    action="turning_right",
                    description=f"{pid} turns to the right.",
                    confidence=round(conf, 2),
                    person_id=person.person_id,
                ))
            elif rel_nose < -turn_thr and ctx.cooldown_ok("body_turn", 2.0, now):
                conf = min(0.92, 0.80 + (abs(rel_nose) - turn_thr) * 0.5)
                actions.append(DetectedAction(
                    category="ACTION",
                    action="turning_left",
                    description=f"{pid} turns to the left.",
                    confidence=round(conf, 2),
                    person_id=person.person_id,
                ))

        # ── 6. Approach / Retreat (scale change) ──────────────────────────
        if len(ctx.scale_history) >= 20 and ctx.cooldown_ok("approach", 2.5, now):
            scales = list(ctx.scale_history)
            past = sum(scales[:7]) / 7
            recent = sum(scales[-7:]) / 7
            ratio = recent / max(1e-5, past)
            if ratio > 1.10 / self.sensitivity:
                conf = min(0.90, 0.78 + (ratio - 1.10) * 2.0)
                actions.append(DetectedAction(
                    category="ACTION",
                    action="stepping_forward",
                    description=f"{pid} steps forward.",
                    confidence=round(conf, 2),
                    person_id=person.person_id,
                ))
            elif ratio < 0.90 * self.sensitivity:
                conf = min(0.90, 0.78 + (0.90 - ratio) * 2.0)
                actions.append(DetectedAction(
                    category="ACTION",
                    action="stepping_backward",
                    description=f"{pid} steps backward.",
                    confidence=round(conf, 2),
                    person_id=person.person_id,
                ))

        # ── 7. Head / Face events ──────────────────────────────────────────
        if person.face is not None:
            face_actions = self._classify_face(person.face, ctx, pid, now)
            actions.extend(face_actions)

        # ── 8. Hand gestures ──────────────────────────────────────────────
        for hand in [person.left_hand, person.right_hand]:
            if hand and hand.gesture and ctx.cooldown_ok(f"gesture_{hand.gesture}", 3.0, now):
                side = hand.handedness.lower()
                actions.append(DetectedAction(
                    category="HAND",
                    action=f"hand_{hand.gesture.lower().replace(' ', '_')}",
                    description=f"{pid} shows {hand.gesture} with their {side} hand.",
                    confidence=round(hand.confidence * 0.9, 2),
                    source="hand_gesture_classifier",
                    person_id=person.person_id,
                ))

        return actions

    def _classify_posture(self, lms: List[Landmark], torso_h: float) -> str:
        l_hip, r_hip = lms[LEFT_HIP], lms[RIGHT_HIP]
        l_kn, r_kn = lms[LEFT_KNEE], lms[RIGHT_KNEE]
        l_ank, r_ank = lms[LEFT_ANKLE], lms[RIGHT_ANKLE]

        angles = []
        if l_hip.visibility > 0.4 and l_kn.visibility > 0.4 and l_ank.visibility > 0.4:
            angles.append(self._angle(l_hip, l_kn, l_ank))
        if r_hip.visibility > 0.4 and r_kn.visibility > 0.4 and r_ank.visibility > 0.4:
            angles.append(self._angle(r_hip, r_kn, r_ank))

        if not angles:
            return "Standing"

        avg = sum(angles) / len(angles)
        if avg < 80:
            return "Crouching"
        if avg < 140:
            return "Sitting"
        return "Standing"

    def _angle(self, a: Landmark, b: Landmark, c: Landmark) -> float:
        v1 = (a.x - b.x, a.y - b.y)
        v2 = (c.x - b.x, c.y - b.y)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 * n2 < 1e-7:
            return 180.0
        cos = max(-1.0, min(1.0, dot / (n1 * n2)))
        return math.degrees(math.acos(cos))

    def _detect_wave(self, wrist_x_hist: deque, hand_raised: bool) -> Tuple[bool, float]:
        if not hand_raised or len(wrist_x_hist) < 14:
            return False, 0.0
        xs = list(wrist_x_hist)
        amp = max(xs) - min(xs)
        if amp < 0.03 / self.sensitivity:
            return False, 0.0
        vels = [xs[i] - xs[i-1] for i in range(1, len(xs))]
        reversals = 0
        prev_dir = 0
        for v in vels:
            if v > 0.002:
                d = 1
            elif v < -0.002:
                d = -1
            else:
                continue
            if prev_dir != 0 and d != prev_dir:
                reversals += 1
            prev_dir = d
        if reversals >= 2:
            conf = min(0.95, 0.82 + reversals * 0.03 + min(0.06, amp * 0.6))
            return True, conf
        return False, 0.0

    def _classify_face(
        self, face, ctx: MovementContext, pid: str, now: float
    ) -> List[DetectedAction]:
        actions = []
        yaw = face.yaw
        pitch = face.pitch

        ctx.yaw_history.append(yaw)
        ctx.pitch_history.append(pitch)

        # Determine head state
        avg_yaw = sum(ctx.yaw_history) / len(ctx.yaw_history) if ctx.yaw_history else 0
        avg_pitch = sum(ctx.pitch_history) / len(ctx.pitch_history) if ctx.pitch_history else 0

        if avg_yaw > 18:
            new_state = "right"
        elif avg_yaw < -18:
            new_state = "left"
        elif avg_pitch > 12:
            new_state = "down"
        elif avg_pitch < -12:
            new_state = "up"
        else:
            new_state = "forward"

        if new_state != ctx.head_state and ctx.cooldown_ok(f"head_{new_state}", 1.5, now):
            ctx.head_state = new_state
            desc_map = {
                "right": f"{pid} turns head to the right.",
                "left": f"{pid} turns head to the left.",
                "up": f"{pid} looks upward.",
                "down": f"{pid} looks downward.",
                "forward": f"{pid} returns head to forward position.",
            }
            actions.append(DetectedAction(
                category="FACE",
                action=f"head_{new_state}",
                description=desc_map.get(new_state, f"{pid} moves head."),
                confidence=0.82,
                source="face_landmarker",
                person_id=int(pid.split()[-1]) if pid.split() else 1,
            ))

        return actions
