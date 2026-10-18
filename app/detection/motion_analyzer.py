"""
Motion Analyzer, Gesture Detector, and Action Classifier for Motion Reader.
Analyzes 2D/3D pose landmarks across time to recognize body states,
arm movements, gestures, and body motions with real confidence scores.
"""
import math
import time
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from app.detection.pose_estimator import (
    TrackingResult, NormalizedLandmark,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
)


def calculate_distance(p1: NormalizedLandmark, p2: NormalizedLandmark) -> float:
    """Euclidean distance in normalized 2D space."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def calculate_angle(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) -> float:
    """
    Calculates angle at vertex b between segments ba and bc in degrees (0 - 180).
    """
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 * norm2 < 1e-6:
        return 180.0
    cosine = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cosine))


@dataclass
class FrameAnalysisResult:
    person_detected: bool = False
    person_id: Optional[int] = None
    subject_name: Optional[str] = None
    subject_confidence: Optional[float] = None
    posture: Optional[str] = None                    # "Standing", "Sitting", "Crouching"
    left_hand_raised: bool = False
    right_hand_raised: bool = False
    both_hands_raised: bool = False
    gestures: List[Dict[str, Any]] = None            # e.g. [{"name": "Waving", "confidence": 0.92}]
    body_movements: List[Dict[str, Any]] = None      # e.g. [{"name": "Stepping forward", "confidence": 0.87}]
    fps: float = 0.0

    def __post_init__(self):
        if self.gestures is None:
            self.gestures = []
        if self.body_movements is None:
            self.body_movements = []


class MotionAnalyzer:
    """
    Stateful temporal analyzer that tracks pose landmarks across successive frames
    to extract reliable actions, gestures, and posture changes.
    """
    def __init__(self, sensitivity: float = 1.0, history_length: int = 45):
        self.sensitivity = max(0.5, min(2.0, sensitivity))
        self.history_length = history_length

        # Temporal rolling buffers
        self.history_landmarks = deque(maxlen=history_length)
        self.history_timestamps = deque(maxlen=history_length)
        self.history_scales = deque(maxlen=history_length)
        self.posture_buffer = deque(maxlen=5)

        # Hysteresis state for hand raise
        self.left_hand_is_raised = False
        self.right_hand_is_raised = False

        # Waving detector state
        self.left_wrist_x_history = deque(maxlen=24)
        self.right_wrist_x_history = deque(maxlen=24)

        # Step detector cooldown
        self.last_step_time = 0.0

    def reset(self) -> None:
        """Resets all temporal history buffers."""
        self.history_landmarks.clear()
        self.history_timestamps.clear()
        self.history_scales.clear()
        self.posture_buffer.clear()
        self.left_hand_is_raised = False
        self.right_hand_is_raised = False
        self.left_wrist_x_history.clear()
        self.right_wrist_x_history.clear()
        self.last_step_time = 0.0

    def analyze(self, tracking_result: TrackingResult, current_timestamp: float) -> FrameAnalysisResult:
        """
        Analyzes a single tracking result in temporal context.
        """
        result = FrameAnalysisResult()

        if not tracking_result.person_detected or not tracking_result.pose_landmarks:
            self.reset()
            return result

        lms = tracking_result.pose_landmarks
        if len(lms) < 33:
            return result

        result.person_detected = True
        result.person_id = tracking_result.person_id

        # Store in rolling buffer
        self.history_landmarks.append(lms)
        self.history_timestamps.append(current_timestamp)

        # Compute key anatomical landmarks
        l_sh = lms[LEFT_SHOULDER]
        r_sh = lms[RIGHT_SHOULDER]
        l_el = lms[LEFT_ELBOW]
        r_el = lms[RIGHT_ELBOW]
        l_wr = lms[LEFT_WRIST]
        r_wr = lms[RIGHT_WRIST]
        l_hip = lms[LEFT_HIP]
        r_hip = lms[RIGHT_HIP]
        l_knee = lms[LEFT_KNEE]
        r_knee = lms[RIGHT_KNEE]
        l_ank = lms[LEFT_ANKLE]
        r_ank = lms[RIGHT_ANKLE]
        nose = lms[NOSE]

        # Geometry metrics
        mid_shoulder_x = (l_sh.x + r_sh.x) / 2.0
        mid_shoulder_y = (l_sh.y + r_sh.y) / 2.0
        mid_hip_x = (l_hip.x + r_hip.x) / 2.0
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0

        torso_height = math.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y)
        shoulder_width = math.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
        body_scale = max(0.05, max(torso_height, shoulder_width))
        self.history_scales.append(body_scale)

        # -------------------------------------------------------------
        # 1. Posture Classification (Standing / Sitting / Crouching)
        # -------------------------------------------------------------
        raw_posture = self._classify_posture(lms, torso_height)
        self.posture_buffer.append(raw_posture)
        # Majority voting over recent frames for stability
        if len(self.posture_buffer) >= 3:
            posture_counts = {}
            for p in self.posture_buffer:
                posture_counts[p] = posture_counts.get(p, 0) + 1
            result.posture = max(posture_counts.items(), key=lambda item: item[1])[0]
        else:
            result.posture = raw_posture

        # -------------------------------------------------------------
        # 2. Hand and Arm State Detection with Hysteresis
        # -------------------------------------------------------------
        # In normalized coords, y=0 is top of image, y=1 is bottom.
        # Hand is raised when wrist y is significantly above shoulder y.
        raise_threshold = 0.03 * self.sensitivity
        lower_threshold = 0.04 / self.sensitivity

        # Left hand raise check
        if l_wr.visibility > 0.4 and l_sh.visibility > 0.4:
            if not self.left_hand_is_raised:
                if l_wr.y < l_sh.y - raise_threshold:
                    self.left_hand_is_raised = True
            else:
                if l_wr.y > l_sh.y + lower_threshold:
                    self.left_hand_is_raised = False

        # Right hand raise check
        if r_wr.visibility > 0.4 and r_sh.visibility > 0.4:
            if not self.right_hand_is_raised:
                if r_wr.y < r_sh.y - raise_threshold:
                    self.right_hand_is_raised = True
            else:
                if r_wr.y > r_sh.y + lower_threshold:
                    self.right_hand_is_raised = False

        result.left_hand_raised = self.left_hand_is_raised
        result.right_hand_raised = self.right_hand_is_raised
        result.both_hands_raised = self.left_hand_is_raised and self.right_hand_is_raised

        # -------------------------------------------------------------
        # 3. Gesture Detection (Waving)
        # -------------------------------------------------------------
        # Track wrist relative to shoulder x
        self.left_wrist_x_history.append((l_wr.x - l_sh.x, l_wr.y))
        self.right_wrist_x_history.append((r_wr.x - r_sh.x, r_wr.y))

        waving_detected, waving_conf = self._check_waving()
        if waving_detected:
            result.gestures.append({
                "name": "Waving",
                "confidence": round(waving_conf, 2)
            })

        # -------------------------------------------------------------
        # 4. Body Movements (Leaning, Turning, Stepping)
        # -------------------------------------------------------------
        # A. Leaning Left / Right
        spine_dx = mid_shoulder_x - mid_hip_x
        spine_dy = mid_shoulder_y - mid_hip_y
        lean_angle = math.degrees(math.atan2(spine_dx, -spine_dy))  # angle from vertical

        lean_threshold = 13.0 / self.sensitivity
        if lean_angle > lean_threshold:
            conf = min(0.96, 0.80 + (lean_angle - lean_threshold) / 25.0)
            result.body_movements.append({
                "name": "Leaning right",
                "confidence": round(conf, 2)
            })
        elif lean_angle < -lean_threshold:
            conf = min(0.96, 0.80 + (abs(lean_angle) - lean_threshold) / 25.0)
            result.body_movements.append({
                "name": "Leaning left",
                "confidence": round(conf, 2)
            })

        # B. Turning Left / Right
        # When person turns their body, nose shifts relative to shoulders and shoulder span foreshortens
        if shoulder_width > 0.05 and torso_height > 0.05:
            rel_nose_x = (nose.x - mid_shoulder_x) / shoulder_width
            turn_threshold = 0.28 / self.sensitivity
            if rel_nose_x > turn_threshold:
                conf = min(0.95, 0.82 + (rel_nose_x - turn_threshold) * 0.4)
                result.body_movements.append({
                    "name": "Turning right",
                    "confidence": round(conf, 2)
                })
            elif rel_nose_x < -turn_threshold:
                conf = min(0.95, 0.82 + (abs(rel_nose_x) - turn_threshold) * 0.4)
                result.body_movements.append({
                    "name": "Turning left",
                    "confidence": round(conf, 2)
                })

        # C. Stepping Forward / Backward
        if len(self.history_scales) >= 18 and current_timestamp - self.last_step_time > 1.8:
            scales = list(self.history_scales)
            # Compare average scale of first 6 frames vs last 6 frames
            past_scale = sum(scales[:6]) / 6.0
            recent_scale = sum(scales[-6:]) / 6.0
            scale_ratio = recent_scale / max(1e-4, past_scale)

            step_fwd_threshold = 1.08 / self.sensitivity
            step_bwd_threshold = 0.92 * self.sensitivity

            if scale_ratio > step_fwd_threshold:
                self.last_step_time = current_timestamp
                conf = min(0.94, 0.82 + (scale_ratio - step_fwd_threshold) * 0.8)
                result.body_movements.append({
                    "name": "Stepping forward",
                    "confidence": round(conf, 2)
                })
            elif scale_ratio < step_bwd_threshold:
                self.last_step_time = current_timestamp
                conf = min(0.94, 0.82 + (step_bwd_threshold - scale_ratio) * 0.8)
                result.body_movements.append({
                    "name": "Stepping backward",
                    "confidence": round(conf, 2)
                })

        return result

    def _classify_posture(self, lms: List[NormalizedLandmark], torso_height: float) -> str:
        """
        Calculates knee angles and hip-to-ankle relations to determine Standing vs Sitting vs Crouching.
        """
        l_hip, r_hip = lms[LEFT_HIP], lms[RIGHT_HIP]
        l_knee, r_knee = lms[LEFT_KNEE], lms[RIGHT_KNEE]
        l_ank, r_ank = lms[LEFT_ANKLE], lms[RIGHT_ANKLE]

        knee_angles = []
        if l_hip.visibility > 0.4 and l_knee.visibility > 0.4 and l_ank.visibility > 0.4:
            knee_angles.append(calculate_angle(l_hip, l_knee, l_ank))
        if r_hip.visibility > 0.4 and r_knee.visibility > 0.4 and r_ank.visibility > 0.4:
            knee_angles.append(calculate_angle(r_hip, r_knee, r_ank))

        # If legs are not visible (e.g. upper-body only webcam view)
        if not knee_angles:
            # Upper body view: if torso is upright and head is in normal position, defaults to Standing
            return "Standing"

        avg_knee_angle = sum(knee_angles) / len(knee_angles)

        # Crouching: very deep knee bend, hips very low
        if avg_knee_angle < 85.0:
            return "Crouching"

        # Sitting: knee bent below 145 degrees
        if avg_knee_angle < 145.0:
            return "Sitting"

        # Otherwise upright standing
        return "Standing"

    def _check_waving(self) -> Tuple[bool, float]:
        """
        Detects lateral oscillatory waving motion of a raised wrist.
        Returns (is_waving, confidence).
        """
        # Check right hand
        r_wave, r_conf = self._eval_wrist_oscillation(self.right_wrist_x_history, self.right_hand_is_raised)
        if r_wave:
            return True, r_conf

        # Check left hand
        l_wave, l_conf = self._eval_wrist_oscillation(self.left_wrist_x_history, self.left_hand_is_raised)
        if l_wave:
            return True, l_conf

        return False, 0.0

    def _eval_wrist_oscillation(self, history: deque, hand_raised: bool) -> Tuple[bool, float]:
        """
        Evaluates lateral velocity zero-crossings and amplitude.
        """
        if not hand_raised or len(history) < 12:
            return False, 0.0

        xs = [item[0] for item in history]
        amplitude = max(xs) - min(xs)

        # Minimum lateral amplitude required for waving
        if amplitude < 0.035 / self.sensitivity:
            return False, 0.0

        # Calculate velocities between consecutive frames
        velocities = [xs[i] - xs[i-1] for i in range(1, len(xs))]

        # Count directional reversals robustly
        reversals = 0
        min_vel = 0.002
        prev_dir = 0
        for v in velocities:
            if v > min_vel:
                curr_dir = 1
            elif v < -min_vel:
                curr_dir = -1
            else:
                continue

            if prev_dir != 0 and curr_dir != prev_dir:
                reversals += 1
            prev_dir = curr_dir

        if reversals >= 2:
            # Good wave pattern detected
            conf = min(0.96, 0.85 + (reversals - 2) * 0.03 + min(0.08, amplitude * 0.8))
            return True, conf

        return False, 0.0
