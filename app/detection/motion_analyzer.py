"""
Motion Analyzer, Gesture Detector, and Action Classifier for Motion Reader.
Analyzes 2D/3D pose landmarks across time to recognize body states,
arm movements, gestures, and body motions with real confidence scores.

One MotionAnalyzer instance tracks one person; the frame processor keeps
one per tracked person.
"""
import math
import time
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from app.detection.pose_estimator import (
    TrackingResult, NormalizedLandmark,
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
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


# Sustained activities: name -> phrase used after a person's name in the terminal
ACTIVITY_PHRASES: Dict[str, str] = {
    "Idle": "is idle",
    "Walking": "is walking",
    "Running": "is running",
    "Arms crossed": "has arms crossed",
    "Hands on head": "has hands on head",
    "Touching face": "is touching face",
    "Arms stretched out": "has arms stretched out",
    "Hands on hips": "has hands on hips",
    "Pointing left": "is pointing left",
    "Pointing right": "is pointing right",
    "Bending over": "is bending over",
}

# Seconds a raw activity must hold before it is reported, and before it is dropped
ACTIVITY_ON_TIME: Dict[str, float] = {"Idle": 2.0, "Walking": 0.5, "Running": 0.4}
DEFAULT_ON_TIME = 0.35
ACTIVITY_OFF_TIME = 0.5

# Joints used to measure overall body motion
MOTION_JOINTS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                 LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]


@dataclass
class FrameAnalysisResult:
    person_detected: bool = False
    person_id: Optional[int] = None
    subject_name: Optional[str] = None
    subject_confidence: Optional[float] = None
    posture: Optional[str] = None                    # "Standing", "Sitting", "Crouching", "Lying down"
    left_hand_raised: bool = False
    right_hand_raised: bool = False
    both_hands_raised: bool = False
    gestures: List[Dict[str, Any]] = None            # momentary, e.g. [{"name": "Waving", "confidence": 0.92}]
    body_movements: List[Dict[str, Any]] = None      # momentary, e.g. [{"name": "Stepping forward", "confidence": 0.87}]
    activities: List[Dict[str, Any]] = None          # sustained, e.g. [{"name": "Arms crossed", "confidence": 0.85}]
    motion_level: float = 0.0                        # body-lengths per second
    fps: float = 0.0

    def __post_init__(self):
        if self.gestures is None:
            self.gestures = []
        if self.body_movements is None:
            self.body_movements = []
        if self.activities is None:
            self.activities = []

    def summary_labels(self) -> List[str]:
        """Short labels describing what the person is doing right now (for overlays)."""
        labels: List[str] = []
        if self.posture:
            labels.append(self.posture)
        if self.both_hands_raised:
            labels.append("Both hands up")
        elif self.right_hand_raised:
            labels.append("Right hand up")
        elif self.left_hand_raised:
            labels.append("Left hand up")
        labels.extend(a["name"] for a in self.activities)
        labels.extend(g["name"] for g in self.gestures)
        return labels


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
        self.history_head_heights = deque(maxlen=history_length)
        self.posture_buffer = deque(maxlen=5)

        # Hysteresis state for hand raise
        self.left_hand_is_raised = False
        self.right_hand_is_raised = False

        # Waving detector state
        self.left_wrist_x_history = deque(maxlen=24)
        self.right_wrist_x_history = deque(maxlen=24)

        # Step detector cooldown
        self.last_step_time = 0.0

        self._init_extended_state()

    def _init_extended_state(self) -> None:
        # Time-windowed histories: (t, value...)
        self.motion_history: deque = deque()     # (t, speed)
        self.hip_history: deque = deque()        # (t, x, y) aspect-corrected
        self.wrist_gap_history: deque = deque()  # (t, gap / shoulder width)
        self.nose_rel_history: deque = deque()   # (t, rel_x, rel_y) in shoulder widths
        self.torso_len_history: deque = deque(maxlen=90)
        self.prev_points: Optional[Dict[int, Tuple[float, float]]] = None
        self.prev_time: Optional[float] = None
        # Activity debounce: name -> [first_raw_true, last_raw_true, active]
        self.activity_state: Dict[str, List[Any]] = {}
        self.last_posture: Optional[str] = None

    def reset(self) -> None:
        """Resets all temporal history buffers."""
        self.history_landmarks.clear()
        self.history_timestamps.clear()
        self.history_scales.clear()
        self.history_head_heights.clear()
        self.posture_buffer.clear()
        self.left_hand_is_raised = False
        self.right_hand_is_raised = False
        self.left_wrist_x_history.clear()
        self.right_wrist_x_history.clear()
        self.last_step_time = 0.0
        self._init_extended_state()

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
        # Hips out of frame (desk webcam) are guessed by the model and jitter a lot,
        # so they only count toward body size when actually visible
        hips_seen = l_hip.visibility > 0.6 and r_hip.visibility > 0.6
        body_scale = max(0.05, max(torso_height, shoulder_width) if hips_seen else shoulder_width)
        self.history_scales.append(body_scale)
        # Independent size cue (head above shoulders) used to confirm stepping
        self.history_head_heights.append(max(1e-3, abs(mid_shoulder_y - nose.y)))

        # -------------------------------------------------------------
        # 1. Posture Classification (Standing / Sitting / Crouching / Lying down)
        # -------------------------------------------------------------
        raw_posture = self._classify_posture(lms, torso_height, tracking_result.image_size)
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
        # Only a small hysteresis band is needed.  The old release threshold
        # required a wrist to travel *well below* its shoulder, so a wrist that
        # was briefly lost by the camera could leave the "raised" state stuck.
        release_margin = 0.01 * self.sensitivity

        # Left hand raise check
        if l_wr.visibility > 0.4 and l_sh.visibility > 0.4:
            if not self.left_hand_is_raised:
                if l_wr.y < l_sh.y - raise_threshold:
                    self.left_hand_is_raised = True
            elif l_wr.y >= l_sh.y - release_margin:
                self.left_hand_is_raised = False
        else:
            # A state must describe the current frame, not the last good one.
            # This also clears the label when a lowered hand leaves the frame.
            self.left_hand_is_raised = False

        # Right hand raise check
        if r_wr.visibility > 0.4 and r_sh.visibility > 0.4:
            if not self.right_hand_is_raised:
                if r_wr.y < r_sh.y - raise_threshold:
                    self.right_hand_is_raised = True
            elif r_wr.y >= r_sh.y - release_margin:
                self.right_hand_is_raised = False
        else:
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
        hips_visible = l_hip.visibility > 0.4 and r_hip.visibility > 0.4
        if hips_visible and lean_threshold < abs(lean_angle) < 60.0:
            conf = min(0.96, 0.80 + (abs(lean_angle) - lean_threshold) / 25.0)
            result.body_movements.append({
                "name": "Leaning right" if lean_angle > 0 else "Leaning left",
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
            heads = list(self.history_head_heights)
            head_ratio = (sum(heads[-6:]) / 6.0) / max(1e-4, sum(heads[:6]) / 6.0)

            step_fwd_threshold = 1.08 / self.sensitivity
            step_bwd_threshold = 0.92 * self.sensitivity
            # Turning shrinks shoulder width and raising an arm shifts a shoulder;
            # a real step changes the head-above-shoulders height the same way
            arms_moving = self.left_hand_is_raised or self.right_hand_is_raised
            head_fwd = head_ratio > 1.0 + (step_fwd_threshold - 1.0) * 0.6
            head_bwd = head_ratio < 1.0 - (1.0 - step_bwd_threshold) * 0.6

            if arms_moving:
                pass
            elif scale_ratio > step_fwd_threshold and head_fwd:
                self.last_step_time = current_timestamp
                conf = min(0.94, 0.82 + (scale_ratio - step_fwd_threshold) * 0.8)
                result.body_movements.append({
                    "name": "Stepping forward",
                    "confidence": round(conf, 2)
                })
            elif scale_ratio < step_bwd_threshold and head_bwd:
                self.last_step_time = current_timestamp
                conf = min(0.94, 0.82 + (step_bwd_threshold - scale_ratio) * 0.8)
                result.body_movements.append({
                    "name": "Stepping backward",
                    "confidence": round(conf, 2)
                })

        # -------------------------------------------------------------
        # 5. Extended activities & gestures (aspect-corrected geometry)
        # -------------------------------------------------------------
        self._analyze_extended(tracking_result, current_timestamp, result)

        # Hands resting on the head are not "raised hands"
        if any(a["name"] == "Hands on head" for a in result.activities):
            self.left_hand_is_raised = False
            self.right_hand_is_raised = False
            result.left_hand_raised = result.right_hand_raised = result.both_hands_raised = False

        # Head shaking moves the nose sideways; don't report it as turning
        if any(g["name"] == "Shaking head" for g in result.gestures):
            result.body_movements = [m for m in result.body_movements if not m["name"].startswith("Turning")]

        self.last_posture = result.posture
        return result

    # =================================================================
    # Extended analysis
    # =================================================================
    def _analyze_extended(self, tr: TrackingResult, t: float, result: FrameAnalysisResult) -> None:
        lms = tr.pose_landmarks
        img_w, img_h = tr.image_size if tr.image_size else (1, 1)
        ar = (img_w / float(img_h)) if img_h else 1.0

        def P(i: int) -> Tuple[float, float]:
            return (lms[i].x * ar, lms[i].y)

        def vis(*idx: int) -> bool:
            return all(lms[i].visibility > 0.5 for i in idx)

        def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        if not vis(LEFT_SHOULDER, RIGHT_SHOULDER):
            self.prev_points = None
            return

        lsh, rsh = P(LEFT_SHOULDER), P(RIGHT_SHOULDER)
        sw = max(0.02, dist(lsh, rsh))
        mid_sh = ((lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2)
        hips_ok = vis(LEFT_HIP, RIGHT_HIP)
        lhip, rhip = P(LEFT_HIP), P(RIGHT_HIP)
        mid_hip = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
        torso = dist(mid_sh, mid_hip) if hips_ok else sw * 1.5
        if hips_ok:
            self.torso_len_history.append(torso)
        body = max(sw, torso)

        # ---- Motion energy (body lengths / second) ----
        pts = {i: P(i) for i in MOTION_JOINTS if lms[i].visibility > 0.5}
        speed = None
        if self.prev_points is not None and self.prev_time is not None:
            dt = t - self.prev_time
            common = [i for i in pts if i in self.prev_points]
            if dt > 1e-3 and common:
                disp = sum(dist(pts[i], self.prev_points[i]) for i in common) / len(common)
                speed = disp / body / dt
        self.prev_points, self.prev_time = pts, t
        if speed is not None:
            self.motion_history.append((t, speed))
        self._trim(self.motion_history, t, 1.0)
        if self.motion_history:
            result.motion_level = round(sum(s for _, s in self.motion_history) / len(self.motion_history), 3)

        raw: Dict[str, float] = {}  # activity -> confidence

        # Idle: barely moving for a sustained period
        if len(self.motion_history) >= 5 and result.motion_level < 0.35 * self.sensitivity:
            raw["Idle"] = round(max(0.80, min(0.97, 0.97 - result.motion_level)), 2)

        # Walking / running: sustained horizontal travel of the hips (or shoulders)
        anchor = mid_hip if hips_ok else mid_sh
        self.hip_history.append((t, anchor[0], anchor[1]))
        self._trim(self.hip_history, t, 1.0)
        if len(self.hip_history) >= 5:
            t0, x0, _ = self.hip_history[0]
            span = t - t0
            if span >= 0.5:
                travel = abs(anchor[0] - x0) / body / span
                if travel > 2.2 / self.sensitivity:
                    raw["Running"] = round(min(0.95, 0.80 + travel / 20.0), 2)
                elif travel > 0.55 / self.sensitivity:
                    raw["Walking"] = round(min(0.95, 0.78 + travel / 10.0), 2)
        if "Walking" in raw or "Running" in raw:
            raw.pop("Idle", None)

        # Arm geometry
        arms_ok = vis(LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW)
        nose = P(NOSE)
        lwr, rwr = P(LEFT_WRIST), P(RIGHT_WRIST)
        lel, rel = P(LEFT_ELBOW), P(RIGHT_ELBOW)
        l_el_ang = calculate_angle(lms[LEFT_SHOULDER], lms[LEFT_ELBOW], lms[LEFT_WRIST])
        r_el_ang = calculate_angle(lms[RIGHT_SHOULDER], lms[RIGHT_ELBOW], lms[RIGHT_WRIST])
        chest_top = mid_sh[1] - 0.1 * sw
        chest_bottom = mid_hip[1] if hips_ok else mid_sh[1] + 1.4 * sw

        if arms_ok:
            # Hands on head: both wrists close to the head and above the shoulders, elbows bent
            if (dist(lwr, nose) < 0.95 * sw and dist(rwr, nose) < 0.95 * sw
                    and lwr[1] < mid_sh[1] and rwr[1] < mid_sh[1]
                    and l_el_ang < 120 and r_el_ang < 120):
                raw["Hands on head"] = 0.90

            # Arms crossed: each wrist near the opposite elbow / across the body midline at chest level
            crossed_x = ((lwr[0] - mid_sh[0]) * (lsh[0] - mid_sh[0]) < 0
                         and (rwr[0] - mid_sh[0]) * (rsh[0] - mid_sh[0]) < 0)
            near_elbows = dist(lwr, rel) < 0.75 * sw and dist(rwr, lel) < 0.75 * sw
            at_chest = all(chest_top < w[1] < chest_bottom for w in (lwr, rwr))
            if at_chest and (crossed_x or near_elbows) and dist(lwr, rwr) < 1.1 * sw:
                raw["Arms crossed"] = 0.88 if (crossed_x and near_elbows) else 0.80

            # Arms stretched out sideways (T-pose)
            if (abs(lwr[1] - lsh[1]) < 0.45 * sw and abs(rwr[1] - rsh[1]) < 0.45 * sw
                    and dist(lwr, rwr) > 2.6 * sw and l_el_ang > 145 and r_el_ang > 145):
                raw["Arms stretched out"] = 0.92

            # Hands on hips: wrists at the hips, elbows bent and pointing outwards
            if hips_ok:
                out_l = abs(lel[0] - mid_sh[0]) > abs(lsh[0] - mid_sh[0]) + 0.15 * sw
                out_r = abs(rel[0] - mid_sh[0]) > abs(rsh[0] - mid_sh[0]) + 0.15 * sw
                if (dist(lwr, lhip) < 0.55 * sw and dist(rwr, rhip) < 0.55 * sw
                        and out_l and out_r and l_el_ang < 140 and r_el_ang < 140):
                    raw["Hands on hips"] = 0.86

        # Touching face: one hand at the face (not both on the head)
        if "Hands on head" not in raw:
            for wi in (LEFT_WRIST, RIGHT_WRIST):
                if lms[wi].visibility > 0.5 and dist(P(wi), nose) < 0.5 * sw:
                    raw["Touching face"] = 0.85
                    break

        # Pointing: one straight arm held out roughly horizontally, the other arm down
        for wi, ei, si, ang, other_wi in (
            (LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER, l_el_ang, RIGHT_WRIST),
            (RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, r_el_ang, LEFT_WRIST),
        ):
            if not vis(wi, ei, si):
                continue
            w, s = P(wi), P(si)
            reach = abs(w[0] - s[0])
            other_low = lms[other_wi].visibility < 0.5 or P(other_wi)[1] > mid_sh[1] + 0.5 * sw
            if (ang > 150 and reach > 1.1 * sw and abs(w[1] - s[1]) < 0.6 * sw
                    and other_low and "Arms stretched out" not in raw):
                # Screen direction; flip by mirroring if the feed is mirrored
                raw["Pointing left" if w[0] < s[0] else "Pointing right"] = 0.84
                break

        # Bending over: torso much shorter than usual while legs are straight
        if hips_ok and len(self.torso_len_history) >= 15 and result.posture in ("Standing", None):
            usual = sorted(self.torso_len_history)[int(len(self.torso_len_history) * 0.8)]
            if usual > 0 and torso < 0.55 * usual and mid_sh[1] < mid_hip[1] + 0.2 * sw:
                raw["Bending over"] = 0.82

        # ---- Momentary gestures ----
        # Clapping: wrists repeatedly coming together in front of the chest
        if arms_ok and all(chest_top - 0.6 * sw < w[1] < chest_bottom for w in (lwr, rwr)):
            self.wrist_gap_history.append((t, dist(lwr, rwr) / sw))
        self._trim(self.wrist_gap_history, t, 1.6)
        claps = self._count_closures([g for _, g in self.wrist_gap_history], close=0.35, open_=0.65)
        if claps >= 2:
            self._emit_gesture(result, "Clapping", min(0.95, 0.82 + 0.04 * claps))

        # Jumping: hips/shoulders rise sharply above the recent baseline (not standing up)
        if len(self.hip_history) >= 6 and self.last_posture not in ("Sitting", "Crouching"):
            ys = sorted(y for _, _, y in self.hip_history)
            baseline = ys[len(ys) // 2]
            rise = (baseline - anchor[1]) / body
            if rise > 0.22 / self.sensitivity:
                self._emit_gesture(result, "Jumping", min(0.95, 0.80 + rise * 0.3))

        # Nodding / shaking head: nose oscillating relative to the ears (a head-local
        # reference, so arm or shoulder movement doesn't count). Skipped while a hand
        # is up, since waving / reaching shakes the whole upper body.
        hand_up = self.left_hand_is_raised or self.right_hand_is_raised
        if vis(NOSE, LEFT_EAR, RIGHT_EAR) and not hand_up:
            lear, rear = P(LEFT_EAR), P(RIGHT_EAR)
            ear_w = max(0.01, dist(lear, rear))
            mid_ear = ((lear[0] + rear[0]) / 2, (lear[1] + rear[1]) / 2)
            self.nose_rel_history.append((t, (nose[0] - mid_ear[0]) / ear_w, (nose[1] - mid_ear[1]) / ear_w))
        else:
            self.nose_rel_history.clear()
        self._trim(self.nose_rel_history, t, 1.4)
        if len(self.nose_rel_history) >= 8 and result.motion_level < 1.2:
            xs = [x for _, x, _ in self.nose_rel_history]
            ys = [y for _, _, y in self.nose_rel_history]
            amp_x, amp_y = max(xs) - min(xs), max(ys) - min(ys)
            if amp_x > 0.35 / self.sensitivity and amp_x > 1.5 * amp_y and self._count_reversals(xs, 0.05) >= 2:
                self._emit_gesture(result, "Shaking head", min(0.93, 0.80 + amp_x * 0.15))
            elif amp_y > 0.18 / self.sensitivity and amp_y > 1.5 * amp_x and self._count_reversals(ys, 0.03) >= 2:
                self._emit_gesture(result, "Nodding", min(0.93, 0.80 + amp_y * 0.3))

        result.activities = self._debounce_activities(raw, t)

    def _debounce_activities(self, raw: Dict[str, float], t: float) -> List[Dict[str, Any]]:
        """Reports an activity only after it held for its on-time, and keeps it briefly through dropouts."""
        active: List[Dict[str, Any]] = []
        for name in set(raw) | set(self.activity_state):
            state = self.activity_state.get(name)
            if name in raw:
                if state is None:
                    state = self.activity_state[name] = [t, t, False, raw[name]]
                state[1] = t
                state[3] = raw[name]
                if not state[2] and t - state[0] >= ACTIVITY_ON_TIME.get(name, DEFAULT_ON_TIME):
                    state[2] = True
            elif state is not None and t - state[1] > ACTIVITY_OFF_TIME:
                del self.activity_state[name]
                continue
            elif state is not None and not state[2]:
                # A short-lived raw detection that never activated: start over next time
                del self.activity_state[name]
                continue
            if state is not None and state[2]:
                active.append({"name": name, "confidence": round(state[3], 2)})
        active.sort(key=lambda a: a["name"])
        return active

    @staticmethod
    def _emit_gesture(result: FrameAnalysisResult, name: str, conf: float) -> None:
        # Reported on every frame it is seen; the EventManager rate-limits terminal lines
        result.gestures.append({"name": name, "confidence": round(conf, 2)})

    @staticmethod
    def _trim(history: deque, t: float, window: float) -> None:
        while history and t - history[0][0] > window:
            history.popleft()

    @staticmethod
    def _count_reversals(values: List[float], min_step: float) -> int:
        reversals, prev_dir, anchor = 0, 0, values[0] if values else 0.0
        for v in values[1:]:
            d = v - anchor
            if abs(d) < min_step:
                continue
            cur = 1 if d > 0 else -1
            if prev_dir != 0 and cur != prev_dir:
                reversals += 1
            prev_dir, anchor = cur, v
        return reversals

    @staticmethod
    def _count_closures(gaps: List[float], close: float, open_: float) -> int:
        """Counts open -> closed transitions in a sequence of wrist gaps."""
        count, is_open = 0, False
        for g in gaps:
            if g > open_:
                is_open = True
            elif g < close and is_open:
                count += 1
                is_open = False
        return count

    def _classify_posture(self, lms: List[NormalizedLandmark], torso_height: float,
                          image_size: Tuple[int, int] = (1, 1)) -> Optional[str]:
        """
        Calculates knee angles and hip-to-ankle relations to determine Standing vs Sitting vs Crouching.
        Returns None when the legs are out of view and posture can't be judged reliably.
        """
        l_sh, r_sh = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
        l_hip, r_hip = lms[LEFT_HIP], lms[RIGHT_HIP]
        l_knee, r_knee = lms[LEFT_KNEE], lms[RIGHT_KNEE]
        l_ank, r_ank = lms[LEFT_ANKLE], lms[RIGHT_ANKLE]

        # Lying down: spine close to horizontal
        if min(l_sh.visibility, r_sh.visibility, l_hip.visibility, r_hip.visibility) > 0.5:
            w, h = image_size if image_size else (1, 1)
            ar = (w / float(h)) if h else 1.0
            dx = ((l_sh.x + r_sh.x) - (l_hip.x + r_hip.x)) / 2.0 * ar
            dy = ((l_sh.y + r_sh.y) - (l_hip.y + r_hip.y)) / 2.0
            if math.hypot(dx, dy) > 0.03 and abs(math.degrees(math.atan2(dx, -dy))) > 65.0:
                return "Lying down"

        knee_angles = []
        if l_hip.visibility > 0.4 and l_knee.visibility > 0.4 and l_ank.visibility > 0.4:
            knee_angles.append(calculate_angle(l_hip, l_knee, l_ank))
        if r_hip.visibility > 0.4 and r_knee.visibility > 0.4 and r_ank.visibility > 0.4:
            knee_angles.append(calculate_angle(r_hip, r_knee, r_ank))

        if not knee_angles:
            # Only the upper body is visible (typical desk webcam view). Knees visible
            # without ankles and close to hip height means a seated person.
            knees_vis = [(k, hp) for k, hp in ((l_knee, l_hip), (r_knee, r_hip)) if k.visibility > 0.5 and hp.visibility > 0.5]
            if knees_vis and torso_height > 0.05:
                drop = sum(k.y - hp.y for k, hp in knees_vis) / len(knees_vis)
                if drop < 0.45 * torso_height:
                    return "Sitting"
            return None

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
