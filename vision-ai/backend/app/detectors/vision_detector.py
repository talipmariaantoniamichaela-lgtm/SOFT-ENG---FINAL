"""
MediaPipe Face, Pose, and Hand detector for VISION AI.
Wraps existing MediaPipe Tasks Vision landmarkers with multi-person awareness.
"""
import os
import sys
import time
import math
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def _resolve_model(filename: str) -> Optional[str]:
    """Find MediaPipe model file in known locations."""
    candidates = [
        Path(__file__).parent.parent.parent.parent.parent / "app" / "assets" / "models" / filename,
        Path(__file__).parent.parent / "assets" / "models" / filename,
        Path.cwd() / "app" / "assets" / "models" / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def cx(self): return (self.x_min + self.x_max) / 2
    @property
    def cy(self): return (self.y_min + self.y_max) / 2
    @property
    def width(self): return self.x_max - self.x_min
    @property
    def height(self): return self.y_max - self.y_min


@dataclass
class FaceResult:
    landmarks: List[Landmark] = field(default_factory=list)
    # Head orientation estimated from face landmarks
    yaw: float = 0.0    # left(-)/right(+) degrees
    pitch: float = 0.0  # up(-)/down(+) degrees
    roll: float = 0.0   # tilt degrees
    bounding_box: Optional[BoundingBox] = None


@dataclass
class HandResult:
    handedness: str = "Unknown"   # "Left" | "Right"
    landmarks: List[Landmark] = field(default_factory=list)
    gesture: Optional[str] = None
    confidence: float = 0.0


@dataclass
class PersonDetection:
    person_id: int = 1
    pose_landmarks: List[Landmark] = field(default_factory=list)
    face: Optional[FaceResult] = None
    left_hand: Optional[HandResult] = None
    right_hand: Optional[HandResult] = None
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 1.0


@dataclass
class DetectionFrame:
    persons: List[PersonDetection] = field(default_factory=list)
    timestamp_ms: int = 0
    inference_ms: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


# ─── Pose Landmark Indices ────────────────────────────────────────────────────

NOSE = 0
LEFT_EYE = 2; RIGHT_EYE = 5
LEFT_EAR = 7; RIGHT_EAR = 8
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_WRIST = 15; RIGHT_WRIST = 16
LEFT_HIP = 23; RIGHT_HIP = 24
LEFT_KNEE = 25; RIGHT_KNEE = 26
LEFT_ANKLE = 27; RIGHT_ANKLE = 28

POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# Face landmarks for head orientation (simplified subset)
FACE_NOSE_TIP = 4
FACE_LEFT_EYE_OUTER = 33
FACE_RIGHT_EYE_OUTER = 263
FACE_CHIN = 152
FACE_FOREHEAD = 10
FACE_LEFT_CHEEK = 234
FACE_RIGHT_CHEEK = 454


# ─── EMA Smoother ─────────────────────────────────────────────────────────────

class EMALandmarkSmoother:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: Optional[List[Landmark]] = None

    def reset(self):
        self._prev = None

    def smooth(self, lms: List[Landmark]) -> List[Landmark]:
        if not lms:
            self._prev = None
            return lms
        if self._prev is None or len(self._prev) != len(lms):
            self._prev = lms
            return lms
        smoothed = []
        for p, c in zip(self._prev, lms):
            smoothed.append(Landmark(
                x=self.alpha * c.x + (1 - self.alpha) * p.x,
                y=self.alpha * c.y + (1 - self.alpha) * p.y,
                z=self.alpha * c.z + (1 - self.alpha) * p.z,
                visibility=c.visibility
            ))
        self._prev = smoothed
        return smoothed


# ─── Main Detector ─────────────────────────────────────────────────────────────

class VisionDetector:
    """
    Unified detector that runs MediaPipe Face, Pose, and Hand landmarkers
    on each frame. Designed to be instantiated once and reused.
    """

    def __init__(
        self,
        enable_face: bool = True,
        enable_pose: bool = True,
        enable_hands: bool = True,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        smoothing: bool = True,
    ):
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self._pose_detector = None
        self._hand_detector = None
        self._face_detector = None
        self._pose_smoother = EMALandmarkSmoother(0.5) if smoothing else None
        self._loaded: Dict[str, bool] = {"pose": False, "face": False, "hands": False}

        self._init_models()

    def _init_models(self):
        """Initialise all requested MediaPipe landmarkers."""
        pose_path = _resolve_model("pose_landmarker.task")
        if self.enable_pose and pose_path:
            try:
                opts = mp_vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=pose_path),
                    min_pose_detection_confidence=self.detection_confidence,
                    min_tracking_confidence=self.tracking_confidence,
                    output_segmentation_masks=False,
                    running_mode=mp_vision.RunningMode.IMAGE,
                )
                self._pose_detector = mp_vision.PoseLandmarker.create_from_options(opts)
                self._loaded["pose"] = True
            except Exception as e:
                print(f"[VisionDetector] Pose init error: {e}")

        hand_path = _resolve_model("hand_landmarker.task")
        if self.enable_hands and hand_path:
            try:
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=hand_path),
                    num_hands=2,
                    min_hand_detection_confidence=self.detection_confidence,
                    min_tracking_confidence=self.tracking_confidence,
                    running_mode=mp_vision.RunningMode.IMAGE,
                )
                self._hand_detector = mp_vision.HandLandmarker.create_from_options(opts)
                self._loaded["hands"] = True
            except Exception as e:
                print(f"[VisionDetector] Hand init error: {e}")

        face_path = _resolve_model("face_landmarker.task")
        if self.enable_face and face_path:
            try:
                opts = mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=face_path),
                    min_face_detection_confidence=self.detection_confidence,
                    min_tracking_confidence=self.tracking_confidence,
                    running_mode=mp_vision.RunningMode.IMAGE,
                )
                self._face_detector = mp_vision.FaceLandmarker.create_from_options(opts)
                self._loaded["face"] = True
            except Exception as e:
                print(f"[VisionDetector] Face init error: {e}")

    @property
    def models_loaded(self) -> Dict[str, bool]:
        return self._loaded.copy()

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int = 0) -> DetectionFrame:
        """
        Run all enabled detectors on an RGB frame.
        Returns a structured DetectionFrame.
        """
        t0 = time.perf_counter()
        h, w = rgb_frame.shape[:2]
        result = DetectionFrame(timestamp_ms=timestamp_ms, frame_width=w, frame_height=h)

        if self._pose_detector is None and self._face_detector is None:
            result.inference_ms = (time.perf_counter() - t0) * 1000
            return result

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── Pose ──
        person = PersonDetection(person_id=1)
        person_detected = False

        if self._pose_detector is not None:
            try:
                pr = self._pose_detector.detect(mp_img)
                if pr and pr.pose_landmarks:
                    raw = [Landmark(x=lm.x, y=lm.y, z=lm.z,
                                    visibility=getattr(lm, "visibility", 1.0) or 1.0)
                           for lm in pr.pose_landmarks[0]]
                    lms = self._pose_smoother.smooth(raw) if self._pose_smoother else raw
                    person.pose_landmarks = lms
                    visible = [lm for lm in lms if lm.visibility > 0.4]
                    if visible:
                        xs = [lm.x for lm in visible]
                        ys = [lm.y for lm in visible]
                        person.bounding_box = BoundingBox(min(xs), min(ys), max(xs), max(ys))
                    person_detected = True
                else:
                    if self._pose_smoother:
                        self._pose_smoother.reset()
            except Exception as e:
                print(f"[VisionDetector] Pose detect error: {e}")

        # ── Hands ──
        if self._hand_detector is not None:
            try:
                hr = self._hand_detector.detect(mp_img)
                if hr and hr.hand_landmarks:
                    for i, handedness in enumerate(hr.handedness):
                        label = handedness[0].category_name
                        h_lms = [Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in hr.hand_landmarks[i]]
                        gesture = self._classify_hand_gesture(h_lms)
                        hand = HandResult(
                            handedness=label,
                            landmarks=h_lms,
                            gesture=gesture,
                            confidence=handedness[0].score
                        )
                        if label == "Right":
                            person.right_hand = hand
                        else:
                            person.left_hand = hand
                        person_detected = True
            except Exception as e:
                pass

        # ── Face ──
        if self._face_detector is not None:
            try:
                fr = self._face_detector.detect(mp_img)
                if fr and fr.face_landmarks:
                    f_lms = [Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in fr.face_landmarks[0]]
                    face = FaceResult(landmarks=f_lms)
                    face.yaw, face.pitch, face.roll = self._estimate_head_orientation(f_lms)
                    # Face bounding box
                    if f_lms:
                        fxs = [lm.x for lm in f_lms]
                        fys = [lm.y for lm in f_lms]
                        face.bounding_box = BoundingBox(min(fxs), min(fys), max(fxs), max(fys))
                    person.face = face
                    person_detected = True
            except Exception as e:
                pass

        if person_detected:
            result.persons.append(person)

        result.inference_ms = (time.perf_counter() - t0) * 1000
        return result

    def _estimate_head_orientation(self, lms: List[Landmark]) -> Tuple[float, float, float]:
        """
        Estimate yaw/pitch/roll from face landmarks.
        Returns (yaw, pitch, roll) in degrees.
        """
        if len(lms) < 468:
            return 0.0, 0.0, 0.0
        try:
            nose = lms[FACE_NOSE_TIP]
            l_eye = lms[FACE_LEFT_EYE_OUTER]
            r_eye = lms[FACE_RIGHT_EYE_OUTER]
            chin = lms[FACE_CHIN]
            forehead = lms[FACE_FOREHEAD]

            # Yaw: nose x relative to mid-eye x
            mid_eye_x = (l_eye.x + r_eye.x) / 2
            eye_width = abs(r_eye.x - l_eye.x)
            yaw = ((nose.x - mid_eye_x) / max(eye_width, 0.01)) * 45.0

            # Pitch: nose y relative to mid-point of chin-forehead
            mid_y = (chin.y + forehead.y) / 2
            face_height = abs(chin.y - forehead.y)
            pitch = ((nose.y - mid_y) / max(face_height, 0.01)) * 30.0

            # Roll: eye line angle
            dy = r_eye.y - l_eye.y
            dx = r_eye.x - l_eye.x
            roll = math.degrees(math.atan2(dy, dx))

            return round(yaw, 1), round(pitch, 1), round(roll, 1)
        except Exception:
            return 0.0, 0.0, 0.0

    def _classify_hand_gesture(self, lms: List[Landmark]) -> Optional[str]:
        """
        Classify simple hand gestures from 21 hand landmarks.
        """
        if len(lms) < 21:
            return None
        try:
            wrist = lms[0]
            # Fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
            # MCP (base knuckle): thumb=2, index=5, middle=9, ring=13, pinky=17
            fingertips = [lms[4], lms[8], lms[12], lms[16], lms[20]]
            mcp_bases = [lms[2], lms[5], lms[9], lms[13], lms[17]]

            # Finger extended: tip y < mcp y (in normalized image coords, y increases downward)
            extended = []
            for tip, mcp in zip(fingertips, mcp_bases):
                extended.append(tip.y < mcp.y - 0.04)

            n_extended = sum(extended)

            # Thumbs up: thumb extended, others closed
            if extended[0] and not any(extended[1:]):
                return "Thumbs Up"

            # Pointing: index extended only
            if extended[1] and not extended[0] and not any(extended[2:]):
                return "Pointing"

            # Open hand: 4+ fingers extended
            if n_extended >= 4:
                return "Open Hand"

            # Fist: nothing extended
            if n_extended == 0:
                return "Closed Fist"

            # Peace / Victory: index + middle
            if extended[1] and extended[2] and not extended[0] and not extended[3] and not extended[4]:
                return "Peace"

            return None
        except Exception:
            return None

    def reset(self):
        """Reset temporal smoothing state."""
        if self._pose_smoother:
            self._pose_smoother.reset()

    def close(self):
        """Release detector resources."""
        for d in [self._pose_detector, self._hand_detector, self._face_detector]:
            try:
                if d:
                    d.close()
            except Exception:
                pass
