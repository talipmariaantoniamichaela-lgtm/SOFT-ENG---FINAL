"""
Pose Estimation and Landmark Tracking module for Motion Reader.
Wraps MediaPipe Tasks Vision (PoseLandmarker, HandLandmarker, FaceLandmarker)
with landmark smoothing and coordinate normalization.
"""
import sys
import os
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.detection.facial_expression import classify_expression


def get_model_path(filename: str) -> str:
    """Resolves model path for both source code execution and PyInstaller bundled binaries."""
    if hasattr(sys, "_MEIPASS"):
        candidate = os.path.join(sys._MEIPASS, "app", "assets", "models", filename)
        if os.path.exists(candidate):
            return candidate
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base_dir, "assets", "models", filename)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(os.getcwd(), "app", "assets", "models", filename)


POSE_MODEL_PATH = get_model_path("pose_landmarker.task")
HAND_MODEL_PATH = get_model_path("hand_landmarker.task")
FACE_MODEL_PATH = get_model_path("face_landmarker.task")


@dataclass
class NormalizedLandmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass
class TrackingResult:
    person_detected: bool = False
    person_id: int = 1
    pose_landmarks: List[NormalizedLandmark] = field(default_factory=list)
    left_hand_landmarks: Optional[List[NormalizedLandmark]] = None
    right_hand_landmarks: Optional[List[NormalizedLandmark]] = None
    face_landmarks: Optional[List[NormalizedLandmark]] = None
    facial_expression: str = "Neutral"
    facial_expression_confidence: float = 0.0
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
    timestamp_ms: int = 0
    image_size: Tuple[int, int] = (1, 1)  # (width, height) of the source frame, for aspect-correct geometry


def compute_bounding_box(landmarks: List[NormalizedLandmark], min_visibility: float = 0.4) -> Optional[Tuple[float, float, float, float]]:
    xs = [lm.x for lm in landmarks if lm.visibility > min_visibility]
    ys = [lm.y for lm in landmarks if lm.visibility > min_visibility]
    if not xs or not ys:
        return None
    return (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))


# Key landmark indices in MediaPipe Pose
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Standard skeletal connection pairs (start_idx, end_idx)
POSE_CONNECTIONS = [
    # Shoulders and spine
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    # Arms
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    # Torso
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # Legs
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    # Feet
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
]

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (5, 9), (9, 13), (13, 17)
]


class LandmarkSmoother:
    """
    Applies Exponential Moving Average (EMA) smoothing to landmark coordinates
    to prevent frame-to-frame jitter.
    """
    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self.prev_landmarks: Optional[List[NormalizedLandmark]] = None

    def reset(self) -> None:
        self.prev_landmarks = None

    def smooth(self, current: List[NormalizedLandmark]) -> List[NormalizedLandmark]:
        if not current:
            self.prev_landmarks = None
            return []

        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current):
            self.prev_landmarks = current
            return current

        smoothed = []
        for prev, curr in zip(self.prev_landmarks, current):
            sx = self.alpha * curr.x + (1.0 - self.alpha) * prev.x
            sy = self.alpha * curr.y + (1.0 - self.alpha) * prev.y
            sz = self.alpha * curr.z + (1.0 - self.alpha) * prev.z
            smoothed.append(NormalizedLandmark(
                x=sx, y=sy, z=sz,
                visibility=curr.visibility,
                presence=curr.presence
            ))

        self.prev_landmarks = smoothed
        return smoothed


class PoseEstimator:
    """
    MediaPipe Tasks Vision pose, hand, and face estimator.
    """
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_hands: bool = True,
        enable_face: bool = True,
        smoothing: bool = True,
        smoothing_factor: float = 0.55,
        num_poses: int = 5
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_hands = enable_hands
        self.enable_face = enable_face
        self.smoothing = smoothing
        self.num_poses = max(1, num_poses)

        self.smoother = LandmarkSmoother(alpha=smoothing_factor)
        self.pose_detector = None
        self.hand_detector = None
        self.face_detector = None

        self._init_models()

    def _init_models(self) -> None:
        """Initializes the MediaPipe vision task landmarker models."""
        try:
            if os.path.exists(POSE_MODEL_PATH):
                base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    num_poses=self.num_poses,
                    min_pose_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    output_segmentation_masks=False,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.pose_detector = vision.PoseLandmarker.create_from_options(options)
            else:
                print(f"Warning: Pose model not found at {POSE_MODEL_PATH}")
        except Exception as e:
            print(f"Error initializing PoseLandmarker: {e}")

        try:
            if self.enable_hands and os.path.exists(HAND_MODEL_PATH):
                base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2 * self.num_poses,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.hand_detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing HandLandmarker: {e}")

        try:
            if self.enable_face and os.path.exists(FACE_MODEL_PATH):
                base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=self.num_poses,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    output_face_blendshapes=True,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.face_detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing FaceLandmarker: {e}")

    def reset(self) -> None:
        """Resets temporal smoothing state."""
        self.smoother.reset()

    def estimate(self, rgb_frame: np.ndarray, timestamp_ms: int = 0) -> TrackingResult:
        """
        Single-person convenience wrapper: returns the most prominent person,
        smoothed with this estimator's own smoother.
        """
        people = self.estimate_all(rgb_frame, timestamp_ms)
        if not people:
            self.smoother.reset()
            return TrackingResult(timestamp_ms=timestamp_ms)
        result = people[0]
        if self.smoothing:
            result.pose_landmarks = self.smoother.smooth(result.pose_landmarks)
            result.bounding_box = compute_bounding_box(result.pose_landmarks)
        return result

    def estimate_all(self, rgb_frame: np.ndarray, timestamp_ms: int = 0) -> List[TrackingResult]:
        """
        Runs pose estimation for every person in the RGB frame (up to num_poses).
        Hands and face meshes are assigned to the person whose wrist / nose is closest.
        Results are unsmoothed and sorted largest-first; smoothing is done per tracked person.
        """
        if self.pose_detector is None or rgb_frame is None:
            return []

        h, w, _ = rgb_frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 1. Pose Landmark Detection
        people: List[TrackingResult] = []
        try:
            pose_result = self.pose_detector.detect(mp_image)
            for pose in (pose_result.pose_landmarks or []):
                lms = [
                    NormalizedLandmark(
                        x=lm.x, y=lm.y, z=lm.z,
                        visibility=getattr(lm, "visibility", 1.0) or 1.0,
                        presence=getattr(lm, "presence", 1.0) or 1.0
                    )
                    for lm in pose
                ]
                bbox = compute_bounding_box(lms)
                if bbox is None:
                    continue
                people.append(TrackingResult(
                    person_detected=True,
                    pose_landmarks=lms,
                    bounding_box=bbox,
                    timestamp_ms=timestamp_ms,
                    image_size=(w, h),
                ))
        except Exception as e:
            print(f"Pose detection error: {e}")
            return []

        if not people:
            return []
        people.sort(key=lambda p: (p.bounding_box[2] - p.bounding_box[0]) * (p.bounding_box[3] - p.bounding_box[1]), reverse=True)

        # 2. Hand Landmark Detection (if enabled), assigned to the nearest pose wrist
        if self.enable_hands and self.hand_detector is not None:
            try:
                hand_result = self.hand_detector.detect(mp_image)
                for idx, hand in enumerate(hand_result.hand_landmarks or []):
                    h_landmarks = [NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand]
                    wrist = h_landmarks[0]
                    best, best_d, best_side = None, 0.12, None
                    for person in people:
                        for side, wi in (("left", LEFT_WRIST), ("right", RIGHT_WRIST)):
                            pw = person.pose_landmarks[wi]
                            d = math.hypot(pw.x - wrist.x, pw.y - wrist.y)
                            if d < best_d:
                                best, best_d, best_side = person, d, side
                    if best is not None:
                        if best_side == "right":
                            best.right_hand_landmarks = h_landmarks
                        else:
                            best.left_hand_landmarks = h_landmarks
            except Exception:
                pass

        # 3. Face Landmark Detection (if enabled), assigned to the nearest pose nose
        if self.enable_face and self.face_detector is not None:
            try:
                face_result = self.face_detector.detect(mp_image)
                face_meshes = face_result.face_landmarks or []
                blendshape_sets = face_result.face_blendshapes or []
                for index, face in enumerate(face_meshes):
                    f_landmarks = [NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face]
                    tip = f_landmarks[1] if len(f_landmarks) > 1 else f_landmarks[0]
                    best, best_d = None, 0.10
                    for person in people:
                        pn = person.pose_landmarks[NOSE]
                        d = math.hypot(pn.x - tip.x, pn.y - tip.y)
                        if d < best_d and person.face_landmarks is None:
                            best, best_d = person, d
                    if best is not None:
                        best.face_landmarks = f_landmarks
                        categories = blendshape_sets[index] if index < len(blendshape_sets) else []
                        scores = {
                            category.category_name: float(category.score)
                            for category in categories
                        }
                        best.facial_expression, best.facial_expression_confidence = classify_expression(scores)
            except Exception:
                pass

        return people
