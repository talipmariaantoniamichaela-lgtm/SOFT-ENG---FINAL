"""
Unit tests for Motion Analyzer and Posture/Movement Classification.
"""
import pytest
from app.detection.pose_estimator import (
    TrackingResult, NormalizedLandmark,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, NOSE
)
from app.detection.motion_analyzer import (
    MotionAnalyzer, calculate_distance, calculate_angle
)


def create_mock_body(
    posture: str = "standing",
    left_hand_raised: bool = False,
    right_hand_raised: bool = False,
    lean_direction: str = "none",
    turn_direction: str = "none"
):
    """Generates 33 mock pose landmarks matching anatomical geometry."""
    lms = [NormalizedLandmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]

    # Head / Nose
    nose_x = 0.5
    if turn_direction == "right":
        nose_x = 0.58
    elif turn_direction == "left":
        nose_x = 0.42
    lms[NOSE] = NormalizedLandmark(x=nose_x, y=0.18, z=0.0, visibility=0.9)

    # Shoulders
    sh_offset = 0.0
    if lean_direction == "right":
        sh_offset = 0.08
    elif lean_direction == "left":
        sh_offset = -0.08

    lms[LEFT_SHOULDER] = NormalizedLandmark(x=0.42 + sh_offset, y=0.28, z=0.0, visibility=0.9)
    lms[RIGHT_SHOULDER] = NormalizedLandmark(x=0.58 + sh_offset, y=0.28, z=0.0, visibility=0.9)

    # Arms
    # Left Arm
    if left_hand_raised:
        lms[LEFT_ELBOW] = NormalizedLandmark(x=0.36, y=0.22, z=0.0, visibility=0.9)
        lms[LEFT_WRIST] = NormalizedLandmark(x=0.32, y=0.14, z=0.0, visibility=0.9)
    else:
        lms[LEFT_ELBOW] = NormalizedLandmark(x=0.40, y=0.42, z=0.0, visibility=0.9)
        lms[LEFT_WRIST] = NormalizedLandmark(x=0.40, y=0.55, z=0.0, visibility=0.9)

    # Right Arm
    if right_hand_raised:
        lms[RIGHT_ELBOW] = NormalizedLandmark(x=0.64, y=0.22, z=0.0, visibility=0.9)
        lms[RIGHT_WRIST] = NormalizedLandmark(x=0.68, y=0.14, z=0.0, visibility=0.9)
    else:
        lms[RIGHT_ELBOW] = NormalizedLandmark(x=0.60, y=0.42, z=0.0, visibility=0.9)
        lms[RIGHT_WRIST] = NormalizedLandmark(x=0.60, y=0.55, z=0.0, visibility=0.9)

    # Hips
    lms[LEFT_HIP] = NormalizedLandmark(x=0.45, y=0.52, z=0.0, visibility=0.9)
    lms[RIGHT_HIP] = NormalizedLandmark(x=0.55, y=0.52, z=0.0, visibility=0.9)

    # Legs based on posture
    if posture == "standing":
        lms[LEFT_KNEE] = NormalizedLandmark(x=0.45, y=0.72, z=0.0, visibility=0.9)
        lms[RIGHT_KNEE] = NormalizedLandmark(x=0.55, y=0.72, z=0.0, visibility=0.9)
        lms[LEFT_ANKLE] = NormalizedLandmark(x=0.45, y=0.92, z=0.0, visibility=0.9)
        lms[RIGHT_ANKLE] = NormalizedLandmark(x=0.55, y=0.92, z=0.0, visibility=0.9)
    elif posture == "sitting":
        # Hips and knees close in Y, knee bent ~90 deg
        lms[LEFT_HIP] = NormalizedLandmark(x=0.45, y=0.55, z=0.0, visibility=0.9)
        lms[RIGHT_HIP] = NormalizedLandmark(x=0.55, y=0.55, z=0.0, visibility=0.9)
        lms[LEFT_KNEE] = NormalizedLandmark(x=0.38, y=0.56, z=0.0, visibility=0.9)
        lms[RIGHT_KNEE] = NormalizedLandmark(x=0.62, y=0.56, z=0.0, visibility=0.9)
        lms[LEFT_ANKLE] = NormalizedLandmark(x=0.38, y=0.82, z=0.0, visibility=0.9)
        lms[RIGHT_ANKLE] = NormalizedLandmark(x=0.62, y=0.82, z=0.0, visibility=0.9)
    elif posture == "crouching":
        # Knee angle acute < 85 deg
        lms[LEFT_KNEE] = NormalizedLandmark(x=0.42, y=0.78, z=0.0, visibility=0.9)
        lms[RIGHT_KNEE] = NormalizedLandmark(x=0.58, y=0.78, z=0.0, visibility=0.9)
        lms[LEFT_ANKLE] = NormalizedLandmark(x=0.46, y=0.82, z=0.0, visibility=0.9)
        lms[RIGHT_ANKLE] = NormalizedLandmark(x=0.54, y=0.82, z=0.0, visibility=0.9)

    return lms


def test_geometry_helpers():
    p1 = NormalizedLandmark(0.0, 0.0, 0.0)
    p2 = NormalizedLandmark(0.3, 0.4, 0.0)
    dist = calculate_distance(p1, p2)
    assert abs(dist - 0.5) < 1e-4

    a = NormalizedLandmark(0.0, 1.0, 0.0)
    b = NormalizedLandmark(0.0, 0.0, 0.0)
    c = NormalizedLandmark(1.0, 0.0, 0.0)
    angle = calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-4


def test_posture_classification():
    analyzer = MotionAnalyzer()

    # Standing
    standing_lms = create_mock_body(posture="standing")
    tr = TrackingResult(person_detected=True, pose_landmarks=standing_lms)
    res = None
    for i in range(5):
        res = analyzer.analyze(tr, 1000.0 + i * 0.033)
    assert res.posture == "Standing"

    # Sitting
    analyzer.reset()
    sitting_lms = create_mock_body(posture="sitting")
    tr_sit = TrackingResult(person_detected=True, pose_landmarks=sitting_lms)
    for i in range(5):
        res = analyzer.analyze(tr_sit, 1005.0 + i * 0.033)
    assert res.posture == "Sitting"


def test_arm_raise_detection():
    analyzer = MotionAnalyzer()

    # Right hand raised
    lms_r_raised = create_mock_body(right_hand_raised=True)
    tr = TrackingResult(person_detected=True, pose_landmarks=lms_r_raised)
    res = analyzer.analyze(tr, 1000.0)
    assert res.right_hand_raised is True
    assert res.left_hand_raised is False
    assert res.both_hands_raised is False

    # Both hands raised
    lms_both = create_mock_body(left_hand_raised=True, right_hand_raised=True)
    tr_both = TrackingResult(person_detected=True, pose_landmarks=lms_both)
    res_both = analyzer.analyze(tr_both, 1001.0)
    assert res_both.left_hand_raised is True
    assert res_both.right_hand_raised is True
    assert res_both.both_hands_raised is True


def test_hand_raise_state_clears_when_hands_are_lowered_or_lost():
    analyzer = MotionAnalyzer()
    raised = TrackingResult(
        person_detected=True,
        pose_landmarks=create_mock_body(left_hand_raised=True, right_hand_raised=True),
    )
    assert analyzer.analyze(raised, 1000.0).both_hands_raised is True

    # Returning both wrists to their normal position must clear immediately.
    lowered = TrackingResult(person_detected=True, pose_landmarks=create_mock_body())
    assert analyzer.analyze(lowered, 1000.1).both_hands_raised is False

    # A hand that is no longer confidently visible must not retain an old pose.
    analyzer.analyze(raised, 1000.2)
    partially_visible = create_mock_body()
    partially_visible[LEFT_WRIST].visibility = 0.0
    partially_visible[RIGHT_WRIST].visibility = 0.0
    res = analyzer.analyze(TrackingResult(person_detected=True, pose_landmarks=partially_visible), 1000.3)
    assert res.left_hand_raised is False
    assert res.right_hand_raised is False


def test_leaning_detection():
    analyzer = MotionAnalyzer()

    # Leaning right
    lms_lean = create_mock_body(lean_direction="right")
    tr = TrackingResult(person_detected=True, pose_landmarks=lms_lean)
    res = analyzer.analyze(tr, 1000.0)
    names = [m["name"] for m in res.body_movements]
    assert "Leaning right" in names
