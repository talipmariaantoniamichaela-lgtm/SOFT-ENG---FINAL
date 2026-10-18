"""
Unit tests for Temporal Gesture Detection (Waving).
"""
import pytest
from app.detection.pose_estimator import TrackingResult, NormalizedLandmark
from app.detection.motion_analyzer import MotionAnalyzer
from tests.test_motion_analyzer import create_mock_body


def test_waving_gesture_recognition():
    analyzer = MotionAnalyzer()

    # Simulate 25 frames of hand waving (right hand oscillating horizontally)
    # Oscillate wrist x between 0.62 and 0.74 at 30 fps
    wave_detected = False
    wave_conf = 0.0

    t_base = 1000.0
    for frame_idx in range(25):
        # Sine oscillation
        osc = 0.06 * (1 if (frame_idx // 4) % 2 == 0 else -1)
        lms = create_mock_body(right_hand_raised=True)
        # Modify right wrist x coordinate
        lms[16] = NormalizedLandmark(x=0.68 + osc, y=0.14, z=0.0, visibility=0.9)

        tr = TrackingResult(person_detected=True, pose_landmarks=lms)
        t = t_base + frame_idx * 0.033
        res = analyzer.analyze(tr, t)

        for g in res.gestures:
            if g["name"] == "Waving":
                wave_detected = True
                wave_conf = g["confidence"]

    assert wave_detected is True
    assert 0.80 <= wave_conf <= 0.98


def test_still_hand_does_not_wave():
    analyzer = MotionAnalyzer()

    # Hand raised but held stationary
    wave_detected = False
    t_base = 1000.0
    for frame_idx in range(25):
        lms = create_mock_body(right_hand_raised=True)
        # Stationary wrist
        lms[16] = NormalizedLandmark(x=0.68, y=0.14, z=0.0, visibility=0.9)

        tr = TrackingResult(person_detected=True, pose_landmarks=lms)
        t = t_base + frame_idx * 0.033
        res = analyzer.analyze(tr, t)

        for g in res.gestures:
            if g["name"] == "Waving":
                wave_detected = True

    assert wave_detected is False
