"""
Unit tests for VISION AI movement classifier and utilities.
Run with: pytest tests/ -v
"""
import math
import time
import pytest
from app.detectors.vision_detector import Landmark, PersonDetection, HandResult, FaceResult
from app.movement.classifier import MovementClassifier, DetectedAction


def make_landmark(x=0.5, y=0.5, z=0.0, vis=1.0):
    return Landmark(x=x, y=y, z=z, visibility=vis)


def make_standing_pose() -> list:
    """Create a minimal 33-landmark pose approximating a standing person."""
    lms = [make_landmark(0.5, 0.1)] * 33  # default all to same position
    # Nose
    lms[0] = make_landmark(0.5, 0.05)
    # Shoulders
    lms[11] = make_landmark(0.4, 0.25)  # left shoulder
    lms[12] = make_landmark(0.6, 0.25)  # right shoulder
    # Elbows
    lms[13] = make_landmark(0.38, 0.40)  # left elbow
    lms[14] = make_landmark(0.62, 0.40)  # right elbow
    # Wrists (arms down)
    lms[15] = make_landmark(0.37, 0.55)  # left wrist
    lms[16] = make_landmark(0.63, 0.55)  # right wrist
    # Hips
    lms[23] = make_landmark(0.45, 0.55)  # left hip
    lms[24] = make_landmark(0.55, 0.55)  # right hip
    # Knees
    lms[25] = make_landmark(0.45, 0.75)  # left knee
    lms[26] = make_landmark(0.55, 0.75)  # right knee
    # Ankles
    lms[27] = make_landmark(0.45, 0.95)  # left ankle
    lms[28] = make_landmark(0.55, 0.95)  # right ankle
    return lms


def make_person(lms=None, **kwargs) -> PersonDetection:
    p = PersonDetection(person_id=1)
    p.pose_landmarks = lms or make_standing_pose()
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestMovementClassifier:

    def setup_method(self):
        self.clf = MovementClassifier(sensitivity=1.0)

    def test_no_events_when_stationary(self):
        """Stationary standing person should produce at most posture events, not arm events."""
        person = make_person()
        # Run enough frames to build history
        actions = []
        for i in range(10):
            a = self.clf.classify(person, float(i) * 0.033)
            actions.extend(a)
        # No arm raise events
        arm_events = [a for a in actions if "arm" in a.action]
        assert len(arm_events) == 0, f"Unexpected arm events: {arm_events}"

    def test_right_arm_raise_detected(self):
        """Raising right wrist above shoulder should trigger right_arm_raised event."""
        clf = MovementClassifier(sensitivity=1.0)
        lms = make_standing_pose()
        person = make_person(lms)

        # Initial frames — arms down
        for i in range(5):
            clf.classify(person, float(i) * 0.033)

        # Raise right wrist above right shoulder
        lms_raised = list(lms)
        lms_raised = make_standing_pose()
        lms_raised[16] = make_landmark(0.63, 0.15)  # right wrist high
        person_raised = make_person(lms_raised)

        events = []
        for i in range(8):
            a = clf.classify(person_raised, 5.0 + float(i) * 0.033)
            events.extend(a)

        arm_events = [e for e in events if e.action == "right_arm_raised"]
        assert len(arm_events) >= 1, "Expected right_arm_raised event"
        assert arm_events[0].confidence > 0.7

    def test_posture_standing(self):
        """Standing pose should be classified as Standing."""
        clf = MovementClassifier()
        person = make_person()
        events = []
        for i in range(10):
            e = clf.classify(person, float(i) * 0.1)
            events.extend(e)
        posture_events = [e for e in events if e.category == "POSTURE"]
        if posture_events:
            assert "standing" in posture_events[0].action.lower()

    def test_cooldown_prevents_duplicate_events(self):
        """Same action should not fire twice within cooldown window."""
        clf = MovementClassifier()
        lms = make_standing_pose()
        lms[16] = make_landmark(0.63, 0.15)   # right wrist high
        person = make_person(lms)

        events = []
        for i in range(30):
            a = clf.classify(person, float(i) * 0.033)
            events.extend(a)

        arm_events = [e for e in events if e.action == "right_arm_raised"]
        assert len(arm_events) <= 2, f"Cooldown not working: {len(arm_events)} duplicate events"

    def test_confidence_range(self):
        """All emitted confidence values must be in [0, 1]."""
        clf = MovementClassifier()
        lms = make_standing_pose()
        lms[15] = make_landmark(0.37, 0.10)   # left wrist raised
        lms[16] = make_landmark(0.63, 0.10)   # right wrist raised
        person = make_person(lms)

        events = []
        for i in range(20):
            a = clf.classify(person, float(i) * 0.1)
            events.extend(a)

        for e in events:
            if e.confidence is not None:
                assert 0.0 <= e.confidence <= 1.0, f"Confidence out of range: {e.confidence}"

    def test_classifier_reset(self):
        """After reset, cooldowns should clear and events can fire again."""
        clf = MovementClassifier()
        lms = make_standing_pose()
        lms[16] = make_landmark(0.63, 0.10)
        person = make_person(lms)

        # First pass
        events1 = []
        for i in range(10):
            e = clf.classify(person, float(i) * 0.1)
            events1.extend(e)

        clf.reset()

        # Second pass — should get events again
        events2 = []
        for i in range(10):
            e = clf.classify(person, float(i) * 0.1)
            events2.extend(e)

        arm2 = [e for e in events2 if e.action == "right_arm_raised"]
        assert len(arm2) >= 1


class TestLandmarkGeometry:

    def test_angle_calculation(self):
        """Straight line should give ~180 degrees."""
        clf = MovementClassifier()
        a = make_landmark(0.0, 0.0)
        b = make_landmark(0.5, 0.0)
        c = make_landmark(1.0, 0.0)
        angle = clf._angle(a, b, c)
        assert abs(angle - 180.0) < 1.0

    def test_right_angle(self):
        """L-shape should give ~90 degrees."""
        clf = MovementClassifier()
        a = make_landmark(0.0, 0.0)
        b = make_landmark(0.0, 0.5)
        c = make_landmark(0.5, 0.5)
        angle = clf._angle(a, b, c)
        assert abs(angle - 90.0) < 5.0


class TestWaveDetection:

    def test_oscillation_triggers_wave(self):
        """Lateral wrist oscillation above raised threshold should trigger wave."""
        from collections import deque
        clf = MovementClassifier()
        hist = deque(maxlen=30)
        # Simulate 3 oscillation cycles: 0.1, -0.1, 0.1, -0.1, ...
        for i in range(20):
            val = 0.12 if i % 4 < 2 else -0.12
            hist.append(val)
        detected, conf = clf._detect_wave(hist, hand_raised=True)
        assert detected, "Wave should be detected from oscillation"
        assert 0.5 < conf <= 1.0

    def test_no_wave_when_hand_not_raised(self):
        """Wave should not trigger when hand is not raised."""
        from collections import deque
        clf = MovementClassifier()
        hist = deque(maxlen=30)
        for i in range(20):
            hist.append(0.12 if i % 4 < 2 else -0.12)
        detected, _ = clf._detect_wave(hist, hand_raised=False)
        assert not detected
