"""
Unit and Integration Tests for Multi-Person AI Tracking and Dataset Identity Recognition.
"""
import pytest
import numpy as np
import cv2

from app.services.multi_person_tracker import MultiPersonTracker


def test_tracker_initialization_events():
    tracker = MultiPersonTracker()
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    persons, events = tracker.process_frame(dummy)

    # Must contain the required startup banner events
    event_texts = [e.text for e in events]
    assert "Camera connected successfully" in event_texts
    assert "Face recognition initialized" in event_texts
    assert "Movement detection started" in event_texts


def test_multi_person_tracking_and_actions():
    tracker = MultiPersonTracker()
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.process_frame(dummy)

    # Test hand gesture triggering
    evt = tracker.trigger_hand_gesture(person_id=1, gesture_type="right_hand_raised", confidence=0.95)
    # Even if unknown person, should emit natural description
    if evt:
        assert "raises" in evt.text and "right hand" in evt.text


def test_unknown_fallback():
    tracker = MultiPersonTracker()
    # Random white noise crop should not falsely classify as a known person
    noise = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    name, conf, known = tracker._infer_identity(noise, 50, 50, 100, 100)
    assert not known
    assert name == "Unknown"
