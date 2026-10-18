"""
Unit tests for Event Manager, duplicate suppression, and log exports.
"""
import os
import json
import csv
import pytest
from app.services.event_manager import EventManager, MovementEvent, EventType


def test_repeated_event_suppression():
    manager = EventManager()

    # Simulate 60 frames of a person continuously standing
    t_start = 1000.0
    for i in range(60):
        manager.process_frame_analysis(
            person_detected=True,
            person_id=1,
            posture="Standing",
            left_hand_raised=False,
            right_hand_raised=False,
            gestures=[],
            body_movements=[],
            current_time=t_start + i * 0.033,
            is_video_relative=False
        )

    # There should only be:
    # 1. Person detected (ID: 1)
    # 2. Standing
    # Total = 2 events, NOT 60 duplicate Standing events!
    assert len(manager.events) == 2
    assert manager.events[0].description == "Person detected"
    assert manager.events[0].person_id == 1
    assert manager.events[1].description == "Standing"


def test_state_transitions():
    manager = EventManager()
    t = 1000.0

    # 1. Person enters
    manager.process_frame_analysis(True, 1, "Standing", False, False, [], [], t)
    t += 1.0

    # 2. Right hand raised
    manager.process_frame_analysis(True, 1, "Standing", False, True, [], [], t)
    t += 1.0

    # 3. Waving gesture
    manager.process_frame_analysis(True, 1, "Standing", False, True, [{"name": "Waving", "confidence": 0.92}], [], t)
    t += 1.0

    # 4. Right hand lowered
    manager.process_frame_analysis(True, 1, "Standing", False, False, [], [], t)
    t += 1.0

    # 5. Sitting
    manager.process_frame_analysis(True, 1, "Sitting", False, False, [], [], t)

    descriptions = [e.description for e in manager.events]
    assert "Person detected" in descriptions
    assert "Standing" in descriptions
    assert "Right hand raised" in descriptions
    assert "Waving" in descriptions
    assert "Right hand lowered" in descriptions
    assert "Sitting" in descriptions


def test_export_csv_and_json(tmp_path):
    manager = EventManager()
    manager.add_event(MovementEvent(
        timestamp=1000.0,
        formatted_time="09:12:01",
        event_type=EventType.PERSON_DETECTION,
        description="Person detected",
        person_id=1
    ))
    manager.add_event(MovementEvent(
        timestamp=1005.0,
        formatted_time="09:12:05",
        event_type=EventType.GESTURE,
        description="Waving",
        confidence=0.92
    ))

    # Test CSV
    csv_path = str(tmp_path / "test_events.csv")
    assert manager.export_csv(csv_path) is True
    assert os.path.exists(csv_path)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3  # Header + 2 entries

    # Test JSON
    json_path = str(tmp_path / "test_events.json")
    assert manager.export_json(json_path) is True
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[1]["description"] == "Waving"
        assert data[1]["confidence"] == 0.92
