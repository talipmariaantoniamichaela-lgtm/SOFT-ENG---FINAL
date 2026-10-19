"""
Tests for sustained activities, multi-person tracking, and identity voting.
"""
from app.detection.pose_estimator import (
    TrackingResult, NormalizedLandmark, compute_bounding_box,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
)
from app.detection.motion_analyzer import MotionAnalyzer, ACTIVITY_PHRASES
from app.detection.person_tracker import PersonTracker
from app.detection.face_identity import FaceMatch
from app.services.event_manager import EventManager
from tests.test_motion_analyzer import create_mock_body


def _run(analyzer, lms, seconds=1.5, fps=30, t0=1000.0, shift_x=0.0):
    res = None
    for i in range(int(seconds * fps)):
        frame = [NormalizedLandmark(l.x + shift_x * i / fps, l.y, l.z, l.visibility) for l in lms]
        res = analyzer.analyze(TrackingResult(person_detected=True, pose_landmarks=frame), t0 + i / fps)
    return res


def _names(res):
    return [a["name"] for a in res.activities]


def test_idle_detected_when_still():
    res = _run(MotionAnalyzer(), create_mock_body(), seconds=3.0)
    assert "Idle" in _names(res)


def test_walking_detected_when_travelling():
    res = _run(MotionAnalyzer(), create_mock_body(), seconds=1.5, shift_x=0.15)
    assert "Walking" in _names(res)
    assert "Idle" not in _names(res)


def test_arms_crossed():
    lms = create_mock_body()
    lms[LEFT_ELBOW] = NormalizedLandmark(0.42, 0.40, 0, 0.9)
    lms[RIGHT_ELBOW] = NormalizedLandmark(0.58, 0.40, 0, 0.9)
    lms[LEFT_WRIST] = NormalizedLandmark(0.55, 0.38, 0, 0.9)
    lms[RIGHT_WRIST] = NormalizedLandmark(0.45, 0.38, 0, 0.9)
    res = _run(MotionAnalyzer(), lms, seconds=1.0)
    assert "Arms crossed" in _names(res)


def test_arms_stretched_out():
    lms = create_mock_body()
    lms[LEFT_ELBOW] = NormalizedLandmark(0.32, 0.28, 0, 0.9)
    lms[LEFT_WRIST] = NormalizedLandmark(0.20, 0.28, 0, 0.9)
    lms[RIGHT_ELBOW] = NormalizedLandmark(0.68, 0.28, 0, 0.9)
    lms[RIGHT_WRIST] = NormalizedLandmark(0.80, 0.28, 0, 0.9)
    res = _run(MotionAnalyzer(), lms, seconds=1.0)
    assert "Arms stretched out" in _names(res)


def test_hands_on_head_is_not_hands_raised():
    lms = create_mock_body()
    lms[LEFT_ELBOW] = NormalizedLandmark(0.34, 0.18, 0, 0.9)
    lms[RIGHT_ELBOW] = NormalizedLandmark(0.66, 0.18, 0, 0.9)
    lms[LEFT_WRIST] = NormalizedLandmark(0.45, 0.10, 0, 0.9)
    lms[RIGHT_WRIST] = NormalizedLandmark(0.55, 0.10, 0, 0.9)
    res = _run(MotionAnalyzer(), lms, seconds=1.0)
    assert "Hands on head" in _names(res)
    assert res.both_hands_raised is False


def test_activity_requires_hold_time():
    lms = create_mock_body()
    lms[LEFT_ELBOW] = NormalizedLandmark(0.32, 0.28, 0, 0.9)
    lms[LEFT_WRIST] = NormalizedLandmark(0.20, 0.28, 0, 0.9)
    lms[RIGHT_ELBOW] = NormalizedLandmark(0.68, 0.28, 0, 0.9)
    lms[RIGHT_WRIST] = NormalizedLandmark(0.80, 0.28, 0, 0.9)
    res = _run(MotionAnalyzer(), lms, seconds=0.1)
    assert "Arms stretched out" not in _names(res)


def _person_at(dx):
    lms = create_mock_body()
    lms = [NormalizedLandmark(l.x * 0.4 + dx, l.y, l.z, l.visibility) for l in lms]
    return TrackingResult(person_detected=True, pose_landmarks=lms, bounding_box=compute_bounding_box(lms), image_size=(1280, 720))


def test_tracker_keeps_ids_and_unique_names():
    tracker = PersonTracker()
    ids = None
    for i in range(10):
        visible, _ = tracker.update([_person_at(0.05), _person_at(0.55)], 1000 + i / 30)
        cur = sorted(t.track_id for t in visible)
        assert ids is None or cur == ids
        ids = cur

        # Both faces look most like "Maria"; the closer match must win the name
        faces = []
        for t in visible:
            nose = t.tracking.pose_landmarks[NOSE]
            x, y = int(nose.x * 1280) - 30, int(nose.y * 720) - 30
            first = t.track_id == ids[0]
            faces.append(FaceMatch(box=(x, y, 60, 60), name="Maria",
                                   score=0.7 if first else 0.5, best_guess="Maria", det_score=0.9,
                                   probability=0.9 if first else 0.6))
        tracker.assign_faces(faces, 1280, 720, visible)

    named = {t.track_id: t for t in tracker.tracks.values()}
    assert named[ids[0]].name == "Maria"
    assert abs(named[ids[0]].match_percent - 90.0) < 0.5
    assert named[ids[1]].name is None
    # The name is taken, so the second person must not be offered it as a guess either
    assert named[ids[1]].closest_name is None


def test_match_percent_drops_when_recognitions_disagree():
    tracker = PersonTracker()
    for i in range(10):
        visible, _ = tracker.update([_person_at(0.3)], 1000 + i / 30)
        t = visible[0]
        nose = t.tracking.pose_landmarks[NOSE]
        x, y = int(nose.x * 1280) - 30, int(nose.y * 720) - 30
        guess = "Mark" if i % 2 else "Lawrenz"   # alternating, never confident
        tracker.assign_faces([FaceMatch(box=(x, y, 60, 60), name=None, score=0.4, best_guess=guess,
                                        det_score=0.9, probability=0.3)], 1280, 720, visible)
    t = next(iter(tracker.tracks.values()))
    assert t.name is None
    assert t.closest_name in ("Mark", "Lawrenz")
    assert t.match_percent <= 15.0 + 1e-6


def test_identity_events_show_match_percent():
    m = EventManager()
    m.hit_threshold = 1
    m.process_frame_analysis(True, 1, None, False, False, [], [], 1000.0,
                             subject_name="Quinneth", subject_confidence=0.87)
    line = m.events[0].formatted_line()
    assert "Quinneth detected" in line and "87% match" in line


def test_tracker_reports_departures():
    tracker = PersonTracker(max_misses=3)
    tracker.update([_person_at(0.3)], 1000.0)
    departed = []
    for i in range(5):
        _, gone = tracker.update([], 1000.1 + i * 0.03)
        departed.extend(gone)
    assert len(departed) == 1


def test_event_manager_keeps_people_separate():
    m = EventManager()
    t = 1000.0
    for i in range(30):
        m.process_frame_analysis(True, 1, "Standing", False, i > 25, [], [], t + i * 0.033,
                                 subject_name="Maria", subject_confidence=0.6)
        m.process_frame_analysis(True, 2, "Sitting", False, False, [], [], t + i * 0.033,
                                 subject_name="Mark", subject_confidence=0.55,
                                 activities=[{"name": "Arms crossed", "confidence": 0.85}],
                                 activity_phrases=ACTIVITY_PHRASES)
    m.person_left(2, t + 2.0)
    d = [e.description for e in m.events]
    assert "Maria detected" in d and "Mark detected" in d
    assert "Maria is standing" in d and "Mark is sitting" in d
    assert "Maria raises right hand" in d
    assert "Mark has arms crossed" in d
    assert "Mark left the frame" in d
    assert d.count("Maria is standing") == 1
