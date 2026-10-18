"""
Event Manager and Event Dispatcher for Motion Reader.
Handles movement events, color coding, suppression of repeated events,
and export to CSV/JSON. State is kept per tracked person so several people
can be logged at the same time.
"""
import csv
import json
import time
from datetime import datetime
from enum import Enum
from typing import List, Optional, Callable, Dict, Any, Iterable
from dataclasses import dataclass, asdict
from PySide6.QtCore import QObject, Signal


class EventType(Enum):
    PERSON_DETECTION = "PERSON_DETECTION"
    NEUTRAL_POSTURE = "NEUTRAL_POSTURE"
    HAND_MOVEMENT = "HAND_MOVEMENT"
    GESTURE = "GESTURE"
    BODY_MOVEMENT = "BODY_MOVEMENT"
    ERROR = "ERROR"
    SYSTEM_INFO = "SYSTEM_INFO"


# Color mapping matching specification
EVENT_COLORS: Dict[EventType, str] = {
    EventType.PERSON_DETECTION: "#22c55e",  # Green
    EventType.NEUTRAL_POSTURE: "#f8fafc",   # White
    EventType.HAND_MOVEMENT: "#38bdf8",     # Cyan
    EventType.GESTURE: "#facc15",           # Yellow
    EventType.BODY_MOVEMENT: "#c084fc",     # Purple
    EventType.ERROR: "#ef4444",             # Red
    EventType.SYSTEM_INFO: "#94a3b8",       # Gray
}


@dataclass
class MovementEvent:
    timestamp: float                       # Unix timestamp or video-relative seconds
    formatted_time: str                    # Human-readable timestamp string [HH:MM:SS]
    event_type: EventType                  # Event category
    description: str                       # Action text e.g. "Right hand raised"
    confidence: Optional[float] = None     # Action recognition confidence (0.0 - 1.0)
    person_id: Optional[int] = None        # Person tracking identifier
    is_video_relative: bool = False        # True if timestamp is from uploaded video
    subject_name: Optional[str] = None     # Recognized identity, if known

    @property
    def color(self) -> str:
        return EVENT_COLORS.get(self.event_type, "#f8fafc")

    def _show_id(self) -> bool:
        # Unrecognized people are told apart by their tracking ID
        return self.person_id is not None and self.subject_name is None

    def formatted_line(self) -> str:
        """Returns the formatted plain text line according to reference."""
        conf_str = f" (conf: {self.confidence:.2f})" if self.confidence is not None else ""
        id_str = f" (ID: {self.person_id})" if self._show_id() else ""
        return f"[{self.formatted_time}] {self.description}{id_str}{conf_str}"

    def formatted_html(self) -> str:
        """Returns HTML formatted line with appropriate color tag."""
        time_color = "#64748b"  # Blue-gray for timestamps
        event_color = self.color
        conf_str = f" <span style='color: #94a3b8;'>(conf: {self.confidence:.2f})</span>" if self.confidence is not None else ""
        id_str = f" <span style='color: #94a3b8;'>(ID: {self.person_id})</span>" if self._show_id() else ""
        return f"<span style='color: {time_color};'>[{self.formatted_time}]</span> <span style='color: {event_color}; font-weight: 500;'>{self.description}{id_str}{conf_str}</span>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "formatted_time": self.formatted_time,
            "event_type": self.event_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "person_id": self.person_id,
            "subject_name": self.subject_name,
            "is_video_relative": self.is_video_relative,
        }


# Phrases used after a recognized person's name, e.g. "Maria is clapping"
GESTURE_PHRASES: Dict[str, str] = {
    "Waving": "appears to be waving",
    "Clapping": "is clapping",
    "Jumping": "jumps",
    "Nodding": "is nodding",
    "Shaking head": "is shaking head",
}

# Seconds to wait for face recognition before announcing an unnamed person
NAME_WAIT_SECONDS = 0.7


class _PersonState:
    """Suppression / transition state for one tracked person."""

    def __init__(self) -> None:
        self.present = False
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.first_hit_time: Optional[float] = None
        self.posture: Optional[str] = None
        self.left_hand_raised = False
        self.right_hand_raised = False
        self.both_hands_raised = False
        self.subject_name: Optional[str] = None
        self.subject_conf: Optional[float] = None
        self.activities: set = set()
        self.last_times: Dict[str, float] = {}


class EventManager(QObject):
    """
    Central event manager that receives detected actions, suppresses duplicates,
    filters noise, formats logs, and notifies UI listeners.
    """
    event_added = Signal(MovementEvent)
    events_cleared = Signal()

    def __init__(self, max_entries: int = 500):
        super().__init__()
        self.max_entries = max_entries
        self.events: List[MovementEvent] = []

        self.hit_threshold: int = 2
        self.miss_threshold: int = 8
        self.gesture_cooldown: float = 1.6   # seconds between same gesture
        self.movement_cooldown: float = 1.8  # seconds between body movement events

        self.people: Dict[int, _PersonState] = {}

    def clear(self) -> None:
        """Clears all stored events."""
        self.events.clear()
        self.events_cleared.emit()

    def reset_state(self) -> None:
        """Resets temporal suppression state (e.g., when switching sources or stopping)."""
        self.people.clear()

    def add_event(self, event: MovementEvent) -> None:
        """Appends and broadcasts a new event."""
        if len(self.events) >= self.max_entries:
            self.events.pop(0)
        self.events.append(event)
        self.event_added.emit(event)

    def log_system_info(self, message: str, timestamp: Optional[float] = None) -> None:
        """Emits a system information event."""
        t = timestamp if timestamp is not None else time.time()
        time_str = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        evt = MovementEvent(
            timestamp=t,
            formatted_time=time_str,
            event_type=EventType.SYSTEM_INFO,
            description=message
        )
        self.add_event(evt)

    def log_error(self, message: str, timestamp: Optional[float] = None) -> None:
        """Emits an error event."""
        t = timestamp if timestamp is not None else time.time()
        time_str = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        evt = MovementEvent(
            timestamp=t,
            formatted_time=time_str,
            event_type=EventType.ERROR,
            description=message
        )
        self.add_event(evt)

    @staticmethod
    def _format_time(current_time: float, is_video_relative: bool) -> str:
        if is_video_relative:
            secs = int(current_time)
            mins = secs // 60
            hrs = mins // 60
            return f"{hrs % 24:02d}:{mins % 60:02d}:{secs % 60:02d}"
        return datetime.fromtimestamp(current_time).strftime("%H:%M:%S")

    def person_left(self, person_id: int, current_time: float, is_video_relative: bool = False) -> None:
        """Called when a tracked person has been out of view long enough to count as gone."""
        state = self.people.pop(person_id, None)
        if state is None or not state.present:
            return
        subj = state.subject_name
        self.add_event(MovementEvent(
            timestamp=current_time,
            formatted_time=self._format_time(current_time, is_video_relative),
            event_type=EventType.PERSON_DETECTION,
            description=f"{subj} left the frame" if subj else "Person left the frame",
            person_id=person_id,
            is_video_relative=is_video_relative,
            subject_name=subj,
        ))

    def process_frame_analysis(
        self,
        person_detected: bool,
        person_id: Optional[int],
        posture: Optional[str],
        left_hand_raised: bool,
        right_hand_raised: bool,
        gestures: List[Dict[str, Any]],
        body_movements: List[Dict[str, Any]],
        current_time: float,
        is_video_relative: bool = False,
        subject_name: Optional[str] = None,
        subject_confidence: Optional[float] = None,
        activities: Optional[Iterable[Dict[str, Any]]] = None,
        activity_phrases: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Ingests real-time frame analysis for one person and performs stateful
        transition & suppression logic. Incorporates subject identification and
        converts distinct actions to structured events.
        """
        pid = person_id if person_id is not None else 1
        state = self.people.setdefault(pid, _PersonState())
        time_str = self._format_time(current_time, is_video_relative)

        def emit(event_type: EventType, desc: str, conf: Optional[float] = None) -> None:
            self.add_event(MovementEvent(
                timestamp=current_time,
                formatted_time=time_str,
                event_type=event_type,
                description=desc,
                confidence=conf,
                person_id=pid,
                is_video_relative=is_video_relative,
                subject_name=state.subject_name,
            ))

        # 1. Person Detection State Transition with Tracking-Loss Hysteresis
        if person_detected:
            state.consecutive_hits += 1
            state.consecutive_misses = 0
            if state.first_hit_time is None:
                state.first_hit_time = current_time
        else:
            state.consecutive_misses += 1
            state.consecutive_hits = 0

        if not state.present:
            if not person_detected:
                state.first_hit_time = None
                return
            # Announce once a name is known, or after a short wait for recognition
            waited = current_time - (state.first_hit_time if state.first_hit_time is not None else current_time)
            if state.consecutive_hits >= self.hit_threshold and (subject_name or waited >= NAME_WAIT_SECONDS):
                state.present = True
                if subject_name:
                    state.subject_name = subject_name
                    state.subject_conf = subject_confidence
                    emit(EventType.PERSON_DETECTION, f"{subject_name} detected", subject_confidence)
                else:
                    emit(EventType.PERSON_DETECTION, "Person detected")
            else:
                return
        elif state.consecutive_misses >= self.miss_threshold:
            # Person leaves frame (must miss multiple consecutive frames)
            self.people.pop(pid, None)
            return

        if not person_detected:
            return

        # Subject identified or changed while in frame
        if subject_name and subject_name != state.subject_name:
            prev_name = state.subject_name
            state.subject_name = subject_name
            state.subject_conf = subject_confidence
            desc = f"{subject_name} recognized" if prev_name is None else f"Identified as {subject_name}"
            emit(EventType.PERSON_DETECTION, desc, subject_confidence)

        subj = state.subject_name

        # 2. Hand State Transitions (Raised / Lowered)
        both_raised = left_hand_raised and right_hand_raised

        if both_raised and not state.both_hands_raised:
            state.both_hands_raised = True
            state.left_hand_raised = True
            state.right_hand_raised = True
            emit(EventType.HAND_MOVEMENT, f"{subj} raises both hands" if subj else "Both hands raised")
        elif not both_raised and state.both_hands_raised:
            state.both_hands_raised = False

        if not both_raised:
            for side, raised in (("right", right_hand_raised), ("left", left_hand_raised)):
                attr = f"{side}_hand_raised"
                was = getattr(state, attr)
                if raised and not was:
                    setattr(state, attr, True)
                    emit(EventType.HAND_MOVEMENT, f"{subj} raises {side} hand" if subj else f"{side.capitalize()} hand raised")
                elif not raised and was:
                    setattr(state, attr, False)
                    emit(EventType.HAND_MOVEMENT, f"{subj} lowers {side} hand" if subj else f"{side.capitalize()} hand lowered")

        # 3. Gestures (momentary, rate-limited)
        for gesture in gestures:
            name = gesture.get("name", "Gesture")
            conf = gesture.get("confidence", 0.90)
            if current_time - state.last_times.get(name, -1e9) >= self.gesture_cooldown:
                state.last_times[name] = current_time
                phrase = GESTURE_PHRASES.get(name, name.lower())
                emit(EventType.GESTURE, f"{subj} {phrase}" if subj else name, conf)

        # 4. Body Movements (e.g., Turning, Leaning, Stepping): logged when they start,
        #    not repeatedly while a person stays turned / leaning
        for movement in body_movements:
            name = movement.get("name", "Movement")
            conf = movement.get("confidence")
            seen_key = f"seen:{name}"
            was_ongoing = current_time - state.last_times.get(seen_key, -1e9) <= 1.0
            state.last_times[seen_key] = current_time
            if not was_ongoing and current_time - state.last_times.get(name, -1e9) >= self.movement_cooldown:
                state.last_times[name] = current_time
                emit(EventType.BODY_MOVEMENT, f"{subj} is {name.lower()}" if subj else name, conf)

        # 5. Sustained activities (Idle, Walking, Arms crossed, ...): log when they start
        if activities is not None:
            current = {a["name"]: a.get("confidence") for a in activities}
            for name, conf in current.items():
                if name not in state.activities:
                    phrase = (activity_phrases or {}).get(name, f"is {name.lower()}")
                    event_type = EventType.NEUTRAL_POSTURE if name == "Idle" else EventType.BODY_MOVEMENT
                    emit(event_type, f"{subj} {phrase}" if subj else name, conf)
            state.activities = set(current)

        # 6. Neutral Posture Transitions (Standing / Sitting / Crouching / Lying down)
        if posture and posture != state.posture:
            state.posture = posture
            emit(EventType.NEUTRAL_POSTURE, f"{subj} is {posture.lower()}" if subj else posture)

    def export_csv(self, filepath: str) -> bool:
        """Exports all recorded events to CSV."""
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Formatted Time", "Type", "Description", "Confidence", "Person ID", "Subject", "Video Relative"])
                for evt in self.events:
                    writer.writerow([
                        evt.timestamp,
                        evt.formatted_time,
                        evt.event_type.value,
                        evt.description,
                        f"{evt.confidence:.2f}" if evt.confidence is not None else "",
                        evt.person_id if evt.person_id is not None else "",
                        evt.subject_name or "",
                        evt.is_video_relative
                    ])
            return True
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False

    def export_json(self, filepath: str) -> bool:
        """Exports all recorded events to JSON."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                data = [evt.to_dict() for evt in self.events]
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            return False
