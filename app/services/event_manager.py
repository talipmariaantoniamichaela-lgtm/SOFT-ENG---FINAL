"""
Event Manager and Event Dispatcher for Motion Reader.
Handles movement events, color coding, suppression of repeated events,
and export to CSV/JSON.
"""
import csv
import json
import time
from datetime import datetime
from enum import Enum
from typing import List, Optional, Callable, Dict, Any
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

    @property
    def color(self) -> str:
        return EVENT_COLORS.get(self.event_type, "#f8fafc")

    def formatted_line(self) -> str:
        """Returns the formatted plain text line according to reference."""
        conf_str = f" (conf: {self.confidence:.2f})" if self.confidence is not None else ""
        id_str = f" (ID: {self.person_id})" if self.person_id is not None and self.description == "Person detected" else ""
        return f"[{self.formatted_time}] {self.description}{id_str}{conf_str}"

    def formatted_html(self) -> str:
        """Returns HTML formatted line with appropriate color tag."""
        time_color = "#64748b"  # Blue-gray for timestamps
        event_color = self.color
        conf_str = f" <span style='color: #94a3b8;'>(conf: {self.confidence:.2f})</span>" if self.confidence is not None else ""
        id_str = f" <span style='color: #94a3b8;'>(ID: {self.person_id})</span>" if self.person_id is not None and self.description == "Person detected" else ""
        return f"<span style='color: {time_color};'>[{self.formatted_time}]</span> <span style='color: {event_color}; font-weight: 500;'>{self.description}{id_str}{conf_str}</span>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "formatted_time": self.formatted_time,
            "event_type": self.event_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "person_id": self.person_id,
            "is_video_relative": self.is_video_relative,
        }


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

        # State tracking for suppression
        self.last_person_detected: bool = False
        self.last_person_id: Optional[int] = None
        self.consecutive_hits: int = 0
        self.consecutive_misses: int = 0
        self.hit_threshold: int = 2
        self.miss_threshold: int = 8
        self.current_posture: Optional[str] = None
        self.last_left_hand_raised: bool = False
        self.last_right_hand_raised: bool = False
        self.last_both_hands_raised: bool = False
        self.last_subject_name: Optional[str] = None
        self.last_subject_conf: Optional[float] = None

        # Gesture & movement cooldown tracking
        self.last_gesture_times: Dict[str, float] = {}
        self.gesture_cooldown: float = 1.6  # seconds between same gesture
        self.movement_cooldown: float = 1.8  # seconds between body movement events
        self.posture_lock_time: float = 0.0

    def clear(self) -> None:
        """Clears all stored events."""
        self.events.clear()
        self.events_cleared.emit()

    def reset_state(self) -> None:
        """Resets temporal suppression state (e.g., when switching sources or stopping)."""
        self.last_person_detected = False
        self.last_person_id = None
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.current_posture = None
        self.last_left_hand_raised = False
        self.last_right_hand_raised = False
        self.last_both_hands_raised = False
        self.last_subject_name = None
        self.last_subject_conf = None
        self.last_gesture_times.clear()
        self.posture_lock_time = 0.0

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
        subject_confidence: Optional[float] = None
    ) -> None:
        """
        Ingests real-time frame analysis and performs stateful transition & suppression logic.
        Incorporates subject identification and converts distinct actions to structured events.
        """
        # Format time string
        if is_video_relative:
            secs = int(current_time)
            mins = secs // 60
            hrs = mins // 60
            time_str = f"{hrs % 24:02d}:{mins % 60:02d}:{secs % 60:02d}"
        else:
            time_str = datetime.fromtimestamp(current_time).strftime("%H:%M:%S")

        # 1. Person Detection State Transition with Tracking-Loss Hysteresis
        if person_detected:
            self.consecutive_hits += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            self.consecutive_hits = 0

        # Person enters frame
        if self.consecutive_hits >= self.hit_threshold and not self.last_person_detected:
            self.last_person_detected = True
            self.last_person_id = person_id or 1
            if subject_name:
                self.last_subject_name = subject_name
                self.last_subject_conf = subject_confidence
                desc = f"{subject_name} detected"
                conf = subject_confidence
            else:
                desc = "Person detected"
                conf = None

            evt = MovementEvent(
                timestamp=current_time,
                formatted_time=time_str,
                event_type=EventType.PERSON_DETECTION,
                description=desc,
                confidence=conf,
                person_id=self.last_person_id,
                is_video_relative=is_video_relative
            )
            self.add_event(evt)
        # Person leaves frame (must miss multiple consecutive frames)
        elif self.consecutive_misses >= self.miss_threshold and self.last_person_detected:
            self.last_person_detected = False
            self.last_person_id = None
            self.last_subject_name = None
            self.last_subject_conf = None
            self.current_posture = None
            self.last_left_hand_raised = False
            self.last_right_hand_raised = False
            self.last_both_hands_raised = False
            return

        if not self.last_person_detected:
            return

        # Check if subject was identified or updated while in frame
        if subject_name and subject_name != self.last_subject_name:
            prev_name = self.last_subject_name
            self.last_subject_name = subject_name
            self.last_subject_conf = subject_confidence
            desc = f"{subject_name} recognized" if prev_name is None else f"Identified as {subject_name}"
            evt = MovementEvent(
                timestamp=current_time,
                formatted_time=time_str,
                event_type=EventType.PERSON_DETECTION,
                description=desc,
                confidence=subject_confidence,
                person_id=self.last_person_id,
                is_video_relative=is_video_relative
            )
            self.add_event(evt)

        subj = self.last_subject_name

        # 2. Hand State Transitions (Raised / Lowered)
        both_raised = left_hand_raised and right_hand_raised

        # Both hands raised
        if both_raised and not self.last_both_hands_raised:
            self.last_both_hands_raised = True
            self.last_left_hand_raised = True
            self.last_right_hand_raised = True
            desc = f"{subj} raises both hands" if subj else "Both hands raised"
            evt = MovementEvent(
                timestamp=current_time,
                formatted_time=time_str,
                event_type=EventType.HAND_MOVEMENT,
                description=desc,
                is_video_relative=is_video_relative
            )
            self.add_event(evt)
        elif not both_raised and self.last_both_hands_raised:
            self.last_both_hands_raised = False

        # Individual hand transitions if not both hands simultaneously triggering
        if not both_raised:
            # Right hand raised
            if right_hand_raised and not self.last_right_hand_raised:
                self.last_right_hand_raised = True
                desc = f"{subj} raises right hand" if subj else "Right hand raised"
                evt = MovementEvent(
                    timestamp=current_time,
                    formatted_time=time_str,
                    event_type=EventType.HAND_MOVEMENT,
                    description=desc,
                    is_video_relative=is_video_relative
                )
                self.add_event(evt)
            # Right hand lowered
            elif not right_hand_raised and self.last_right_hand_raised:
                self.last_right_hand_raised = False
                desc = f"{subj} lowers right hand" if subj else "Right hand lowered"
                evt = MovementEvent(
                    timestamp=current_time,
                    formatted_time=time_str,
                    event_type=EventType.HAND_MOVEMENT,
                    description=desc,
                    is_video_relative=is_video_relative
                )
                self.add_event(evt)

            # Left hand raised
            if left_hand_raised and not self.last_left_hand_raised:
                self.last_left_hand_raised = True
                desc = f"{subj} raises left hand" if subj else "Left hand raised"
                evt = MovementEvent(
                    timestamp=current_time,
                    formatted_time=time_str,
                    event_type=EventType.HAND_MOVEMENT,
                    description=desc,
                    is_video_relative=is_video_relative
                )
                self.add_event(evt)
            # Left hand lowered
            elif not left_hand_raised and self.last_left_hand_raised:
                self.last_left_hand_raised = False
                desc = f"{subj} lowers left hand" if subj else "Left hand lowered"
                evt = MovementEvent(
                    timestamp=current_time,
                    formatted_time=time_str,
                    event_type=EventType.HAND_MOVEMENT,
                    description=desc,
                    is_video_relative=is_video_relative
                )
                self.add_event(evt)

        # 3. Gestures (e.g., Waving)
        for gesture in gestures:
            name = gesture.get("name", "Gesture")
            conf = gesture.get("confidence", 0.90)
            last_t = self.last_gesture_times.get(name, 0.0)
            if current_time - last_t >= self.gesture_cooldown:
                self.last_gesture_times[name] = current_time
                desc = f"{subj} appears to be waving" if (subj and "Waving" in name) else (f"{subj}: {name}" if subj else name)
                evt = MovementEvent(
                    timestamp=current_time,
                    formatted_time=time_str,
                    event_type=EventType.GESTURE,
                    description=desc,
                    confidence=conf,
                    is_video_relative=is_video_relative
                )
                self.add_event(evt)

        # 4. Body Movements (e.g., Turning, Leaning, Stepping)
        for movement in body_movements:
            name = movement.get("name", "Movement")
            conf = movement.get("confidence")
            last_t = self.last_gesture_times.get(name, 0.0)
            if current_time - last_t >= self.movement_cooldown:
                self.last_gesture_times[name] = current_time
                desc = f"{subj} is {name.lower()}" if subj else name
                evt = MovementEvent(
                    timestamp=current_time,
                    formatted_time=time_str,
                    event_type=EventType.BODY_MOVEMENT,
                    description=desc,
                    confidence=conf,
                    is_video_relative=is_video_relative
                )
                self.add_event(evt)

        # 5. Neutral Posture Transitions (Standing / Sitting / Crouching)
        if posture:
            if posture != self.current_posture:
                self.current_posture = posture
                desc = f"{subj} is {posture.lower()}" if subj else posture
                evt = MovementEvent(
                    timestamp=current_time,
                    formatted_time=time_str,
                    event_type=EventType.NEUTRAL_POSTURE,
                    description=desc,
                    is_video_relative=is_video_relative
                )
                self.add_event(evt)

    def export_csv(self, filepath: str) -> bool:
        """Exports all recorded events to CSV."""
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Formatted Time", "Type", "Description", "Confidence", "Person ID", "Video Relative"])
                for evt in self.events:
                    writer.writerow([
                        evt.timestamp,
                        evt.formatted_time,
                        evt.event_type.value,
                        evt.description,
                        f"{evt.confidence:.2f}" if evt.confidence is not None else "",
                        evt.person_id if evt.person_id is not None else "",
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
