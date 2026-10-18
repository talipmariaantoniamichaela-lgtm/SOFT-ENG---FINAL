"""
Configuration and Settings module for Motion Reader.
"""
import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")


@dataclass
class VideoSettings:
    camera_index: int = 0
    camera_name: str = "Camera (Default)"
    preferred_width: int = 1280
    preferred_height: int = 720
    target_fps: int = 30


@dataclass
class DetectionSettings:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    action_sensitivity: float = 1.0
    landmark_smoothing: bool = True
    smoothing_factor: float = 0.55
    enable_gesture_recognition: bool = True
    enable_hand_tracking: bool = True
    enable_face_tracking: bool = True


@dataclass
class DisplaySettings:
    show_skeleton: bool = True
    show_landmarks: bool = True
    show_hands: bool = True
    show_face: bool = True
    show_fps_badge: bool = True
    show_corner_markers: bool = True
    show_person_id: bool = True
    mirror_camera: bool = True


@dataclass
class TerminalSettings:
    auto_scroll: bool = True
    timestamp_format: str = "%H:%M:%S"
    max_retained_entries: int = 500
    export_format: str = "CSV"


@dataclass
class AppConfig:
    video: VideoSettings = field(default_factory=VideoSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    terminal: TerminalSettings = field(default_factory=TerminalSettings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        config = cls()
        if "video" in data:
            config.video = VideoSettings(**{k: v for k, v in data["video"].items() if hasattr(config.video, k)})
        if "detection" in data:
            config.detection = DetectionSettings(**{k: v for k, v in data["detection"].items() if hasattr(config.detection, k)})
        if "display" in data:
            config.display = DisplaySettings(**{k: v for k, v in data["display"].items() if hasattr(config.display, k)})
        if "terminal" in data:
            config.terminal = TerminalSettings(**{k: v for k, v in data["terminal"].items() if hasattr(config.terminal, k)})
        return config

    def save(self, filepath: str = SETTINGS_FILE) -> None:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    @classmethod
    def load(cls, filepath: str = SETTINGS_FILE) -> 'AppConfig':
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return cls.from_dict(data)
            except Exception as e:
                print(f"Error loading settings, using defaults: {e}")
        return cls()
