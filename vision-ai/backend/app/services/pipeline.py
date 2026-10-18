"""
Video analysis pipeline for VISION AI.
Runs MediaPipe detection + movement classification in a background thread
and broadcasts events over WebSocket.
"""
import asyncio
import time
import threading
import queue
import cv2
import numpy as np
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from app.detectors.vision_detector import VisionDetector, DetectionFrame
from app.movement.classifier import MovementClassifier, DetectedAction
from app.core.config import settings


class AnalysisPipeline:
    """
    Manages a background thread that:
    1. Reads frames from a camera or video file.
    2. Runs VisionDetector inference.
    3. Runs MovementClassifier.
    4. Puts structured events on an asyncio-safe queue for WebSocket delivery.
    """

    def __init__(
        self,
        session_id: str,
        config: Dict[str, Any],
        event_callback: Callable[[Dict], None],
    ):
        self.session_id = session_id
        self.config = config
        self.event_callback = event_callback   # sync callback, called from thread

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()   # not paused initially

        # Detector configuration
        self._detector = VisionDetector(
            enable_face=config.get("enable_face", True),
            enable_pose=config.get("enable_pose", True),
            enable_hands=config.get("enable_hands", True),
            detection_confidence=config.get("detection_confidence", 0.5),
            tracking_confidence=config.get("tracking_confidence", 0.5),
        )
        self._classifier = MovementClassifier(
            sensitivity=config.get("sensitivity", 1.0)
        )

        # Processing config
        self._inference_w = config.get("inference_width", 640)
        self._inference_h = config.get("inference_height", 480)
        processing_mode = config.get("processing_mode", "balanced")
        self._frame_skip = {"performance": 3, "balanced": 2, "accuracy": 1}.get(processing_mode, 2)

        # FPS tracking
        self._fps_ts: list = []
        self._latency_ms = 0.0
        self._person_count = 0
        self._face_count = 0
        self._hand_count = 0
        self._frame_count = 0

        # Source
        self.input_type = config.get("input_type", "live_camera")
        self.camera_index = config.get("camera_index", 0)
        self.video_path: Optional[str] = config.get("video_path")

        # Video state
        self._video_paused = False
        self._seek_target: Optional[float] = None   # 0.0–1.0

    @property
    def models_loaded(self) -> Dict[str, bool]:
        return self._detector.models_loaded

    def start(self):
        self._stop_event.clear()
        target = (self._run_camera if self.input_type == "live_camera"
                  else self._run_video)
        self._thread = threading.Thread(target=target, daemon=True, name=f"pipeline-{self.session_id[:8]}")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._paused.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._detector.close()

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def seek(self, progress: float):
        self._seek_target = max(0.0, min(1.0, progress))

    def _emit(self, event: Dict):
        try:
            self.event_callback(event)
        except Exception:
            pass

    def _measure_fps(self) -> float:
        now = time.time()
        self._fps_ts.append(now)
        self._fps_ts = [t for t in self._fps_ts if now - t <= 1.0]
        return float(len(self._fps_ts))

    def _build_stats_event(self, fps: float, timestamp_ms: int) -> Dict:
        return {
            "type": "frame_stats",
            "session_id": self.session_id,
            "fps": round(fps, 1),
            "latency_ms": round(self._latency_ms, 1),
            "person_count": self._person_count,
            "face_count": self._face_count,
            "hand_count": self._hand_count,
            "processing_mode": self.config.get("processing_mode", "balanced"),
            "device": settings.get_device(),
            "timestamp_ms": timestamp_ms,
        }

    def _process_frame(self, rgb_frame: np.ndarray, timestamp_ms: int, timestamp_s: float):
        """Detect + classify + emit events for a single frame."""
        t0 = time.perf_counter()
        frame = cv2.resize(rgb_frame, (self._inference_w, self._inference_h))

        detection: DetectionFrame = self._detector.detect(frame, timestamp_ms)
        self._latency_ms = (time.perf_counter() - t0) * 1000

        self._person_count = len(detection.persons)
        self._face_count = sum(1 for p in detection.persons if p.face is not None)
        self._hand_count = sum(
            (1 if p.left_hand else 0) + (1 if p.right_hand else 0)
            for p in detection.persons
        )

        for person in detection.persons:
            actions: list[DetectedAction] = self._classifier.classify(person, timestamp_s)
            for action in actions:
                self._emit({
                    "type": "movement_event",
                    "session_id": self.session_id,
                    "timestamp_ms": timestamp_ms,
                    "person_id": f"person-{action.person_id:02d}",
                    "category": action.category,
                    "action": action.action,
                    "description": action.description,
                    "confidence": action.confidence,
                    "source": action.source,
                    "severity": action.severity,
                })

        fps = self._measure_fps()
        self._frame_count += 1
        # Emit stats every 15 processed frames
        if self._frame_count % 15 == 0:
            self._emit(self._build_stats_event(fps, timestamp_ms))

        # If person count changed, emit detect event
        # (handled in start logic via initial detect)

    def _run_camera(self):
        """Camera acquisition loop."""
        self._emit({
            "type": "movement_event", "session_id": self.session_id,
            "timestamp_ms": 0, "person_id": None, "category": "SYSTEM",
            "action": "camera_init", "description": "Requesting camera stream...",
            "confidence": None, "source": "system", "severity": "info",
        })

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._emit({
                "type": "movement_event", "session_id": self.session_id,
                "timestamp_ms": 0, "person_id": None, "category": "ERROR",
                "action": "camera_error",
                "description": f"Camera access failed: Cannot open camera {self.camera_index}.",
                "confidence": None, "source": "system", "severity": "alert",
            })
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._emit({
            "type": "movement_event", "session_id": self.session_id,
            "timestamp_ms": 0, "person_id": None, "category": "SYSTEM",
            "action": "camera_ready", "description": "Camera initialized successfully.",
            "confidence": None, "source": "system", "severity": "info",
        })

        frame_n = 0
        start_time = time.time()

        try:
            while not self._stop_event.is_set():
                self._paused.wait()
                if self._stop_event.is_set():
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)
                frame_n += 1
                if frame_n % self._frame_skip != 0:
                    continue

                now = time.time()
                timestamp_ms = int((now - start_time) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._process_frame(rgb, timestamp_ms, now - start_time)
                time.sleep(0.001)
        finally:
            cap.release()

    def _run_video(self):
        """Video file analysis loop."""
        if not self.video_path:
            self._emit({
                "type": "movement_event", "session_id": self.session_id,
                "timestamp_ms": 0, "person_id": None, "category": "ERROR",
                "action": "video_error",
                "description": "No video file specified.",
                "confidence": None, "source": "system", "severity": "alert",
            })
            return

        vpath = Path(self.video_path)
        if not vpath.exists():
            vpath = settings.UPLOAD_DIR / vpath.name

        if not vpath.exists():
            self._emit({
                "type": "movement_event", "session_id": self.session_id,
                "timestamp_ms": 0, "person_id": None, "category": "ERROR",
                "action": "video_error",
                "description": f"Video file not found: {Path(self.video_path).name}",
                "confidence": None, "source": "system", "severity": "alert",
            })
            return

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            self._emit({
                "type": "movement_event", "session_id": self.session_id,
                "timestamp_ms": 0, "person_id": None, "category": "ERROR",
                "action": "video_error",
                "description": f"Cannot open video file: {Path(self.video_path).name}",
                "confidence": None, "source": "system", "severity": "alert",
            })
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_s = total_frames / video_fps if total_frames > 0 else 1.0
        frame_interval = 1.0 / max(10, min(60, video_fps))
        name = Path(self.video_path).name

        self._emit({
            "type": "movement_event", "session_id": self.session_id,
            "timestamp_ms": 0, "person_id": None, "category": "SYSTEM",
            "action": "video_loaded",
            "description": f"Loaded video: {name} ({duration_s:.1f}s, {video_fps:.0f} fps)",
            "confidence": None, "source": "system", "severity": "info",
        })

        frame_n = 0
        try:
            while not self._stop_event.is_set():
                self._paused.wait()
                if self._stop_event.is_set():
                    break

                # Handle seek
                if self._seek_target is not None:
                    target_frame = int(self._seek_target * total_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    self._classifier.reset()
                    self._seek_target = None

                t_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    # End of video
                    self._emit({
                        "type": "movement_event", "session_id": self.session_id,
                        "timestamp_ms": int(duration_s * 1000), "person_id": None,
                        "category": "SYSTEM", "action": "video_complete",
                        "description": "Video analysis complete.",
                        "confidence": None, "source": "system", "severity": "info",
                    })
                    break

                curr_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                curr_s = curr_frame_idx / video_fps
                timestamp_ms = int(curr_s * 1000)

                frame_n += 1
                if frame_n % self._frame_skip != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._process_frame(rgb, timestamp_ms, curr_s)

                # Also emit video position
                self._emit({
                    "type": "video_position",
                    "session_id": self.session_id,
                    "current_s": round(curr_s, 2),
                    "duration_s": round(duration_s, 2),
                    "progress": round(curr_s / max(duration_s, 0.001), 4),
                })

                # Frame pacing
                elapsed = time.time() - t_start
                sleep = max(0.001, frame_interval - elapsed)
                time.sleep(sleep)
        finally:
            cap.release()
