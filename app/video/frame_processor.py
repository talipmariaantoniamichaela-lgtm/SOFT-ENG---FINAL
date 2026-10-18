"""
Background Video Frame Processor Worker Thread for Motion Reader.
Performs frame acquisition, MediaPipe inference, multi-person tracking,
face identification, temporal analysis, and measured FPS tracking off the GUI thread.
"""
import os
import time
import cv2
import numpy as np
from enum import Enum
from typing import List, Optional
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from app.detection.pose_estimator import PoseEstimator
from app.detection.motion_analyzer import ACTIVITY_PHRASES
from app.detection.face_identity import FaceIdentifier
from app.detection.person_tracker import PersonTracker, PersonSnapshot
from app.services.event_manager import EventManager
from app.config import AppConfig


class SourceMode(Enum):
    LIVE_CAMERA = "LIVE_CAMERA"
    UPLOADED_VIDEO = "UPLOADED_VIDEO"


# Run face recognition on every Nth frame (pose runs on every frame)
IDENTIFY_EVERY_N_FRAMES = 3


class FrameProcessorThread(QThread):
    """
    Worker thread that captures frames, runs AI pose estimation, face recognition
    & motion analysis for every person in view, and emits results to the UI.
    """
    frame_ready = Signal(object, object, float, str)          # (rgb_frame, List[PersonSnapshot], fps, res_str)
    video_position_changed = Signal(float, float, float)     # (current_sec, total_sec, progress_pct)
    status_changed = Signal(str, str)                         # (status_text, status_type)
    source_changed = Signal(SourceMode, str)                  # (mode, description)
    error_occurred = Signal(str)                              # (error_message)

    def __init__(self, config: AppConfig, event_manager: EventManager):
        super().__init__()
        self.config = config
        self.event_manager = event_manager

        # Processing state
        self._is_running = False
        self._is_paused = False
        self._mutex = QMutex()
        self._wait_cond = QWaitCondition()

        # Current source config
        self.source_mode = SourceMode.LIVE_CAMERA
        self.camera_index = config.video.camera_index
        self.video_filepath: Optional[str] = None

        # Video file playback state
        self._seek_requested = False
        self._seek_target_pct = 0.0

        # Sub-modules
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=config.detection.min_detection_confidence,
            min_tracking_confidence=config.detection.min_tracking_confidence,
            enable_hands=config.detection.enable_hand_tracking,
            enable_face=config.detection.enable_face_tracking,
            smoothing=config.detection.landmark_smoothing,
            smoothing_factor=config.detection.smoothing_factor,
            num_poses=config.detection.max_people,
        )
        self.person_tracker = PersonTracker(
            sensitivity=config.detection.action_sensitivity,
            smoothing=config.detection.landmark_smoothing,
            smoothing_factor=config.detection.smoothing_factor,
        )

        # Face identity (pretrained YuNet + SFace matched against the dataset gallery)
        self.face_identifier = FaceIdentifier(
            threshold=config.detection.face_match_threshold,
            margin=config.detection.face_match_margin,
        )
        self._frame_counter = 0

        # FPS measurement
        self.fps_timestamps = []

    def apply_detection_settings(self, config: AppConfig) -> None:
        self.config = config
        self.pose_estimator.min_detection_confidence = config.detection.min_detection_confidence
        self.pose_estimator.min_tracking_confidence = config.detection.min_tracking_confidence
        self.pose_estimator.enable_hands = config.detection.enable_hand_tracking
        self.pose_estimator.enable_face = config.detection.enable_face_tracking
        self.person_tracker.smoothing = config.detection.landmark_smoothing
        self.person_tracker.set_sensitivity(config.detection.action_sensitivity)
        if config.detection.face_match_threshold is not None:
            self.face_identifier.threshold = config.detection.face_match_threshold
        if config.detection.face_match_margin is not None:
            self.face_identifier.margin = config.detection.face_match_margin

    def log_identity_status(self) -> None:
        fi = self.face_identifier
        if fi.is_ready:
            self.event_manager.log_system_info(
                f"Face recognition ready: {', '.join(fi.names)} (threshold {fi.threshold:.2f}, margin {fi.margin:.2f})"
            )
        elif fi.models_loaded:
            self.event_manager.log_error("Face gallery missing - run 'python enroll_faces.py' to enroll the dataset")
        else:
            self.event_manager.log_error(fi.error or "Face recognition models unavailable")

    def reload_face_gallery(self) -> bool:
        """Reloads face_gallery.npz after re-enrollment."""
        self._mutex.lock()
        ok = self.face_identifier.load_gallery(self.face_identifier.gallery_path)
        self.person_tracker.reset()
        self._mutex.unlock()
        self.log_identity_status()
        return ok

    def reload_subject_model(self, model_path: str, meta_path: Optional[str] = None) -> bool:
        """
        Kept for the Training Studio. Live identification now uses the face gallery,
        so a newly trained classifier reloads the gallery instead.
        """
        return self.reload_face_gallery()

    def _reset_tracking(self) -> None:
        self.event_manager.reset_state()
        self.person_tracker.reset()
        self.pose_estimator.reset()

    def set_source_camera(self, camera_index: int) -> None:
        """Switches to webcam mode."""
        self._mutex.lock()
        self.source_mode = SourceMode.LIVE_CAMERA
        self.camera_index = camera_index
        self._reset_tracking()
        self._is_paused = False
        self._mutex.unlock()
        self.source_changed.emit(SourceMode.LIVE_CAMERA, f"Camera {camera_index}")

    def set_source_video_file(self, filepath: str) -> None:
        """Switches to uploaded video file mode."""
        self._mutex.lock()
        self.source_mode = SourceMode.UPLOADED_VIDEO
        self.video_filepath = filepath
        self._reset_tracking()
        self._is_paused = False
        self._mutex.unlock()
        self.source_changed.emit(SourceMode.UPLOADED_VIDEO, filepath)

    def play(self) -> None:
        """Resumes processing / playback."""
        self._mutex.lock()
        self._is_paused = False
        self._wait_cond.wakeAll()
        self._mutex.unlock()
        self.status_changed.emit("Tracking...", "active")

    def pause(self) -> None:
        """Pauses processing / playback."""
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()
        self.status_changed.emit("Paused", "warning")

    def toggle_play_pause(self) -> bool:
        """Toggles between play and pause. Returns True if now running, False if paused."""
        self._mutex.lock()
        self._is_paused = not self._is_paused
        is_p = self._is_paused
        if not is_p:
            self._wait_cond.wakeAll()
        self._mutex.unlock()
        if is_p:
            self.status_changed.emit("Paused", "warning")
            return False
        else:
            self.status_changed.emit("Tracking...", "active")
            return True

    def seek_to(self, progress_pct: float) -> None:
        """Requests seeking to a percentage [0.0, 1.0] in the video file."""
        self._mutex.lock()
        self._seek_requested = True
        self._seek_target_pct = max(0.0, min(1.0, progress_pct))
        self._wait_cond.wakeAll()
        self._mutex.unlock()

    def restart_video(self) -> None:
        """Restarts video playback from beginning."""
        self.seek_to(0.0)
        self.event_manager.reset_state()
        self.person_tracker.reset()

    def stop_worker(self) -> None:
        """Stops the worker thread cleanly."""
        self._mutex.lock()
        self._is_running = False
        self._is_paused = False
        self._wait_cond.wakeAll()
        self._mutex.unlock()
        self.wait(1500)

    def run(self) -> None:
        """Main processing thread execution loop."""
        self._is_running = True
        self.status_changed.emit("Initializing...", "neutral")

        while self._is_running:
            if self.source_mode == SourceMode.LIVE_CAMERA:
                self._run_camera_loop()
            else:
                self._run_video_file_loop()

        self.status_changed.emit("Stopped", "neutral")

    def _measure_fps(self, now: float) -> int:
        self.fps_timestamps.append(now)
        while self.fps_timestamps and now - self.fps_timestamps[0] > 1.0:
            self.fps_timestamps.pop(0)
        return len(self.fps_timestamps)

    def _process_frame(self, bgr_frame: np.ndarray, rgb_frame: np.ndarray, t: float,
                       t_ms: int, is_video: bool, fps: float) -> List[PersonSnapshot]:
        """Pose -> tracking -> identity -> motion analysis -> events, for every person in the frame."""
        self._mutex.lock()
        try:
            detections = self.pose_estimator.estimate_all(rgb_frame, timestamp_ms=t_ms)
            visible, departed = self.person_tracker.update(detections, t)

            self._frame_counter += 1
            if visible and self.face_identifier.models_loaded and (
                self._frame_counter % IDENTIFY_EVERY_N_FRAMES == 0
                or any(tr.name is None and tr.age < 30 for tr in visible)
            ):
                h, w = bgr_frame.shape[:2]
                faces = self.face_identifier.identify(bgr_frame)
                self.person_tracker.assign_faces(faces, w, h, visible)

            people = self.person_tracker.snapshot(visible)
        finally:
            self._mutex.unlock()

        for track in departed:
            self.event_manager.person_left(track.track_id, t, is_video_relative=is_video)

        for person in people:
            a = person.analysis
            a.fps = fps
            self.event_manager.process_frame_analysis(
                person_detected=True,
                person_id=person.track_id,
                posture=a.posture,
                left_hand_raised=a.left_hand_raised,
                right_hand_raised=a.right_hand_raised,
                gestures=a.gestures if self.config.detection.enable_gesture_recognition else [],
                body_movements=a.body_movements,
                current_time=t,
                is_video_relative=is_video,
                subject_name=person.name,
                subject_confidence=person.name_score if person.name else None,
                activities=a.activities,
                activity_phrases=ACTIVITY_PHRASES,
            )

        if not people:
            self.status_changed.emit("No Person Detected", "warning")
        else:
            self.status_changed.emit(f"Tracking {len(people)} {'person' if len(people) == 1 else 'people'}", "active")
        return people

    def _run_camera_loop(self) -> None:
        """Execution loop for live camera feed."""
        self.status_changed.emit("Initializing Camera...", "neutral")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.status_changed.emit("Camera Unavailable", "error")
            self.event_manager.log_error(f"Cannot open Camera {self.camera_index}")
            time.sleep(1.0)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.preferred_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.preferred_height)

        self.status_changed.emit("Tracking...", "active")
        self.event_manager.log_system_info("Camera connected successfully")
        self.log_identity_status()
        self.event_manager.log_system_info("Movement detection started")

        self.fps_timestamps.clear()

        try:
            while self._is_running and self.source_mode == SourceMode.LIVE_CAMERA:
                # Handle paused state
                self._mutex.lock()
                while self._is_paused and self._is_running and self.source_mode == SourceMode.LIVE_CAMERA:
                    self._wait_cond.wait(self._mutex, 200)
                self._mutex.unlock()

                if not self._is_running or self.source_mode != SourceMode.LIVE_CAMERA:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                # Mirror if configured
                if self.config.display.mirror_camera:
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape
                res_str = f"{w} × {h}"
                now = time.time()
                measured_fps = self._measure_fps(now)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                people = self._process_frame(frame, rgb_frame, now, int(now * 1000), False, float(measured_fps))
                self.frame_ready.emit(rgb_frame, people, float(measured_fps), res_str)

                # Small yield to maintain ~30 FPS if camera runs faster
                time.sleep(0.005)
        finally:
            cap.release()

    def _run_video_file_loop(self) -> None:
        """Execution loop for uploaded video file playback."""
        if not self.video_filepath:
            time.sleep(0.2)
            return

        cap = cv2.VideoCapture(self.video_filepath)
        if not cap.isOpened():
            self.status_changed.emit("Invalid Video File", "error")
            self.event_manager.log_error(f"Cannot open video file: {self.video_filepath}")
            time.sleep(1.0)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / video_fps if total_frames > 0 else 1.0
        frame_interval = 1.0 / max(10.0, min(60.0, video_fps))

        self.status_changed.emit("Tracking...", "active")
        self.event_manager.log_system_info(f"Video opened: {os.path.basename(self.video_filepath)}")
        self.log_identity_status()
        self.event_manager.log_system_info("Movement detection started")

        self.fps_timestamps.clear()

        try:
            while self._is_running and self.source_mode == SourceMode.UPLOADED_VIDEO:
                # Handle paused state
                self._mutex.lock()
                while self._is_paused and self._is_running and self.source_mode == SourceMode.UPLOADED_VIDEO and not self._seek_requested:
                    self._wait_cond.wait(self._mutex, 200)

                # Handle seek request
                if self._seek_requested:
                    target_frame = int(self._seek_target_pct * total_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    self._seek_requested = False
                    self.person_tracker.reset()
                self._mutex.unlock()

                if not self._is_running or self.source_mode != SourceMode.UPLOADED_VIDEO:
                    break

                t_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Video completed: loop back
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.event_manager.reset_state()
                    self.person_tracker.reset()
                    time.sleep(0.1)
                    continue

                curr_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                curr_sec = curr_frame_idx / video_fps
                progress_pct = curr_frame_idx / max(1, total_frames)

                h, w, _ = frame.shape
                res_str = f"{w} × {h}"
                measured_fps = self._measure_fps(time.time())
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                people = self._process_frame(frame, rgb_frame, curr_sec, int(curr_sec * 1000), True, float(measured_fps))
                self.frame_ready.emit(rgb_frame, people, float(measured_fps), res_str)
                self.video_position_changed.emit(curr_sec, duration_sec, progress_pct)

                # Frame pacing to match video FPS
                elapsed = time.time() - t_start
                time.sleep(max(0.001, frame_interval - elapsed))
        finally:
            cap.release()
