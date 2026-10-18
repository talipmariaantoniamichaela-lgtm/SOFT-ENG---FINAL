"""
Background Video Frame Processor Worker Thread for Motion Reader.
Performs frame acquisition, MediaPipe inference, temporal analysis,
and measured FPS tracking off the GUI thread.
"""
import os
import time
import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

from app.detection.pose_estimator import PoseEstimator, TrackingResult
from app.detection.motion_analyzer import MotionAnalyzer, FrameAnalysisResult
from app.detection.subject_classifier import SubjectRecognitionEngine, SubjectPrediction
from app.services.event_manager import EventManager
from app.config import AppConfig


class SourceMode(Enum):
    LIVE_CAMERA = "LIVE_CAMERA"
    UPLOADED_VIDEO = "UPLOADED_VIDEO"


class FrameProcessorThread(QThread):
    """
    Worker thread that captures frames, runs AI pose estimation & motion analysis,
    and emits processed frames and metrics back to the UI.
    """
    frame_ready = Signal(object, object, object, float, str)  # (rgb_frame, tracking_result, frame_analysis, fps, res_str)
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
            smoothing_factor=config.detection.smoothing_factor
        )
        self.motion_analyzer = MotionAnalyzer(
            sensitivity=config.detection.action_sensitivity
        )

        # Subject Identity Classifier
        self.subject_engine = SubjectRecognitionEngine()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_model = os.path.join(base_dir, "assets", "models", "subject_classifier.pth")
        default_meta = os.path.join(base_dir, "assets", "models", "subject_classifier_meta.json")
        if os.path.exists(default_model):
            self.subject_engine.load_model(default_model, default_meta)

        # FPS measurement
        self.fps_timestamps = []

    def reload_subject_model(self, model_path: str, meta_path: Optional[str] = None) -> bool:
        """Hot-reloads custom trained or selected subject recognition model weights."""
        self._mutex.lock()
        success = self.subject_engine.load_model(model_path, meta_path)
        self._mutex.unlock()
        if success:
            classes_str = ", ".join(self.subject_engine.classes)
            self.event_manager.log_system_info(
                f"Subject recognition model loaded ({len(self.subject_engine.classes)} classes: {classes_str})"
            )
        else:
            self.event_manager.log_error(f"Failed to reload subject model: {model_path}")
        return success

    def set_source_camera(self, camera_index: int) -> None:
        """Switches to webcam mode."""
        self._mutex.lock()
        self.source_mode = SourceMode.LIVE_CAMERA
        self.camera_index = camera_index
        self.event_manager.reset_state()
        self.motion_analyzer.reset()
        self.pose_estimator.reset()
        self._is_paused = False
        self._mutex.unlock()
        self.source_changed.emit(SourceMode.LIVE_CAMERA, f"Camera {camera_index}")

    def set_source_video_file(self, filepath: str) -> None:
        """Switches to uploaded video file mode."""
        self._mutex.lock()
        self.source_mode = SourceMode.UPLOADED_VIDEO
        self.video_filepath = filepath
        self.event_manager.reset_state()
        self.motion_analyzer.reset()
        self.pose_estimator.reset()
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
        self.motion_analyzer.reset()

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

    def _recognize_subject(
        self,
        rgb_frame: np.ndarray,
        tracking_result: TrackingResult,
        analysis: FrameAnalysisResult
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Crops the bounding box of the detected person and performs recognition
        using the active custom-trained model.
        """
        if not self.subject_engine.is_loaded or not tracking_result.person_detected or not tracking_result.bounding_box:
            return None, None

        h, w, _ = rgb_frame.shape
        best_pred = None

        # Test 1: Face / Head crop if pose landmarks are available
        if tracking_result.pose_landmarks and len(tracking_result.pose_landmarks) > 10:
            lms = tracking_result.pose_landmarks
            head_pts = [lms[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if lms[i].visibility > 0.30]
            if head_pts:
                min_hx = min(p.x for p in head_pts)
                max_hx = max(p.x for p in head_pts)
                min_hy = min(p.y for p in head_pts)
                max_hy = max(p.y for p in head_pts)
                pad_w = max(0.04, (max_hx - min_hx) * 0.35)
                pad_h = max(0.04, (max_hy - min_hy) * 0.35)
                hx1 = max(0, int((min_hx - pad_w) * w))
                hy1 = max(0, int((min_hy - pad_h) * h))
                hx2 = min(w, int((max_hx + pad_w) * w))
                hy2 = min(h, int((max_hy + pad_h) * h))
                if (hx2 - hx1) > 20 and (hy2 - hy1) > 20:
                    face_crop = rgb_frame[hy1:hy2, hx1:hx2]
                    pred_face = self.subject_engine.predict_raw(face_crop, is_bgr=False)
                    if pred_face.is_known:
                        best_pred = pred_face

        # Test 2: Full / Upper person bounding box crop
        min_x, min_y, max_x, max_y = tracking_result.bounding_box
        x1 = max(0, int((min_x - (max_x - min_x) * 0.10) * w))
        y1 = max(0, int((min_y - (max_y - min_y) * 0.10) * h))
        x2 = min(w, int((max_x + (max_x - min_x) * 0.10) * w))
        y2 = min(h, int((max_y + (max_y - min_y) * 0.10) * h))

        if (x2 - x1) > 20 and (y2 - y1) > 20:
            body_crop = rgb_frame[y1:y2, x1:x2]
            pred_body = self.subject_engine.predict_raw(body_crop, is_bgr=False)
            if best_pred is None or (pred_body.is_known and pred_body.confidence > best_pred.confidence):
                best_pred = pred_body

        if best_pred and best_pred.is_known:
            analysis.subject_name = best_pred.subject_name
            analysis.subject_confidence = best_pred.confidence
            return best_pred.subject_name, best_pred.confidence

        return None, None

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
        self.event_manager.log_system_info("Face recognition initialized")
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

                # Calculate measured FPS
                now = time.time()
                self.fps_timestamps.append(now)
                # Keep timestamps within the last 1.0 second
                while self.fps_timestamps and now - self.fps_timestamps[0] > 1.0:
                    self.fps_timestamps.pop(0)
                measured_fps = len(self.fps_timestamps)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Pose Estimation
                t_ms = int(now * 1000)
                tracking_result = self.pose_estimator.estimate(rgb_frame, timestamp_ms=t_ms)

                # Motion Analysis
                analysis = self.motion_analyzer.analyze(tracking_result, now)
                analysis.fps = float(measured_fps)

                # Subject Recognition (Custom Trained Model)
                subject_name, subject_conf = self._recognize_subject(rgb_frame, tracking_result, analysis)

                # Process events
                self.event_manager.process_frame_analysis(
                    person_detected=analysis.person_detected,
                    person_id=analysis.person_id,
                    posture=analysis.posture,
                    left_hand_raised=analysis.left_hand_raised,
                    right_hand_raised=analysis.right_hand_raised,
                    gestures=analysis.gestures,
                    body_movements=analysis.body_movements,
                    current_time=now,
                    is_video_relative=False,
                    subject_name=subject_name,
                    subject_confidence=subject_conf
                )

                # Update tracking status based on person detection
                if not analysis.person_detected:
                    self.status_changed.emit("No Person Detected", "warning")
                else:
                    self.status_changed.emit("Tracking...", "active")

                # Emit to UI
                self.frame_ready.emit(rgb_frame, tracking_result, analysis, float(measured_fps), res_str)

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
        self.event_manager.log_system_info("Camera connected successfully")
        self.event_manager.log_system_info("Face recognition initialized")
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
                self._mutex.unlock()

                if not self._is_running or self.source_mode != SourceMode.UPLOADED_VIDEO:
                    break

                t_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Video completed: loop back or pause
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.event_manager.reset_state()
                    self.motion_analyzer.reset()
                    self.subject_engine.history.clear()
                    time.sleep(0.1)
                    continue

                curr_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                curr_sec = curr_frame_idx / video_fps
                progress_pct = curr_frame_idx / max(1, total_frames)

                h, w, _ = frame.shape
                res_str = f"{w} × {h}"

                # Calculate measured FPS
                now = time.time()
                self.fps_timestamps.append(now)
                while self.fps_timestamps and now - self.fps_timestamps[0] > 1.0:
                    self.fps_timestamps.pop(0)
                measured_fps = len(self.fps_timestamps)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Pose Estimation
                t_ms = int(curr_sec * 1000)
                tracking_result = self.pose_estimator.estimate(rgb_frame, timestamp_ms=t_ms)

                # Motion Analysis
                analysis = self.motion_analyzer.analyze(tracking_result, curr_sec)
                analysis.fps = float(measured_fps)

                # Subject Recognition (Custom Trained Model)
                subject_name, subject_conf = self._recognize_subject(rgb_frame, tracking_result, analysis)

                # Process events with video-relative timestamps
                self.event_manager.process_frame_analysis(
                    person_detected=analysis.person_detected,
                    person_id=analysis.person_id,
                    posture=analysis.posture,
                    left_hand_raised=analysis.left_hand_raised,
                    right_hand_raised=analysis.right_hand_raised,
                    gestures=analysis.gestures,
                    body_movements=analysis.body_movements,
                    current_time=curr_sec,
                    is_video_relative=True,
                    subject_name=subject_name,
                    subject_confidence=subject_conf
                )

                # Status
                if not analysis.person_detected:
                    self.status_changed.emit("No Person Detected", "warning")
                else:
                    self.status_changed.emit("Tracking...", "active")

                # Emit to UI
                self.frame_ready.emit(rgb_frame, tracking_result, analysis, float(measured_fps), res_str)
                self.video_position_changed.emit(curr_sec, duration_sec, progress_pct)

                # Frame pacing to match video FPS
                elapsed = time.time() - t_start
                sleep_time = max(0.001, frame_interval - elapsed)
                time.sleep(sleep_time)
        finally:
            cap.release()
