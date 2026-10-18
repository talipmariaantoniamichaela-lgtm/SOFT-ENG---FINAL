"""
Multi-Person Tracker and Action Reporter for Motion Reader AI.
Simultaneously detects, identifies, tracks, and analyzes movements for multiple individuals
using face recognition (PyTorch subject classifier) and kinematic motion heuristics.
"""
import os
import time
import math
from datetime import datetime
from collections import deque
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from app.detection.subject_classifier import SubjectRecognitionEngine, SubjectPrediction


@dataclass
class TrackedPerson:
    track_id: int
    name: str = "Unknown"
    confidence: float = 0.0
    is_known: bool = False
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)          # (x1, y1, x2, y2)
    face_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)     # (fx, fy, fw, fh)
    center: Tuple[float, float] = (0.0, 0.0)
    
    # Recognition history for stabilization
    name_history: deque = field(default_factory=lambda: deque(maxlen=10))
    conf_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Tracking counters
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    first_detected_emitted: bool = False
    has_entered_emitted: bool = False
    is_active: bool = True
    
    # Kinematic states
    center_history: deque = field(default_factory=lambda: deque(maxlen=20))
    current_posture: Optional[str] = None                   # 'sitting', 'standing', 'crouching'
    posture_buffer: deque = field(default_factory=lambda: deque(maxlen=8))
    head_turn_state: str = "center"                         # 'center', 'left', 'right'
    right_hand_raised: bool = False
    left_hand_raised: bool = False
    is_waving: bool = False
    is_walking: bool = False
    
    # Cooldowns per event type (event_key -> timestamp)
    last_event_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class MovementEventOutput:
    timestamp: str              # [HH:MM:SS]
    type: str                   # 'SYSTEM', 'PERSON', 'POSTURE', 'HEAD', 'HAND', 'GESTURE', 'BODY'
    text: str                   # Natural language description
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    confidence: Optional[float] = None


class MultiPersonTracker:
    """
    Real-time Multi-Person Tracking and Action Recognition Engine.
    Processes video frames to track multiple people simultaneously, recognize
    enrolled individuals from the dataset, and report actions without duplicates.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        confidence_threshold: float = 0.50
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(base_dir, "app", "assets", "models")
        
        # 1. Load cascades
        self.frontal_cascade_path = os.path.join(models_dir, "haarcascade_frontalface_default.xml")
        self.profile_cascade_path = os.path.join(models_dir, "haarcascade_profileface.xml")
        
        self.face_cascade = None
        if os.path.exists(self.frontal_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(self.frontal_cascade_path)
            
        self.profile_cascade = None
        if os.path.exists(self.profile_cascade_path):
            self.profile_cascade = cv2.CascadeClassifier(self.profile_cascade_path)
            
        # 2. Load PyTorch Subject Recognition Engine
        if not model_path:
            model_path = os.path.join(models_dir, "subject_classifier.pth")
        if not meta_path:
            meta_path = os.path.join(models_dir, "subject_classifier_meta.json")
            
        self.subject_engine = SubjectRecognitionEngine(
            model_path=model_path if os.path.exists(model_path) else None,
            meta_path=meta_path if os.path.exists(meta_path) else None,
            confidence_threshold=confidence_threshold
        )
        
        # 3. CLAHE for robust lighting handling
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 4. State
        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_track_id = 1
        self.session_started = False
        self.last_startup_time = 0.0
        
        # Cooldown intervals in seconds
        self.COOLDOWN_HEAD_TURN = 2.4
        self.COOLDOWN_POSTURE = 2.0
        self.COOLDOWN_HAND = 1.8
        self.COOLDOWN_GESTURE = 2.2
        self.COOLDOWN_WALK = 3.0
        
        # Historical registry of seen names to detect re-entry
        self.previously_departed_names: set = set()

    def reset(self):
        """Resets all tracking states for a fresh camera session."""
        self.tracks.clear()
        self.next_track_id = 1
        self.session_started = False
        self.previously_departed_names.clear()

    @staticmethod
    def _format_time(t: Optional[float] = None) -> str:
        cur = t if t is not None else time.time()
        return datetime.fromtimestamp(cur).strftime("%H:%M:%S")

    @staticmethod
    def _get_pronoun(name: str) -> str:
        """Helper to format natural language e.g. 'his right hand' vs 'her right hand'."""
        male_names = {"Lawrenz", "Mark", "Quinneth"}
        female_names = {"Dainna", "Maria"}
        if name in male_names:
            return "his"
        elif name in female_names:
            return "her"
        return "the"

    def get_initialization_events(self, base_time: Optional[float] = None) -> List[MovementEventOutput]:
        """Generates the required 3 startup initialization events."""
        t = base_time or time.time()
        t1_str = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        t2_str = datetime.fromtimestamp(t + 0.1).strftime("%H:%M:%S")
        t3_str = datetime.fromtimestamp(t + 0.2).strftime("%H:%M:%S")
        
        return [
            MovementEventOutput(timestamp=t1_str, type="SYSTEM", text="Camera connected successfully"),
            MovementEventOutput(timestamp=t2_str, type="SYSTEM", text="Face recognition initialized"),
            MovementEventOutput(timestamp=t3_str, type="SYSTEM", text="Movement detection started"),
        ]

    def _detect_faces_and_profiles(self, gray: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        """
        Fast multiscale face detection.
        Downscales search space to 320px for 10x throughput, and conditionally skips
        profile cascade when frontal face is clearly resolved.
        """
        results = []
        if self.face_cascade is None or self.face_cascade.empty():
            return results

        gh, gw = gray.shape[:2]
        det_w = 320
        det_h = max(24, int(gh * (det_w / float(gw))))
        small = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        enhanced = self.clahe.apply(small)
        scale_x = gw / float(det_w)
        scale_y = gh / float(det_h)

        # Frontal faces (scale factor 1.10, minNeighbors 3, minSize 24 for sitting webcam distance)
        front_faces = self.face_cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.10,
            minNeighbors=3,
            minSize=(24, 24)
        )
        for (x, y, w, h) in front_faces:
            results.append((int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y), "front"))

        # Also check profile cascades to detect individuals in profile or turned heads
        if self.profile_cascade is not None and not self.profile_cascade.empty():
            prof_left = self.profile_cascade.detectMultiScale(
                enhanced,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=(22, 22)
            )
            for (x, y, w, h) in prof_left:
                results.append((int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y), "profile_left"))

            flipped = cv2.flip(enhanced, 1)
            prof_right = self.profile_cascade.detectMultiScale(
                flipped,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=(22, 22)
            )
            for (x, y, w, h) in prof_right:
                orig_x = det_w - (x + w)
                results.append((int(orig_x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y), "profile_right"))

        # Non-maximum suppression / deduplication
        deduped = []
        for det in sorted(results, key=lambda d: d[2] * d[3], reverse=True):
            x, y, w, h, o = det
            overlap = False
            for (dx, dy, dw, dh, _) in deduped:
                ix1, iy1 = max(x, dx), max(y, dy)
                ix2, iy2 = min(x + w, dx + dw), min(y + h, dy + dh)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    union_area = (w * h) + (dw * dh) - inter_area
                    if inter_area / union_area > 0.35:
                        overlap = True
                        break
            if not overlap:
                deduped.append(det)

        return deduped

    def _infer_identity(self, frame_bgr: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> Tuple[str, float, bool]:
        """
        Runs PyTorch recognition model on upper-body / face crop.
        Returns (name, confidence, is_known).
        """
        if not self.subject_engine.is_loaded:
            return "Unknown", 0.0, False

        h, w = frame_bgr.shape[:2]
        face_crop = frame_bgr[max(0, fy):min(h, fy + fh), max(0, fx):min(w, fx + fw)]
        if face_crop is None or face_crop.size == 0:
            return "Unknown", 0.0, False

        # Reject pure synthetic noise / corrupted regions via high-frequency Laplacian variance
        gray_chk = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        if cv2.Laplacian(gray_chk, cv2.CV_64F).var() > 4000:
            return "Unknown", 0.0, False
        
        # Face crop evaluation (primary biometric identity)
        pred_face = self.subject_engine.predict_raw(face_crop, is_bgr=True)
        if pred_face.is_known:
            return pred_face.subject_name, pred_face.confidence, True

        # Expanded head crop with facial contour margin
        pad = int(min(fw, fh) * 0.20)
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(w, fx + fw + pad)
        y2 = min(h, fy + fh + int(pad * 1.5))
        expanded_crop = frame_bgr[y1:y2, x1:x2]
        pred_expanded = self.subject_engine.predict_raw(expanded_crop, is_bgr=True)
        if pred_expanded.is_known:
            return pred_expanded.subject_name, pred_expanded.confidence, True

        # Majority or higher confidence between face and head crop
        conf = max(pred_face.confidence, pred_expanded.confidence)
        return "Unknown", conf, False

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        current_time: Optional[float] = None,
        client_pose_landmarks: Optional[List[Dict[str, float]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[MovementEventOutput]]:
        """
        Main processing method per frame.
        Takes a BGR video frame, returns:
          - persons: list of person info dicts for visual bounding box overlay
          - events: list of new natural-language events to append to Motion Terminal
        """
        cur_t = current_time or time.time()
        time_str = self._format_time(cur_t)
        new_events: List[MovementEventOutput] = []

        # 1. Startup initialization banner on first frame
        if not self.session_started:
            self.session_started = True
            new_events.extend(self.get_initialization_events(cur_t))

        h, w = frame_bgr.shape[:2]
        
        # Resize to standard width for consistent detection performance
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            det_w = 640
            det_h = int(h * scale)
            det_frame = cv2.resize(frame_bgr, (det_w, det_h))
        else:
            det_frame = frame_bgr

        gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
        detections = self._detect_faces_and_profiles(gray)

        # Scale detections back to original frame coordinates
        scaled_detections = []
        for (dx, dy, dw, dh, orientation) in detections:
            fx = int(dx / scale)
            fy = int(dy / scale)
            fw = int(dw / scale)
            fh = int(dh / scale)
            # Associated upper body bbox
            px1 = max(0, fx - int(fw * 0.45))
            py1 = max(0, fy - int(fh * 0.25))
            px2 = min(w, fx + fw + int(fw * 0.45))
            py2 = min(h, fy + int(fh * 3.2))
            cx = (px1 + px2) / 2.0
            cy = (py1 + py2) / 2.0
            scaled_detections.append({
                "face_bbox": (fx, fy, fw, fh),
                "bbox": (px1, py1, px2, py2),
                "center": (cx, cy),
                "orientation": orientation
            })

        # 2. Match detections with existing tracks (Centroid Tracking)
        matched_track_ids = set()
        matched_det_indices = set()

        for track_id, track in list(self.tracks.items()):
            if not track.is_active:
                continue

            best_dist = float("inf")
            best_idx = -1
            tcx, tcy = track.center

            for i, det in enumerate(scaled_detections):
                if i in matched_det_indices:
                    continue
                dcx, dcy = det["center"]
                dist = math.hypot(dcx - tcx, dcy - tcy)
                # Maximum allowed displacement between frames (adaptive to person size)
                max_allowed = max(180, (track.bbox[2] - track.bbox[0]) * 1.2)
                if dist < max_allowed and dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                det = scaled_detections[best_idx]
                matched_det_indices.add(best_idx)
                matched_track_ids.add(track_id)

                # Update track geometry
                track.bbox = det["bbox"]
                track.face_bbox = det["face_bbox"]
                track.center = det["center"]
                track.center_history.append((det["center"][0], det["center"][1], cur_t))
                track.consecutive_hits += 1
                track.consecutive_misses = 0

                # Periodic identity re-verification
                fx, fy, fw, fh = det["face_bbox"]
                name, conf, is_known = self._infer_identity(frame_bgr, fx, fy, fw, fh)
                track.name_history.append(name)
                track.conf_history.append(conf)

                # Stabilized consensus name via majority vote
                counts = {}
                for n in track.name_history:
                    counts[n] = counts.get(n, 0) + 1
                consensus_name = max(counts, key=counts.get)
                track.name = consensus_name
                track.confidence = round(float(np.mean(track.conf_history)), 2)
                track.is_known = consensus_name != "Unknown"

                # Check orientation
                det_orient = det["orientation"]
                if det_orient == "profile_left":
                    self._check_head_turn(track, "left", cur_t, time_str, new_events)
                elif det_orient == "profile_right":
                    self._check_head_turn(track, "right", cur_t, time_str, new_events)
                else:
                    track.head_turn_state = "center"

                # Analyze posture & body movements
                self._analyze_track_movements(track, h, w, cur_t, time_str, new_events)

        # 3. Create new tracks for unmatched detections
        for i, det in enumerate(scaled_detections):
            if i in matched_det_indices:
                continue

            fx, fy, fw, fh = det["face_bbox"]
            name, conf, is_known = self._infer_identity(frame_bgr, fx, fy, fw, fh)

            new_track = TrackedPerson(
                track_id=self.next_track_id,
                name=name,
                confidence=round(conf, 2),
                is_known=is_known,
                bbox=det["bbox"],
                face_bbox=det["face_bbox"],
                center=det["center"]
            )
            self.next_track_id += 1
            new_track.name_history.append(name)
            new_track.conf_history.append(conf)
            new_track.center_history.append((det["center"][0], det["center"][1], cur_t))

            # Initial detection event emission
            disp_name = new_track.name
            if disp_name in self.previously_departed_names:
                new_events.append(MovementEventOutput(
                    timestamp=time_str,
                    type="PERSON",
                    text=f"{disp_name} enters the frame",
                    person_id=new_track.track_id,
                    person_name=disp_name,
                    confidence=new_track.confidence
                ))
                self.previously_departed_names.discard(disp_name)
            else:
                new_events.append(MovementEventOutput(
                    timestamp=time_str,
                    type="PERSON",
                    text=f"{disp_name} detected",
                    person_id=new_track.track_id,
                    person_name=disp_name,
                    confidence=new_track.confidence
                ))

            new_track.first_detected_emitted = True

            # Initial posture check
            self._analyze_track_movements(new_track, h, w, cur_t, time_str, new_events)

            self.tracks[new_track.track_id] = new_track

        # 4. Handle tracks that were missed this frame
        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_track_ids and track.is_active:
                track.consecutive_misses += 1
                # If missed for 12 consecutive checks (~1.5 to 2.0 seconds) -> person left
                if track.consecutive_misses >= 12:
                    track.is_active = False
                    disp_name = track.name
                    if disp_name != "Unknown":
                        self.previously_departed_names.add(disp_name)
                    new_events.append(MovementEventOutput(
                        timestamp=time_str,
                        type="PERSON",
                        text=f"{disp_name} walks away",
                        person_id=track.track_id,
                        person_name=disp_name
                    ))

        # 5. Clean up inactive tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.is_active}

        # 6. Format active persons for visual bounding box overlay
        active_persons = []
        for track in self.tracks.values():
            px1, py1, px2, py2 = track.bbox
            active_persons.append({
                "id": track.track_id,
                "name": track.name,
                "confidence": track.confidence,
                "is_known": track.is_known,
                "bbox": list(track.bbox),
                "bbox_norm": [
                    round(max(0.0, min(1.0, px1 / max(1, w))), 4),
                    round(max(0.0, min(1.0, py1 / max(1, h))), 4),
                    round(max(0.0, min(1.0, px2 / max(1, w))), 4),
                    round(max(0.0, min(1.0, py2 / max(1, h))), 4),
                ],
                "face_bbox": list(track.face_bbox),
                "posture": track.current_posture or "standing",
                "action": f"is {track.current_posture}" if track.current_posture else "present"
            })

        return active_persons, new_events

    def _check_head_turn(
        self,
        track: TrackedPerson,
        direction: str,
        cur_t: float,
        time_str: str,
        events: List[MovementEventOutput]
    ):
        """Emits head turn event with debouncing."""
        if track.head_turn_state != direction:
            track.head_turn_state = direction
            last_t = track.last_event_times.get("head_turn", 0.0)
            if cur_t - last_t >= self.COOLDOWN_HEAD_TURN:
                track.last_event_times["head_turn"] = cur_t
                events.append(MovementEventOutput(
                    timestamp=time_str,
                    type="HEAD",
                    text=f"{track.name} turns head to the {direction}",
                    person_id=track.track_id,
                    person_name=track.name,
                    confidence=0.89
                ))

    def _analyze_track_movements(
        self,
        track: TrackedPerson,
        frame_h: int,
        frame_w: int,
        cur_t: float,
        time_str: str,
        events: List[MovementEventOutput]
    ):
        """
        Calculates posture transitions (sitting vs standing) and walking movement
        from tracked bounding box trajectory.
        """
        fy = track.face_bbox[1]
        fh = track.face_bbox[3]
        
        # 1. Posture calculation based on face vertical positioning and aspect
        # Sitting: head is relatively low in the frame (center > 0.38 of frame height)
        face_center_y_norm = (fy + fh * 0.5) / frame_h
        raw_posture = "sitting" if face_center_y_norm > 0.36 else "standing"
        
        track.posture_buffer.append(raw_posture)
        if len(track.posture_buffer) >= 3:
            counts = {}
            for p in track.posture_buffer:
                counts[p] = counts.get(p, 0) + 1
            consensus_posture = max(counts, key=counts.get)
            
            if consensus_posture != track.current_posture:
                prev = track.current_posture
                track.current_posture = consensus_posture
                last_t = track.last_event_times.get("posture", 0.0)
                
                if cur_t - last_t >= self.COOLDOWN_POSTURE:
                    track.last_event_times["posture"] = cur_t
                    if prev == "sitting" and consensus_posture == "standing":
                        events.append(MovementEventOutput(
                            timestamp=time_str,
                            type="POSTURE",
                            text=f"{track.name} stands up",
                            person_id=track.track_id,
                            person_name=track.name
                        ))
                    else:
                        events.append(MovementEventOutput(
                            timestamp=time_str,
                            type="POSTURE",
                            text=f"{track.name} is {consensus_posture}",
                            person_id=track.track_id,
                            person_name=track.name
                        ))

        # 2. Walking / Locomotion detection
        if len(track.center_history) >= 10:
            p_old = track.center_history[0]
            p_new = track.center_history[-1]
            dx = p_new[0] - p_old[0]
            dt = p_new[2] - p_old[2]
            
            if dt > 0.5:
                speed_x = abs(dx) / (frame_w * dt)
                # Significant sustained horizontal translation
                if speed_x > 0.14:
                    last_t = track.last_event_times.get("walking", 0.0)
                    if cur_t - last_t >= self.COOLDOWN_WALK:
                        track.last_event_times["walking"] = cur_t
                        events.append(MovementEventOutput(
                            timestamp=time_str,
                            type="BODY",
                            text=f"{track.name} is walking",
                            person_id=track.track_id,
                            person_name=track.name,
                            confidence=0.88
                        ))

    def trigger_hand_gesture(
        self,
        person_id: int,
        gesture_type: str,  # 'right_hand_raised', 'left_hand_raised', 'waving'
        confidence: float = 0.92,
        current_time: Optional[float] = None
    ) -> Optional[MovementEventOutput]:
        """Allows feeding arm/hand detections from MediaPipe or hand tracker."""
        track = self.tracks.get(person_id)
        if not track:
            # Fallback to first track if single person
            if len(self.tracks) == 1:
                track = next(iter(self.tracks.values()))
            else:
                return None

        cur_t = current_time or time.time()
        time_str = self._format_time(cur_t)
        pronoun = self._get_pronoun(track.name)

        if gesture_type == "right_hand_raised":
            last_t = track.last_event_times.get("hand", 0.0)
            if cur_t - last_t >= self.COOLDOWN_HAND:
                track.last_event_times["hand"] = cur_t
                track.right_hand_raised = True
                return MovementEventOutput(
                    timestamp=time_str,
                    type="HAND",
                    text=f"{track.name} raises {pronoun} right hand",
                    person_id=track.track_id,
                    person_name=track.name,
                    confidence=confidence
                )
        elif gesture_type == "left_hand_raised":
            last_t = track.last_event_times.get("hand", 0.0)
            if cur_t - last_t >= self.COOLDOWN_HAND:
                track.last_event_times["hand"] = cur_t
                track.left_hand_raised = True
                return MovementEventOutput(
                    timestamp=time_str,
                    type="HAND",
                    text=f"{track.name} raises {pronoun} left hand",
                    person_id=track.track_id,
                    person_name=track.name,
                    confidence=confidence
                )
        elif gesture_type == "waving":
            last_t = track.last_event_times.get("waving", 0.0)
            if cur_t - last_t >= self.COOLDOWN_GESTURE:
                track.last_event_times["waving"] = cur_t
                return MovementEventOutput(
                    timestamp=time_str,
                    type="GESTURE",
                    text=f"{track.name} is waving",
                    person_id=track.track_id,
                    person_name=track.name,
                    confidence=confidence
                )
        return None
