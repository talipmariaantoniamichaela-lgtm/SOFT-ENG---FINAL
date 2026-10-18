"""
Subject / Person Identity Recognition Module for Motion Reader AI.
Trains and infers identity on facial & body features of enrolled individuals.
Uses a lightweight PyTorch CNN architecture for fast CPU/GPU inference.
"""
import os
import json
import time
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class SubjectPrediction:
    subject_name: str
    confidence: float
    is_known: bool
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    raw_probabilities: List[float] = field(default_factory=list)


class SubjectClassifierNet(nn.Module):
    """
    Lightweight, high-accuracy Convolutional Neural Network for Subject Recognition.
    Optimized for low-latency inference on CPU while supporting CUDA acceleration.
    """
    def __init__(self, num_classes: int = 5, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 256 x 4 x 4 = 4096
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class SubjectRecognitionEngine:
    """
    Runtime inference engine for Subject Recognition.
    Integrates with the live frame processor to identify subjects in bounding box crops.
    Includes temporal smoothing and confidence debounce to prevent flickers.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        confidence_threshold: float = 0.50,
        device: Optional[str] = None,
        smoothing_window: int = 5,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[SubjectClassifierNet] = None
        self.classes: List[str] = []
        self.input_size: Tuple[int, int] = (128, 128)
        self.is_loaded = False
        self.metrics: Dict = {}
        self.model_path = model_path
        self.meta_path = meta_path

        # Normalization constants (ImageNet standard)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Temporal smoothing buffer
        self.history = deque(maxlen=smoothing_window)
        self.track_histories: Dict[int, deque] = {}
        self.smoothing_window = smoothing_window
        self.prototypes: Optional[Dict[str, torch.Tensor]] = None
        self._face_cascade = None
        self._profile_cascade = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path, meta_path)

    def load_model(self, model_path: str, meta_path: Optional[str] = None) -> bool:
        """Loads trained weights and class mapping."""
        try:
            if not meta_path or not os.path.exists(meta_path):
                meta_path = str(Path(model_path).with_suffix(".json"))
                if not os.path.exists(meta_path):
                    meta_path = str(Path(model_path).parent / f"{Path(model_path).stem}_meta.json")

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.classes = meta.get("classes", [])
                    self.metrics = meta.get("metrics", {})
                    size = meta.get("input_size", [128, 128])
                    self.input_size = (size[0], size[1])
            else:
                self.classes = ["Dainna", "Lawrenz", "Maria", "Mark", "Quinneth"]

            num_classes = len(self.classes)
            self.model = SubjectClassifierNet(num_classes=num_classes)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # Load feature prototypes if available
            proto_path = str(Path(model_path).parent / "subject_prototypes.pt")
            if os.path.exists(proto_path):
                self.prototypes = torch.load(proto_path, map_location=self.device)

            self.model_path = model_path
            self.meta_path = meta_path
            self.is_loaded = True
            self.history.clear()
            self.track_histories.clear()
            print(f"[SubjectRecognitionEngine] Loaded model from {model_path} with {num_classes} classes: {self.classes}")
            return True
        except Exception as e:
            print(f"[SubjectRecognitionEngine] Failed to load model: {e}")
            self.is_loaded = False
            return False

    def _extract_face_region(self, img: np.ndarray, is_bgr: bool = False) -> Optional[np.ndarray]:
        """Extracts primary face region from a full frame or large uncropped image."""
        try:
            if self._face_cascade is None and self.model_path:
                model_dir = Path(self.model_path).parent
                f_path = str(model_dir / "haarcascade_frontalface_default.xml")
                if os.path.exists(f_path):
                    self._face_cascade = cv2.CascadeClassifier(f_path)
            if self._face_cascade is None:
                return None

            h, w = img.shape[:2]
            scale = 800.0 / w if w > 800 else 1.0
            small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img

            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY)
            else:
                gray = small

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enh = clahe.apply(gray)

            faces = self._face_cascade.detectMultiScale(enh, 1.10, 3, minSize=(40, 40))
            if len(faces) == 0 and self.model_path:
                p_path = str(Path(self.model_path).parent / "haarcascade_profileface.xml")
                if os.path.exists(p_path):
                    if self._profile_cascade is None:
                        self._profile_cascade = cv2.CascadeClassifier(p_path)
                    faces = self._profile_cascade.detectMultiScale(enh, 1.10, 3, minSize=(40, 40))
                    if len(faces) == 0:
                        f_faces = self._profile_cascade.detectMultiScale(cv2.flip(enh, 1), 1.10, 3, minSize=(40, 40))
                        if len(f_faces) > 0:
                            fx, fy, fw, fh = f_faces[0]
                            faces = [(enh.shape[1] - (fx + fw), fy, fw, fh)]

            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                sx = 1.0 / scale
                sy = 1.0 / scale
                orig_x = max(0, int(fx * sx))
                orig_y = max(0, int(fy * sy))
                orig_w = min(w - orig_x, int(fw * sx))
                orig_h = min(h - orig_y, int(fh * sy))
                pad = int(min(orig_w, orig_h) * 0.12)
                x1 = max(0, orig_x - pad)
                y1 = max(0, orig_y - pad)
                x2 = min(w, orig_x + orig_w + pad)
                y2 = min(h, orig_y + orig_h + pad)
                return img[y1:y2, x1:x2]
        except Exception:
            pass
        return None

    def preprocess_crop(self, crop: np.ndarray, is_bgr: bool = False) -> Optional[torch.Tensor]:
        """Prepares an RGB or BGR crop into a normalized PyTorch tensor."""
        if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        # Automatically extract face if a full frame / large capture was passed
        if crop.shape[0] > 240 or crop.shape[1] > 240:
            face_region = self._extract_face_region(crop, is_bgr=is_bgr)
            if face_region is not None and face_region.size > 0:
                crop = face_region

        # Convert to RGB if needed
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        elif is_bgr:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor

    @torch.no_grad()
    def predict_raw(self, crop: np.ndarray, is_bgr: bool = False) -> SubjectPrediction:
        """Runs raw inference on a cropped region without temporal smoothing."""
        if not self.is_loaded or self.model is None or crop is None:
            return SubjectPrediction(
                subject_name="Unknown",
                confidence=0.0,
                is_known=False
            )

        tensor = self.preprocess_crop(crop, is_bgr=is_bgr)
        if tensor is None:
            return SubjectPrediction(
                subject_name="Unknown",
                confidence=0.0,
                is_known=False
            )

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        prob_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}

        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        # Subject identification decision
        pred_name = self.classes[best_idx]
        
        # Cosine prototype check
        best_sim = 0.0
        best_sim_name = pred_name
        if self.prototypes:
            feat = self.model.features(tensor).flatten(1)
            feat = feat / torch.norm(feat, dim=1, keepdim=True)
            sims = {name: float(torch.cosine_similarity(feat, proto.view(1, -1).to(self.device)).detach()) for name, proto in self.prototypes.items()}
            best_sim_name = max(sims, key=sims.get)
            best_sim = sims[best_sim_name]

        # Recognized if confidence exceeds threshold with distinguishable margin, or prototype agreement
        is_known = (best_conf >= self.confidence_threshold and margin >= 0.06) or (best_sim >= 0.40 and best_sim_name == pred_name)
        predicted_name = pred_name if is_known else "Unknown"
        final_conf = best_conf if is_known else round(max(best_conf, best_sim), 2)

        return SubjectPrediction(
            subject_name=predicted_name,
            confidence=round(final_conf, 2),
            is_known=is_known,
            class_probabilities=prob_dict,
            raw_probabilities=probs.tolist()
        )

    @torch.no_grad()
    def predict(
        self,
        crop: np.ndarray,
        is_bgr: bool = False,
        track_id: Optional[int] = None
    ) -> SubjectPrediction:
        """Runs inference on a cropped region with per-track or global temporal smoothing."""
        raw = self.predict_raw(crop, is_bgr=is_bgr)
        if not raw.is_known and raw.confidence == 0.0:
            return raw

        # Select history buffer
        if track_id is not None:
            if track_id not in self.track_histories:
                self.track_histories[track_id] = deque(maxlen=self.smoothing_window)
            buf = self.track_histories[track_id]
        else:
            buf = self.history

        # Best class name from raw probs
        if raw.class_probabilities:
            best_name = max(raw.class_probabilities, key=raw.class_probabilities.get)
            best_conf = raw.class_probabilities[best_name]
        else:
            best_name = raw.subject_name
            best_conf = raw.confidence

        buf.append((best_name, best_conf))
        recent_names = [name for name, _ in buf]
        most_common = max(set(recent_names), key=recent_names.count)
        avg_conf = float(np.mean([conf for name, conf in buf if name == most_common]))

        is_known = avg_conf >= self.confidence_threshold
        final_name = most_common if is_known else "Unknown"

        return SubjectPrediction(
            subject_name=final_name,
            confidence=round(avg_conf, 2),
            is_known=is_known,
            class_probabilities=raw.class_probabilities,
            raw_probabilities=raw.raw_probabilities
        )
