"""
Training Service and Dataset Management for Motion Reader AI.
Handles automatic dataset discovery, validation, preprocessing,
and background model training using PyTorch without freezing the UI.
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from PySide6.QtCore import QThread, Signal

from app.detection.subject_classifier import SubjectClassifierNet


@dataclass
class DatasetReport:
    is_valid: bool
    dataset_path: str
    total_files: int
    classes: List[str]
    class_counts: Dict[str, int]
    invalid_files: List[str] = field(default_factory=list)
    image_formats: List[str] = field(default_factory=list)
    average_resolution: Tuple[int, int] = (0, 0)
    message: str = ""


class DatasetValidator:
    """Discovers and validates image datasets for Subject Recognition."""

    @staticmethod
    def find_dataset_root(search_dir: Optional[str] = None) -> Optional[Path]:
        """Finds the dataset directory within app/assets/dataset."""
        if search_dir and os.path.exists(search_dir):
            p = Path(search_dir)
            if any(d.is_dir() for d in p.iterdir()):
                return p

        # Standard project location
        base = Path(__file__).resolve().parent.parent / "assets" / "dataset"
        if not base.exists():
            return None

        # Look for subfolder Datasets
        for candidate in base.rglob("Datasets"):
            if candidate.is_dir() and any(d.is_dir() for d in candidate.iterdir()):
                return candidate

        for sub in base.iterdir():
            if sub.is_dir() and any(d.is_dir() for d in sub.iterdir()):
                return sub

        return None

    find_dataset_dir = find_dataset_root

    @classmethod
    def validate_dataset(cls, dataset_path: Optional[str] = None) -> DatasetReport:
        root = cls.find_dataset_root(dataset_path)
        if not root or not root.exists():
            return DatasetReport(
                is_valid=False,
                dataset_path="",
                total_files=0,
                classes=[],
                class_counts={},
                message="Dataset directory not found."
            )

        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        if not classes:
            return DatasetReport(
                is_valid=False,
                dataset_path=str(root),
                total_files=0,
                classes=[],
                class_counts={},
                message="No class subdirectories found in dataset."
            )

        class_counts = {}
        invalid_files = []
        formats = set()
        resolutions = []
        total = 0

        for cls_name in classes:
            cls_dir = root / cls_name
            files = list(cls_dir.glob("*"))
            valid_count = 0
            for f in files:
                if f.is_file():
                    ext = f.suffix.lower()
                    formats.add(ext)
                    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                        valid_count += 1
                        total += 1
                        # Sample resolution on first 3 files per class
                        if valid_count <= 3:
                            try:
                                img = cv2.imread(str(f))
                                if img is not None:
                                    resolutions.append((img.shape[1], img.shape[0]))
                                else:
                                    invalid_files.append(str(f))
                            except Exception:
                                invalid_files.append(str(f))
                    else:
                        invalid_files.append(str(f))
            class_counts[cls_name] = valid_count

        avg_res = (
            int(np.mean([r[0] for r in resolutions])) if resolutions else 0,
            int(np.mean([r[1] for r in resolutions])) if resolutions else 0,
        )

        is_valid = total >= 10 and len(classes) >= 2
        msg = f"Validated {total} samples across {len(classes)} classes: {', '.join(classes)}"

        return DatasetReport(
            is_valid=is_valid,
            dataset_path=str(root),
            total_files=total,
            classes=classes,
            class_counts=class_counts,
            invalid_files=invalid_files,
            image_formats=sorted(list(formats)),
            average_resolution=avg_res,
            message=msg
        )


class InMemoryTensorDataset(Dataset):
    """Ultra-fast in-memory dataset storing pre-resized tensors."""
    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor):
        self.tensors = tensors
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tensors[idx], self.labels[idx]


class TrainingWorkerThread(QThread):
    """
    Background worker thread executing PyTorch training without blocking PySide6 UI.
    Uses pre-cached 128x128 tensors for ultra-fast multi-epoch training on CPU.
    """
    progress_updated = Signal(int, int, float, float, float)  # epoch, total, train_loss, val_loss, val_acc
    log_message = Signal(str, str)                            # message, level ('info', 'success', 'warning', 'error')
    training_completed = Signal(dict)                         # metrics dict
    training_failed = Signal(str)                             # error message

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        target_size: Tuple[int, int] = (128, 128),
        output_dir: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_size = target_size
        self.output_dir = output_dir or str(
            Path(__file__).resolve().parent.parent / "assets" / "models"
        )
        self._stop_requested = False

    def stop(self) -> None:
        """Requests graceful training cancellation."""
        self._stop_requested = True

    def run(self) -> None:
        try:
            self.log_message.emit("Scanning and validating dataset...", "info")
            report = DatasetValidator.validate_dataset(self.dataset_path)
            if not report.is_valid:
                self.training_failed.emit(f"Invalid dataset: {report.message}")
                return

            self.log_message.emit(
                f"Found {report.total_files} samples in {len(report.classes)} classes: {', '.join(report.classes)}",
                "info"
            )

            # Build file lists and labels
            classes = report.classes
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            all_paths = []
            all_labels = []

            root = Path(report.dataset_path)
            for cls_name in classes:
                for f in sorted((root / cls_name).glob("*")):
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        all_paths.append(str(f))
                        all_labels.append(class_to_idx[cls_name])

            # Check cache
            cache_file = root.parent / f".cache_{self.target_size[0]}x{self.target_size[1]}.pt"
            cached_data = None
            if cache_file.exists():
                try:
                    loaded = torch.load(cache_file)
                    if len(loaded.get("labels", [])) == len(all_labels):
                        cached_data = loaded
                        self.log_message.emit("Loaded preprocessed 128x128 image cache.", "info")
                except Exception:
                    cached_data = None

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            if cached_data is not None:
                all_tensors = cached_data["tensors"]
                all_labels_tensor = cached_data["labels"]
            else:
                self.log_message.emit(f"Preprocessing {len(all_paths)} images to {self.target_size[0]}x{self.target_size[1]}...", "info")
                tensor_list = []
                for i, path in enumerate(all_paths):
                    if self._stop_requested:
                        self.training_failed.emit("Cancelled during preprocessing.")
                        return
                    if (i + 1) % 150 == 0 or i == 0 or i == len(all_paths) - 1:
                        self.log_message.emit(f"Preprocessing image {i + 1}/{len(all_paths)}...", "info")

                    img = cv2.imread(path)
                    if img is None:
                        img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
                    else:
                        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    t = (t - mean) / std
                    tensor_list.append(t)

                all_tensors = torch.stack(tensor_list)
                all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

                try:
                    torch.save({"tensors": all_tensors, "labels": all_labels_tensor}, cache_file)
                except Exception:
                    pass

            # Stratified 70/15/15 split
            indices = np.arange(len(all_labels))
            train_idx, test_idx, train_y, test_y = train_test_split(
                indices, all_labels, test_size=0.30, random_state=42, stratify=all_labels
            )
            val_idx, test_idx, val_y, test_y = train_test_split(
                test_idx, test_y, test_size=0.50, random_state=42, stratify=test_y
            )

            self.log_message.emit(
                f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} (70/15/15 Stratified)",
                "info"
            )

            # In-memory Datasets
            train_ds = InMemoryTensorDataset(all_tensors[train_idx], all_labels_tensor[train_idx])
            val_ds = InMemoryTensorDataset(all_tensors[val_idx], all_labels_tensor[val_idx])
            test_ds = InMemoryTensorDataset(all_tensors[test_idx], all_labels_tensor[test_idx])

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_message.emit(f"Training on device: {device}", "info")

            model = SubjectClassifierNet(num_classes=len(classes))
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            best_val_acc = 0.0
            best_weights = None

            # Training Loop
            self.log_message.emit(f"Starting training for {self.epochs} epochs...", "info")
            start_time = time.time()

            for epoch in range(1, self.epochs + 1):
                if self._stop_requested:
                    self.log_message.emit("Training cancelled by user.", "warning")
                    self.training_failed.emit("Training cancelled.")
                    return

                model.train()
                running_loss = 0.0
                total_train = 0

                for x_b, y_b in train_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    out = model(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * x_b.size(0)
                    total_train += x_b.size(0)

                train_loss = running_loss / max(1, total_train)

                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total_val = 0

                with torch.no_grad():
                    for x_b, y_b in val_loader:
                        x_b, y_b = x_b.to(device), y_b.to(device)
                        out = model(x_b)
                        loss = criterion(out, y_b)
                        val_loss += loss.item() * x_b.size(0)
                        preds = out.argmax(dim=1)
                        correct += (preds == y_b).sum().item()
                        total_val += x_b.size(0)

                val_loss = val_loss / max(1, total_val)
                val_acc = correct / max(1, total_val)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_weights = model.state_dict().copy()

                self.progress_updated.emit(epoch, self.epochs, train_loss, val_loss, val_acc)
                self.log_message.emit(
                    f"Epoch [{epoch:02d}/{self.epochs:02d}] - Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%",
                    "info"
                )

            # Test Evaluation
            self.log_message.emit("Running final evaluation on held-out test set...", "info")
            if best_weights:
                model.load_state_dict(best_weights)
            model.eval()

            y_true, y_pred = [], []
            with torch.no_grad():
                for x_b, y_b in test_loader:
                    x_b = x_b.to(device)
                    out = model(x_b)
                    preds = out.argmax(dim=1).cpu().numpy()
                    y_pred.extend(preds)
                    y_true.extend(y_b.numpy())

            test_acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )

            cm = confusion_matrix(y_true, y_pred).tolist()
            train_duration = round(time.time() - start_time, 2)

            metrics = {
                "accuracy": round(float(test_acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "train_duration_sec": train_duration,
                "epochs": self.epochs,
                "confusion_matrix": cm,
                "classes": classes,
                "dataset_path": report.dataset_path,
                "total_samples": report.total_files,
                "test_samples": len(test_idx),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Save Checkpoint
            os.makedirs(self.output_dir, exist_ok=True)
            model_save_path = os.path.join(self.output_dir, "subject_classifier.pth")
            meta_save_path = os.path.join(self.output_dir, "subject_classifier_meta.json")

            torch.save(model.state_dict(), model_save_path)
            with open(meta_save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_type": "SubjectClassifierNet",
                    "classes": classes,
                    "input_size": list(self.target_size),
                    "metrics": metrics,
                }, f, indent=2)

            self.log_message.emit(
                f"Training complete! Test Accuracy: {test_acc*100:.2f}% | F1: {f1:.4f}",
                "success"
            )
            self.log_message.emit(f"Model saved to {model_save_path}", "success")
            self.training_completed.emit(metrics)

        except Exception as e:
            err = f"Training failed with error: {str(e)}"
            self.log_message.emit(err, "error")
            self.training_failed.emit(err)
