"""
Automated Unit Tests for Subject Classifier, Dataset Validator,
Trained Model Checkpoint Loading, and Recognition Integration.
"""
import os
import cv2
import torch
import pytest
from pathlib import Path

from app.detection.subject_classifier import (
    SubjectClassifierNet,
    SubjectRecognitionEngine,
    SubjectPrediction
)
from app.services.training_service import DatasetValidator
from app.services.event_manager import EventManager, EventType


def test_dataset_validator_discovery():
    """Validates automatic dataset directory discovery, classes, and file counts."""
    dataset_dir = DatasetValidator.find_dataset_dir()
    assert dataset_dir is not None, "Dataset directory should be automatically discovered"
    assert os.path.exists(dataset_dir), f"Path must exist: {dataset_dir}"

    info = DatasetValidator.validate_dataset(dataset_dir)
    assert info.is_valid is True
    # The original five; locally captured people (capture_person.py) may be added
    assert {"Dainna", "Lawrenz", "Maria", "Mark", "Quinneth"} <= set(info.classes)
    assert info.total_files == sum(info.class_counts.values())
    assert info.class_counts["Dainna"] == 200
    assert info.class_counts["Lawrenz"] == 200
    assert info.class_counts["Maria"] == 200
    assert info.class_counts["Mark"] == 200
    assert info.class_counts["Quinneth"] == 100


def test_subject_classifier_net_architecture():
    """Validates PyTorch neural net architecture forward pass and tensor dimensions."""
    model = SubjectClassifierNet(num_classes=5)
    model.eval()

    dummy_input = torch.randn(4, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 5)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_trained_model_checkpoint():
    """Validates that the trained model weights and metadata exist and load correctly."""
    base_dir = Path(__file__).resolve().parent.parent
    model_path = str(base_dir / "app" / "assets" / "models" / "subject_classifier.pth")
    meta_path = str(base_dir / "app" / "assets" / "models" / "subject_classifier_meta.json")

    assert os.path.exists(model_path), f"Checkpoint missing: {model_path}"
    assert os.path.exists(meta_path), f"Metadata missing: {meta_path}"

    engine = SubjectRecognitionEngine(model_path=model_path, meta_path=meta_path)
    assert engine.is_loaded is True
    assert len(engine.classes) == 5
    assert "Maria" in engine.classes
    assert engine.metrics.get("accuracy", 0) >= 0.90
    assert engine.metrics.get("f1_score", 0) >= 0.90


def test_real_sample_inference():
    """Tests inference on an actual sample image from the dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = DatasetValidator.find_dataset_dir()
    assert dataset_dir is not None

    maria_files = list((Path(dataset_dir) / "Maria").glob("*.jpg"))
    assert len(maria_files) > 0, "No JPG files found in Maria directory"
    sample_img_path = maria_files[0]

    img = cv2.imread(str(sample_img_path))
    assert img is not None

    model_path = str(base_dir / "app" / "assets" / "models" / "subject_classifier.pth")
    engine = SubjectRecognitionEngine(model_path=model_path)
    assert engine.is_loaded is True

    # Run inference with is_bgr=True
    prediction = engine.predict(img, is_bgr=True)
    assert isinstance(prediction, SubjectPrediction)
    assert prediction.is_known is True
    assert prediction.subject_name == "Maria"
    assert prediction.confidence >= 0.85
    assert "Maria" in prediction.class_probabilities


def test_event_manager_with_subject_identity():
    """Tests that EventManager integrates subject names into logged descriptions and formats."""
    manager = EventManager()
    manager.hit_threshold = 1  # Instant trigger for test
    t = 1000.0

    # Person detected as Maria
    manager.process_frame_analysis(
        person_detected=True,
        person_id=1,
        posture="Standing",
        left_hand_raised=False,
        right_hand_raised=False,
        gestures=[],
        body_movements=[],
        current_time=t,
        is_video_relative=False,
        subject_name="Maria",
        subject_confidence=0.96
    )

    assert len(manager.events) >= 1
    assert manager.events[0].description == "Maria detected"
    assert manager.events[0].confidence == 0.96
    assert "Maria detected" in manager.events[0].formatted_line()

    # Maria raises right hand
    t += 1.0
    manager.process_frame_analysis(
        person_detected=True,
        person_id=1,
        posture="Standing",
        left_hand_raised=False,
        right_hand_raised=True,
        gestures=[],
        body_movements=[],
        current_time=t,
        is_video_relative=False,
        subject_name="Maria",
        subject_confidence=0.96
    )

    descriptions = [e.description for e in manager.events]
    assert "Maria raises right hand" in descriptions
