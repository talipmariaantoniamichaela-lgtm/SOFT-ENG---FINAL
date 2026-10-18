"""
Smoke test to verify MainWindow and TrainingPanel UI initialization and model binding.
"""
import os
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from app.config import AppConfig
from app.ui.main_window import MainWindow


def test_main_window_and_training_panel_integration():
    app = QApplication.instance() or QApplication([])

    config = AppConfig.load()
    window = MainWindow(config=config)

    # Verify MainWindow components
    assert window.processor_thread is not None
    assert window.camera_panel is not None
    assert window.terminal_panel is not None
    assert window.training_panel is not None

    # Verify custom trained model is loaded
    assert window.processor_thread.subject_engine.is_loaded is True
    assert len(window.processor_thread.subject_engine.classes) == 5

    # Verify TrainingPanel dataset info
    assert window.training_panel.report is not None
    assert window.training_panel.report.total_files == 900

    # Verify view navigation
    window._show_training_view()
    assert window.view_stack.currentIndex() == 1

    window._show_live_view()
    assert window.view_stack.currentIndex() == 0

    # Clean shutdown
    window.processor_thread.stop_worker()
    window.close()
