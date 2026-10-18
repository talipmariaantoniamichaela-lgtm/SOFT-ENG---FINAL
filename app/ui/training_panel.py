"""
Dataset & Training Management Panel for Motion Reader AI.
Allows discovering datasets, configuring hyperparameters,
running background PyTorch training, viewing live epoch progress & metrics,
and managing the active recognition model.
"""
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSpinBox, QComboBox, QTextEdit, QFrame,
    QGridLayout, QGroupBox, QScrollArea, QMessageBox
)
from PySide6.QtGui import QFont, QIcon, QColor
from PySide6.QtCore import Qt, Signal

from app.services.training_service import DatasetValidator, TrainingWorkerThread
from app.ui.icons import IconFactory


class TrainingPanel(QWidget):
    """
    Compact, modern dark-themed panel for dataset inspection, model training,
    and checkpoint management.
    """
    model_reloaded = Signal(str, str)  # model_path, meta_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[TrainingWorkerThread] = None
        self._init_ui()
        self._check_active_model()
        self.validate_dataset_info()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        title_box = QHBoxLayout()
        title_icon = QLabel()
        title_icon.setPixmap(IconFactory.create_gear_icon("#0088ff", 24).pixmap(24, 24))
        title_lbl = QLabel("Dataset & Training Console")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #ffffff;")
        subtitle_lbl = QLabel("Train and integrate custom subject recognition models")
        subtitle_lbl.setStyleSheet("font-size: 12px; color: #94a3b8; margin-left: 12px;")

        title_box.addWidget(title_icon)
        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        title_box.addStretch()

        main_layout.addLayout(title_box)

        # Two-Column Layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(16)

        # -------------------------------------------------------------
        # Left Column: Dataset & Controls
        # -------------------------------------------------------------
        left_col = QVBoxLayout()
        left_col.setSpacing(14)

        # 1. Dataset Card
        dataset_card = QFrame()
        dataset_card.setStyleSheet("background-color: #0d121f; border: 1px solid #1a2235; border-radius: 8px; padding: 14px;")
        ds_layout = QVBoxLayout(dataset_card)
        ds_layout.setSpacing(10)

        ds_header = QHBoxLayout()
        ds_title = QLabel("DATASET INFORMATION")
        ds_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;")
        self.lbl_dataset_status = QLabel("Checking...")
        self.lbl_dataset_status.setStyleSheet("background-color: #1e293b; color: #94a3b8; padding: 2px 8px; border-radius: 4px; font-size: 11px;")
        ds_header.addWidget(ds_title)
        ds_header.addStretch()
        ds_header.addWidget(self.lbl_dataset_status)
        ds_layout.addLayout(ds_header)

        self.lbl_dataset_path = QLabel("Path: Searching...")
        self.lbl_dataset_path.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_dataset_path.setWordWrap(True)
        ds_layout.addWidget(self.lbl_dataset_path)

        # Stats grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(8)

        lbl_s_title = QLabel("Total Samples:")
        lbl_s_title.setStyleSheet("color: #94a3b8;")
        self.lbl_sample_count = QLabel("0")
        self.lbl_sample_count.setStyleSheet("font-weight: 700; color: #ffffff;")

        lbl_c_title = QLabel("Classes:")
        lbl_c_title.setStyleSheet("color: #94a3b8;")
        self.lbl_class_names = QLabel("None")
        self.lbl_class_names.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.lbl_class_names.setWordWrap(True)

        stats_grid.addWidget(lbl_s_title, 0, 0)
        stats_grid.addWidget(self.lbl_sample_count, 0, 1)
        stats_grid.addWidget(lbl_c_title, 1, 0)
        stats_grid.addWidget(self.lbl_class_names, 1, 1)
        ds_layout.addLayout(stats_grid)

        self.btn_validate = QPushButton("Validate Dataset")
        self.btn_validate.setStyleSheet("background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 5px; padding: 6px;")
        self.btn_validate.setCursor(Qt.PointingHandCursor)
        self.btn_validate.clicked.connect(self.validate_dataset_info)
        ds_layout.addWidget(self.btn_validate)

        left_col.addWidget(dataset_card)

        # 2. Training Hyperparameters Card
        params_card = QFrame()
        params_card.setStyleSheet("background-color: #0d121f; border: 1px solid #1a2235; border-radius: 8px; padding: 14px;")
        p_layout = QVBoxLayout(params_card)
        p_layout.setSpacing(10)

        p_title = QLabel("TRAINING CONFIGURATION")
        p_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;")
        p_layout.addWidget(p_title)

        p_grid = QGridLayout()
        p_grid.setSpacing(8)

        # Epochs
        p_grid.addWidget(QLabel("Epochs:"), 0, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 50)
        self.spin_epochs.setValue(8)
        self.spin_epochs.setStyleSheet("background-color: #161f30; border: 1px solid #2d3b55; border-radius: 4px; padding: 4px; color: white;")
        p_grid.addWidget(self.spin_epochs, 0, 1)

        # Batch Size
        p_grid.addWidget(QLabel("Batch Size:"), 1, 0)
        self.combo_batch = QComboBox()
        self.combo_batch.addItems(["16", "32", "64"])
        self.combo_batch.setCurrentText("32")
        self.combo_batch.setStyleSheet("background-color: #161f30; border: 1px solid #2d3b55; border-radius: 4px; padding: 4px; color: white;")
        p_grid.addWidget(self.combo_batch, 1, 1)

        # Learning Rate
        p_grid.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.combo_lr = QComboBox()
        self.combo_lr.addItems(["0.001", "0.0005", "0.0001"])
        self.combo_lr.setStyleSheet("background-color: #161f30; border: 1px solid #2d3b55; border-radius: 4px; padding: 4px; color: white;")
        p_grid.addWidget(self.combo_lr, 2, 1)

        p_layout.addLayout(p_grid)

        # Action Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        self.btn_start_train = QPushButton("Start Training")
        self.btn_start_train.setStyleSheet("""
            QPushButton { background-color: #0088ff; color: white; font-weight: 600; border-radius: 6px; padding: 8px 16px; border: none; }
            QPushButton:hover { background-color: #0070d8; }
            QPushButton:disabled { background-color: #1e293b; color: #64748b; }
        """)
        self.btn_start_train.setCursor(Qt.PointingHandCursor)
        self.btn_start_train.clicked.connect(self._start_training)

        self.btn_cancel_train = QPushButton("Cancel")
        self.btn_cancel_train.setEnabled(False)
        self.btn_cancel_train.setStyleSheet("""
            QPushButton { background-color: #334155; color: white; border-radius: 6px; padding: 8px 12px; border: none; }
            QPushButton:hover { background-color: #ef4444; }
            QPushButton:disabled { background-color: #1e293b; color: #475569; }
        """)
        self.btn_cancel_train.clicked.connect(self._cancel_training)

        btn_box.addWidget(self.btn_start_train, stretch=2)
        btn_box.addWidget(self.btn_cancel_train, stretch=1)
        p_layout.addLayout(btn_box)

        left_col.addWidget(params_card)
        left_col.addStretch()
        columns_layout.addLayout(left_col, stretch=1)

        # -------------------------------------------------------------
        # Right Column: Progress & Terminal Logs
        # -------------------------------------------------------------
        right_col = QVBoxLayout()
        right_col.setSpacing(14)

        # 1. Active Model Status Card
        model_card = QFrame()
        model_card.setStyleSheet("background-color: #0d121f; border: 1px solid #1a2235; border-radius: 8px; padding: 14px;")
        m_layout = QVBoxLayout(model_card)
        m_layout.setSpacing(8)

        m_header = QHBoxLayout()
        m_title = QLabel("ACTIVE RECOGNITION MODEL")
        m_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;")
        self.lbl_model_badge = QLabel("Not Trained")
        self.lbl_model_badge.setStyleSheet("background-color: #1e293b; color: #f59e0b; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;")
        m_header.addWidget(m_title)
        m_header.addStretch()
        m_header.addWidget(self.lbl_model_badge)
        m_layout.addLayout(m_header)

        self.lbl_model_info = QLabel("No trained model weights found. Train a model from the dataset.")
        self.lbl_model_info.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        m_layout.addWidget(self.lbl_model_info)

        reload_btn = QPushButton("Reload Active Model")
        reload_btn.setStyleSheet("background-color: #161f30; color: #38bdf8; border: 1px solid #1e293b; border-radius: 4px; padding: 4px 10px; font-size: 11px;")
        reload_btn.clicked.connect(self._check_active_model)
        m_layout.addWidget(reload_btn, alignment=Qt.AlignLeft)

        right_col.addWidget(model_card)

        # 2. Training Progress Card
        progress_card = QFrame()
        progress_card.setStyleSheet("background-color: #0d121f; border: 1px solid #1a2235; border-radius: 8px; padding: 14px;")
        prg_layout = QVBoxLayout(progress_card)
        prg_layout.setSpacing(10)

        prg_title = QLabel("TRAINING PROGRESS")
        prg_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;")
        prg_layout.addWidget(prg_title)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background-color: #161f30; border: 1px solid #1e293b; border-radius: 5px; height: 16px; text-align: center; color: white; font-size: 11px; }
            QProgressBar::chunk { background-color: #0088ff; border-radius: 4px; }
        """)
        prg_layout.addWidget(self.progress_bar)

        # Live Metrics KPI Box
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(10)

        self.kpi_epoch = self._create_kpi_widget("EPOCH", "-")
        self.kpi_loss = self._create_kpi_widget("TRAIN LOSS", "-")
        self.kpi_val_acc = self._create_kpi_widget("VAL ACCURACY", "-")
        self.kpi_test_f1 = self._create_kpi_widget("TEST F1", "-")

        kpi_layout.addWidget(self.kpi_epoch)
        kpi_layout.addWidget(self.kpi_loss)
        kpi_layout.addWidget(self.kpi_val_acc)
        kpi_layout.addWidget(self.kpi_test_f1)
        prg_layout.addLayout(kpi_layout)

        # Training Log Terminal
        log_header = QHBoxLayout()
        log_title = QLabel("TRAINING LOGS")
        log_title.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("background: transparent; color: #64748b; border: none; font-size: 11px;")
        clear_btn.clicked.connect(lambda: self.txt_log.clear())
        log_header.addWidget(log_title)
        log_header.addStretch()
        log_header.addWidget(clear_btn)
        prg_layout.addLayout(log_header)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("""
            QTextEdit { background-color: #080c14; border: 1px solid #161f30; border-radius: 6px; color: #94a3b8; font-family: 'Consolas', monospace; font-size: 11px; padding: 8px; }
        """)
        self.txt_log.setFixedHeight(180)
        prg_layout.addWidget(self.txt_log)

        right_col.addWidget(progress_card)
        columns_layout.addLayout(right_col, stretch=2)

        main_layout.addLayout(columns_layout)

    def _create_kpi_widget(self, title: str, initial_val: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("background-color: #111726; border: 1px solid #1a2235; border-radius: 6px; padding: 6px 10px;")
        l = QVBoxLayout(card)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)

        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("font-size: 10px; color: #64748b; font-weight: 600;")
        v_lbl = QLabel(initial_val)
        v_lbl.setObjectName("kpi_value")
        v_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #ffffff;")

        l.addWidget(t_lbl)
        l.addWidget(v_lbl)
        return card

    def _update_kpi(self, widget: QFrame, val: str, color: Optional[str] = None) -> None:
        lbl = widget.findChild(QLabel, "kpi_value")
        if lbl:
            lbl.setText(val)
            if color:
                lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {color};")

    def validate_dataset_info(self) -> None:
        self.report = DatasetValidator.validate_dataset()
        report = self.report
        if report.is_valid:
            self.lbl_dataset_status.setText("Validated & Ready")
            self.lbl_dataset_status.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;")
            self.lbl_dataset_path.setText(f"Path: {report.dataset_path}")
            self.lbl_sample_count.setText(f"{report.total_files} images")
            breakdown = ", ".join([f"{k} ({v})" for k, v in report.class_counts.items()])
            self.lbl_class_names.setText(breakdown)
            self.log("Dataset verified successfully: " + breakdown, "success")
        else:
            self.lbl_dataset_status.setText("Missing / Invalid")
            self.lbl_dataset_status.setStyleSheet("background-color: #7f1d1d; color: #f87171; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;")
            self.lbl_dataset_path.setText(f"Path: {report.dataset_path or 'Not found'}")
            self.lbl_sample_count.setText("0")
            self.lbl_class_names.setText("None")
            self.log(f"Dataset validation failed: {report.message}", "error")

    def _check_active_model(self) -> None:
        models_dir = Path(__file__).resolve().parent.parent / "assets" / "models"
        model_path = models_dir / "subject_classifier.pth"
        meta_path = models_dir / "subject_classifier_meta.json"

        if model_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                acc = meta.get("metrics", {}).get("accuracy", 0.0)
                classes = meta.get("classes", [])
                date = meta.get("metrics", {}).get("trained_at", "")

                self.lbl_model_badge.setText("Active & Loaded")
                self.lbl_model_badge.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;")
                self.lbl_model_info.setText(
                    f"Model: subject_classifier.pth | Test Accuracy: {acc*100:.1f}%\n"
                    f"Enrolled Classes ({len(classes)}): {', '.join(classes)}\n"
                    f"Trained: {date}"
                )
                self.model_reloaded.emit(str(model_path), str(meta_path))
            except Exception as e:
                self.lbl_model_badge.setText("Error")
                self.lbl_model_info.setText(f"Error reading model metadata: {e}")
        else:
            self.lbl_model_badge.setText("Not Trained")
            self.lbl_model_badge.setStyleSheet("background-color: #1e293b; color: #f59e0b; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;")
            self.lbl_model_info.setText("No custom subject recognition model is currently active.")

    def _start_training(self) -> None:
        epochs = self.spin_epochs.value()
        batch_size = int(self.combo_batch.currentText())
        lr = float(self.combo_lr.currentText())

        self.btn_start_train.setEnabled(False)
        self.btn_cancel_train.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log(f"Initializing training: Epochs={epochs}, BatchSize={batch_size}, LR={lr}...", "info")

        self.worker = TrainingWorkerThread(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr
        )
        self.worker.log_message.connect(self.log)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.training_completed.connect(self._on_completed)
        self.worker.training_failed.connect(self._on_failed)
        self.worker.start()

    def _cancel_training(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Cancelling training worker...", "warning")
            self.btn_cancel_train.setEnabled(False)

    def _on_progress(self, ep: int, tot: int, t_loss: float, v_loss: float, v_acc: float) -> None:
        pct = int((ep / tot) * 100)
        self.progress_bar.setValue(pct)
        self._update_kpi(self.kpi_epoch, f"{ep}/{tot}")
        self._update_kpi(self.kpi_loss, f"{t_loss:.4f}")
        self._update_kpi(self.kpi_val_acc, f"{v_acc*100:.1f}%", "#22c55e")

    def _on_completed(self, metrics: dict) -> None:
        self.progress_bar.setValue(100)
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        self._update_kpi(self.kpi_test_f1, f"{f1:.4f}", "#38bdf8")

        self.btn_start_train.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        self._check_active_model()
        QMessageBox.information(
            self,
            "Training Complete",
            f"Recognition model trained successfully!\n\n"
            f"Test Accuracy: {acc*100:.2f}%\n"
            f"Weighted F1 Score: {f1:.4f}\n"
            f"Active classes: {', '.join(metrics.get('classes', []))}\n\n"
            "The model has been automatically loaded into Live Camera and Uploaded Video analysis."
        )

    def _on_failed(self, err: str) -> None:
        self.btn_start_train.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        self.log(f"Training stopped: {err}", "error")

    def log(self, text: str, level: str = "info") -> None:
        colors = {
            "info": "#94a3b8",
            "success": "#34d399",
            "warning": "#facc15",
            "error": "#f87171",
        }
        color = colors.get(level, "#94a3b8")
        timestamp = time.strftime("%H:%M:%S")
        html = f"<span style='color: #475569;'>[{timestamp}]</span> <span style='color: {color};'>{text}</span>"
        self.txt_log.append(html)
