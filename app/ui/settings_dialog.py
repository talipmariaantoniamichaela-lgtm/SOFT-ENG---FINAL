"""
Settings Modal Dialog Component for Motion Reader.
Allows configuring video, detection, display, and terminal settings with persistence.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QSlider, QCheckBox, QComboBox,
    QSpinBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from app.config import AppConfig
from app.services.event_manager import EventManager


class SettingsDialog(QDialog):
    """
    Compact settings modal dialog.
    """
    settings_saved = Signal(AppConfig)

    def __init__(self, config: AppConfig, event_manager: EventManager, parent=None):
        super().__init__(parent)
        self.config = config
        self.event_manager = event_manager
        self.setWindowTitle("Settings — Motion Reader")
        self.setFixedSize(540, 480)
        self.setModal(True)

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Title
        title_lbl = QLabel("Application Settings")
        title_lbl.setStyleSheet("font-size: 17px; font-weight: 700; color: #ffffff;")
        main_layout.addWidget(title_lbl)

        # Tab Widget
        self.tabs = QTabWidget()

        # Tab 1: Video Settings
        tab_video = QWidget()
        v_layout = QVBoxLayout(tab_video)
        v_layout.setSpacing(14)

        lbl_res = QLabel("Preferred Camera Resolution:")
        self.combo_res = QComboBox()
        self.combo_res.addItems(["1280 × 720 (Recommended)", "1920 × 1080 (Full HD)", "640 × 480 (Performance)"])
        if self.config.video.preferred_width == 1920:
            self.combo_res.setCurrentIndex(1)
        elif self.config.video.preferred_width == 640:
            self.combo_res.setCurrentIndex(2)
        else:
            self.combo_res.setCurrentIndex(0)

        v_layout.addWidget(lbl_res)
        v_layout.addWidget(self.combo_res)

        self.chk_mirror = QCheckBox("Mirror camera preview (selfie orientation)")
        self.chk_mirror.setChecked(self.config.display.mirror_camera)
        v_layout.addWidget(self.chk_mirror)

        v_layout.addStretch()
        self.tabs.addTab(tab_video, "Video")

        # Tab 2: Detection Settings
        tab_detect = QWidget()
        d_layout = QVBoxLayout(tab_detect)
        d_layout.setSpacing(12)

        # Detection Confidence
        self.lbl_det_conf = QLabel(f"Min Detection Confidence: {self.config.detection.min_detection_confidence:.2f}")
        self.slider_det_conf = QSlider(Qt.Horizontal)
        self.slider_det_conf.setRange(20, 90)
        self.slider_det_conf.setValue(int(self.config.detection.min_detection_confidence * 100))
        self.slider_det_conf.valueChanged.connect(lambda v: self.lbl_det_conf.setText(f"Min Detection Confidence: {v/100.0:.2f}"))

        d_layout.addWidget(self.lbl_det_conf)
        d_layout.addWidget(self.slider_det_conf)

        # Action Sensitivity
        self.lbl_sens = QLabel(f"Action Sensitivity: {self.config.detection.action_sensitivity:.1f}x")
        self.slider_sens = QSlider(Qt.Horizontal)
        self.slider_sens.setRange(5, 20)
        self.slider_sens.setValue(int(self.config.detection.action_sensitivity * 10))
        self.slider_sens.valueChanged.connect(lambda v: self.lbl_sens.setText(f"Action Sensitivity: {v/10.0:.1f}x"))

        d_layout.addWidget(self.lbl_sens)
        d_layout.addWidget(self.slider_sens)

        self.chk_smoothing = QCheckBox("Enable landmark temporal smoothing (EMA filter)")
        self.chk_smoothing.setChecked(self.config.detection.landmark_smoothing)
        d_layout.addWidget(self.chk_smoothing)

        self.chk_gestures = QCheckBox("Enable gesture recognition (waving, arm raises)")
        self.chk_gestures.setChecked(self.config.detection.enable_gesture_recognition)
        d_layout.addWidget(self.chk_gestures)

        d_layout.addStretch()
        self.tabs.addTab(tab_detect, "Detection")

        # Tab 3: Display Settings
        tab_disp = QWidget()
        disp_layout = QVBoxLayout(tab_disp)
        disp_layout.setSpacing(12)

        self.chk_skeleton = QCheckBox("Show skeletal segment connections (bones)")
        self.chk_skeleton.setChecked(self.config.display.show_skeleton)
        disp_layout.addWidget(self.chk_skeleton)

        self.chk_landmarks = QCheckBox("Show circular landmark joint nodes")
        self.chk_landmarks.setChecked(self.config.display.show_landmarks)
        disp_layout.addWidget(self.chk_landmarks)

        self.chk_hands = QCheckBox("Show detailed 21-point hand tracking")
        self.chk_hands.setChecked(self.config.display.show_hands)
        disp_layout.addWidget(self.chk_hands)

        self.chk_face = QCheckBox("Show facial mesh / contour")
        self.chk_face.setChecked(self.config.display.show_face)
        disp_layout.addWidget(self.chk_face)

        self.chk_fps_badge = QCheckBox("Show performance & status badge (FPS, resolution)")
        self.chk_fps_badge.setChecked(self.config.display.show_fps_badge)
        disp_layout.addWidget(self.chk_fps_badge)

        self.chk_corners = QCheckBox("Show corner tracking brackets")
        self.chk_corners.setChecked(self.config.display.show_corner_markers)
        disp_layout.addWidget(self.chk_corners)

        disp_layout.addStretch()
        self.tabs.addTab(tab_disp, "Display")

        # Tab 4: Terminal Settings
        tab_term = QWidget()
        term_layout = QVBoxLayout(tab_term)
        term_layout.setSpacing(12)

        self.chk_autoscroll = QCheckBox("Auto-scroll to latest recognized event")
        self.chk_autoscroll.setChecked(self.config.terminal.auto_scroll)
        term_layout.addWidget(self.chk_autoscroll)

        term_layout.addSpacing(10)
        lbl_export = QLabel("Export Movement Event Logs:")
        term_layout.addWidget(lbl_export)

        btn_box = QHBoxLayout()
        btn_csv = QPushButton("Export as CSV")
        btn_csv.setObjectName("ModalSecondaryBtn")
        btn_csv.clicked.connect(self._export_csv)

        btn_json = QPushButton("Export as JSON")
        btn_json.setObjectName("ModalSecondaryBtn")
        btn_json.clicked.connect(self._export_json)

        btn_box.addWidget(btn_csv)
        btn_box.addWidget(btn_json)
        term_layout.addLayout(btn_box)

        term_layout.addStretch()
        self.tabs.addTab(tab_term, "Terminal")

        main_layout.addWidget(self.tabs)

        # Bottom Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setObjectName("ModalSecondaryBtn")
        btn_reset.clicked.connect(self._reset_defaults)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("ModalSecondaryBtn")
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("Save Settings")
        btn_save.setObjectName("ModalPrimaryBtn")
        btn_save.clicked.connect(self._save_settings)

        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)

        main_layout.addLayout(btn_row)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Terminal Logs as CSV", "motion_events.csv", "CSV Files (*.csv)")
        if path:
            success = self.event_manager.export_csv(path)
            if success:
                QMessageBox.information(self, "Export Complete", f"Successfully exported logs to:\n{path}")

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Terminal Logs as JSON", "motion_events.json", "JSON Files (*.json)")
        if path:
            success = self.event_manager.export_json(path)
            if success:
                QMessageBox.information(self, "Export Complete", f"Successfully exported logs to:\n{path}")

    def _reset_defaults(self) -> None:
        self.config = AppConfig()
        self.slider_det_conf.setValue(50)
        self.slider_sens.setValue(10)
        self.chk_smoothing.setChecked(True)
        self.chk_gestures.setChecked(True)
        self.chk_skeleton.setChecked(True)
        self.chk_landmarks.setChecked(True)
        self.chk_hands.setChecked(True)
        self.chk_face.setChecked(True)
        self.chk_fps_badge.setChecked(True)
        self.chk_corners.setChecked(True)
        self.chk_mirror.setChecked(True)
        self.chk_autoscroll.setChecked(True)
        self.combo_res.setCurrentIndex(0)

    def _save_settings(self) -> None:
        # Save into config object
        res_idx = self.combo_res.currentIndex()
        if res_idx == 0:
            self.config.video.preferred_width, self.config.video.preferred_height = 1280, 720
        elif res_idx == 1:
            self.config.video.preferred_width, self.config.video.preferred_height = 1920, 1080
        else:
            self.config.video.preferred_width, self.config.video.preferred_height = 640, 480

        self.config.display.mirror_camera = self.chk_mirror.isChecked()
        self.config.detection.min_detection_confidence = self.slider_det_conf.value() / 100.0
        self.config.detection.action_sensitivity = self.slider_sens.value() / 10.0
        self.config.detection.landmark_smoothing = self.chk_smoothing.isChecked()
        self.config.detection.enable_gesture_recognition = self.chk_gestures.isChecked()

        self.config.display.show_skeleton = self.chk_skeleton.isChecked()
        self.config.display.show_landmarks = self.chk_landmarks.isChecked()
        self.config.display.show_hands = self.chk_hands.isChecked()
        self.config.display.show_face = self.chk_face.isChecked()
        self.config.display.show_fps_badge = self.chk_fps_badge.isChecked()
        self.config.display.show_corner_markers = self.chk_corners.isChecked()

        self.config.terminal.auto_scroll = self.chk_autoscroll.isChecked()

        self.config.save()
        self.settings_saved.emit(self.config)
        self.accept()
