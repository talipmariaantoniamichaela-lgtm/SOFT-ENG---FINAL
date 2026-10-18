"""
Main Window for Motion Reader Application.
Coordinates top header, left sidebar with single 'Live View' navigation,
main split workspace (Camera Feed & Motion Terminal), and bottom control bar.
"""
import os
import sys
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QFrame, QApplication, QStackedWidget
)
from PySide6.QtGui import QIcon, QFont, QPixmap
from PySide6.QtCore import Qt, QSize

from typing import Optional
from app.config import AppConfig
from app.services.event_manager import EventManager
from app.video.frame_processor import FrameProcessorThread, SourceMode
from app.video.camera_manager import CameraManager
from app.ui.camera_panel import CameraPanel
from app.ui.terminal_panel import TerminalPanel
from app.ui.training_panel import TrainingPanel
from app.ui.control_bar import ControlBar
from app.ui.settings_dialog import SettingsDialog
from app.ui.help_dialog import HelpDialog
from app.ui.icons import IconFactory
from app.ui.styles import MAIN_STYLESHEET


class MainWindow(QMainWindow):
    """
    Main Application Window replicating the exact UI specification.
    """
    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or AppConfig.load()
        self.event_manager = EventManager(max_entries=self.config.terminal.max_retained_entries)

        # Worker Thread
        self.processor_thread = FrameProcessorThread(self.config, self.event_manager)

        self.setWindowTitle("Motion Reader — Real-time Motion Detection & Pose Tracking")
        self.setMinimumSize(1180, 720)
        self.resize(1360, 840)

        self._init_ui()
        self._connect_signals()
        self._init_camera_devices()

        # Start processing
        self.processor_thread.start()

    def _init_ui(self) -> None:
        self.setStyleSheet(MAIN_STYLESHEET)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # -------------------------------------------------------------
        # A. Top Header
        # -------------------------------------------------------------
        header_widget = QWidget()
        header_widget.setObjectName("HeaderBar")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(18, 0, 16, 0)
        header_layout.setSpacing(12)

        # Logo and Title
        logo_lbl = QLabel()
        logo_lbl.setPixmap(IconFactory.create_app_logo(34))

        title_box = QHBoxLayout()
        title_box.setSpacing(4)
        lbl_motion = QLabel("Motion")
        lbl_motion.setObjectName("AppTitleLabel")
        lbl_reader = QLabel("Reader")
        lbl_reader.setObjectName("AppTitleLabel")
        lbl_reader.setStyleSheet("color: #0088ff;")
        title_box.addWidget(lbl_motion)
        title_box.addWidget(lbl_reader)

        lbl_subtitle = QLabel("Real-time Motion Detection & Pose Tracking")
        lbl_subtitle.setObjectName("AppSubtitleLabel")

        header_layout.addWidget(logo_lbl)
        header_layout.addLayout(title_box)
        header_layout.addWidget(lbl_subtitle)
        header_layout.addStretch()

        # Settings and Help Buttons
        self.btn_settings = QPushButton(" Settings")
        self.btn_settings.setObjectName("HeaderButton")
        self.btn_settings.setIcon(IconFactory.create_gear_icon("#94a3b8", 18))
        self.btn_settings.setCursor(Qt.PointingHandCursor)
        self.btn_settings.clicked.connect(self._open_settings)

        # Divider line
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("color: #1f2b42; max-height: 20px;")

        self.btn_help = QPushButton(" Help")
        self.btn_help.setObjectName("HeaderButton")
        self.btn_help.setIcon(IconFactory.create_help_icon("#94a3b8", 18))
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.clicked.connect(self._open_help)

        header_layout.addWidget(self.btn_settings)
        header_layout.addWidget(divider)
        header_layout.addWidget(self.btn_help)

        # Window Controls (Min, Max, Close)
        btn_min = QPushButton("─")
        btn_min.setObjectName("WindowControlButton")
        btn_min.clicked.connect(self.showMinimized)

        btn_max = QPushButton("□")
        btn_max.setObjectName("WindowControlButton")
        btn_max.clicked.connect(self._toggle_maximize)

        btn_close = QPushButton("✕")
        btn_close.setObjectName("WindowControlButton")
        btn_close.setStyleSheet("QPushButton:hover { background-color: #ef4444; color: white; }")
        btn_close.clicked.connect(self.close)

        header_layout.addSpacing(16)
        header_layout.addWidget(btn_min)
        header_layout.addWidget(btn_max)
        header_layout.addWidget(btn_close)

        root_layout.addWidget(header_widget)

        # -------------------------------------------------------------
        # Middle Area: Sidebar + Main Content Splitter
        # -------------------------------------------------------------
        middle_widget = QWidget()
        middle_layout = QHBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)

        # B. Left Sidebar
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(12, 16, 12, 16)
        sidebar_layout.setSpacing(8)

        self.btn_sidebar_live = QPushButton(" Live View")
        self.btn_sidebar_live.setObjectName("SidebarNavBtnActive")
        self.btn_sidebar_live.setIcon(IconFactory.create_camera_icon("#ffffff", 18))
        self.btn_sidebar_live.setCursor(Qt.PointingHandCursor)
        self.btn_sidebar_live.clicked.connect(self._show_live_view)

        self.btn_sidebar_train = QPushButton(" Training")
        self.btn_sidebar_train.setObjectName("SidebarNavBtnInactive")
        self.btn_sidebar_train.setIcon(IconFactory.create_dataset_icon("#94a3b8", 18))
        self.btn_sidebar_train.setCursor(Qt.PointingHandCursor)
        self.btn_sidebar_train.clicked.connect(self._show_training_view)

        sidebar_layout.addWidget(self.btn_sidebar_live)
        sidebar_layout.addWidget(self.btn_sidebar_train)
        sidebar_layout.addStretch()

        middle_layout.addWidget(sidebar_widget)

        # C. Main Content Workspace (QStackedWidget: 0 = Live View, 1 = Dataset & Training)
        self.view_stack = QStackedWidget()

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setStyleSheet("QSplitter::handle { background-color: transparent; width: 14px; }")
        content_splitter.setContentsMargins(16, 16, 16, 16)

        # Left Panel: Camera Feed
        self.camera_panel = CameraPanel(self.config, self)

        # Right Panel: Motion Terminal
        self.terminal_panel = TerminalPanel(self.config, self.event_manager, self)

        content_splitter.addWidget(self.camera_panel)
        content_splitter.addWidget(self.terminal_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

        self.view_stack.addWidget(content_splitter)

        # Training Studio Panel
        self.training_panel = TrainingPanel(self)
        self.view_stack.addWidget(self.training_panel)

        middle_layout.addWidget(self.view_stack, stretch=1)
        root_layout.addWidget(middle_widget, stretch=1)

        # -------------------------------------------------------------
        # D. Bottom Control Bar
        # -------------------------------------------------------------
        self.control_bar = ControlBar(self.config, self)
        root_layout.addWidget(self.control_bar)

    def _connect_signals(self) -> None:
        # Worker thread signals
        self.processor_thread.frame_ready.connect(self._on_frame_ready)
        self.processor_thread.status_changed.connect(self._on_status_changed)
        self.processor_thread.video_position_changed.connect(self.camera_panel.update_video_progress)
        self.processor_thread.error_occurred.connect(lambda msg: self.event_manager.log_error(msg))

        # Camera Panel interactions
        self.camera_panel.live_camera_selected.connect(self._on_live_camera_selected)
        self.camera_panel.upload_video_selected.connect(self._on_upload_video_selected)
        self.camera_panel.play_pause_toggled.connect(self._toggle_file_play_pause)
        self.camera_panel.restart_requested.connect(self.processor_thread.restart_video)
        self.camera_panel.seek_requested.connect(self.processor_thread.seek_to)

        # Control Bar interactions
        self.control_bar.camera_selected.connect(self._on_camera_dropdown_selected)
        self.control_bar.toggle_clicked.connect(self._on_main_action_toggled)

        # Training Panel model reload
        self.training_panel.model_reloaded.connect(self._on_model_reloaded)

    def _show_live_view(self) -> None:
        """Navigates to Live View (Camera & Terminal)."""
        self.view_stack.setCurrentIndex(0)
        self.btn_sidebar_live.setObjectName("SidebarNavBtnActive")
        self.btn_sidebar_train.setObjectName("SidebarNavBtnInactive")
        self.btn_sidebar_live.setIcon(IconFactory.create_camera_icon("#ffffff", 18))
        self.btn_sidebar_train.setIcon(IconFactory.create_dataset_icon("#94a3b8", 18))
        self.btn_sidebar_live.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0070f3, stop:1 #0088ff); color: #ffffff; border: 1px solid #0099ff;")
        self.btn_sidebar_train.setStyleSheet("background-color: transparent; color: #94a3b8; border: 1px solid transparent;")

    def _show_training_view(self) -> None:
        """Navigates to Dataset & Training Studio."""
        self.view_stack.setCurrentIndex(1)
        self.btn_sidebar_train.setObjectName("SidebarNavBtnActive")
        self.btn_sidebar_live.setObjectName("SidebarNavBtnInactive")
        self.btn_sidebar_train.setIcon(IconFactory.create_dataset_icon("#ffffff", 18))
        self.btn_sidebar_live.setIcon(IconFactory.create_camera_icon("#94a3b8", 18))
        self.btn_sidebar_train.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0070f3, stop:1 #0088ff); color: #ffffff; border: 1px solid #0099ff;")
        self.btn_sidebar_live.setStyleSheet("background-color: transparent; color: #94a3b8; border: 1px solid transparent;")

    def _on_model_reloaded(self, model_path: str, meta_path: str) -> None:
        """Called when a new model is trained or loaded from the Training Panel."""
        success = self.processor_thread.reload_subject_model(model_path, meta_path)
        if success:
            filename = os.path.basename(model_path)
            self.event_manager.log_system_info(f"Custom recognition model loaded: {filename}")

    def _init_camera_devices(self) -> None:
        """Discovers available cameras and loads into dropdown."""
        cameras = CameraManager.enumerate_cameras()
        self.control_bar.populate_cameras(cameras)

    def _on_frame_ready(self, rgb_frame, people, fps, res_str) -> None:
        """Sends processed frame and per-person tracking data to viewport."""
        source_lbl = "LIVE" if self.processor_thread.source_mode == SourceMode.LIVE_CAMERA else "FILE"
        is_p = self.processor_thread._is_paused
        self.camera_panel.viewport.update_frame(
            rgb_frame, people, fps, res_str,
            source_label=source_lbl, is_paused=is_p
        )

    def _on_status_changed(self, status_text: str, status_type: str) -> None:
        """Updates the status indicator in the bottom control bar."""
        self.control_bar.set_status(status_text, status_type)

    def _on_live_camera_selected(self) -> None:
        """Switches to webcam mode."""
        self.control_bar.set_camera_enabled(True)
        cam_idx = self.control_bar.camera_combo.currentData() or 0
        self.processor_thread.set_source_camera(cam_idx)
        self.control_bar.set_running_state(True)

    def _on_upload_video_selected(self, filepath: str) -> None:
        """Switches to video file mode."""
        self.control_bar.set_camera_enabled(False)
        self.processor_thread.set_source_video_file(filepath)
        self.control_bar.set_running_state(True)

    def _on_camera_dropdown_selected(self, camera_index: int) -> None:
        """Changes webcam index."""
        if self.processor_thread.source_mode == SourceMode.LIVE_CAMERA:
            self.processor_thread.set_source_camera(camera_index)

    def _on_main_action_toggled(self, is_running: bool) -> None:
        """Handles central circular Start/Stop button click."""
        if is_running:
            self.processor_thread.play()
        else:
            self.processor_thread.pause()

    def _toggle_file_play_pause(self) -> None:
        """Toggles video file playback."""
        is_playing = self.processor_thread.toggle_play_pause()
        self.camera_panel.btn_play_pause.setText("Pause" if is_playing else "Play")
        self.control_bar.set_running_state(is_playing)

    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self.event_manager, self)
        dialog.settings_saved.connect(self._on_settings_applied)
        dialog.exec()

    def _on_settings_applied(self, new_config: AppConfig) -> None:
        self.config = new_config
        # Re-apply detection options
        self.processor_thread.apply_detection_settings(self.config)
        self.camera_panel.viewport.config = self.config
        self.camera_panel.viewport.update()

    def _open_help(self) -> None:
        dialog = HelpDialog(self)
        dialog.exec()

    def closeEvent(self, event) -> None:
        """Clean shutdown of worker thread and cameras."""
        self.processor_thread.stop_worker()
        super().closeEvent(event)
