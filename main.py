"""
Motion Reader — Real-time Motion Detection and Pose Tracking
Main entrypoint.
"""
import sys
import os
import argparse

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt

from app.config import AppConfig
from app.ui.main_window import MainWindow
from app.ui.icons import IconFactory


def main():
    parser = argparse.ArgumentParser(description="Motion Reader — Real-time Motion Detection & Pose Tracking")
    parser.add_argument("--camera", type=int, default=None, help="Initial webcam device index")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for immediate analysis")
    args = parser.parse_args()

    # Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Motion Reader")
    app.setOrganizationName("MotionReaderAI")

    # Set Window Icon
    logo_pixmap = IconFactory.create_app_logo(64)
    app.setWindowIcon(QIcon(logo_pixmap))

    # Load configuration
    config = AppConfig.load()
    if args.camera is not None:
        config.video.camera_index = args.camera

    # Instantiate Main Window
    window = MainWindow(config=config)

    # If video file argument provided, switch to it immediately
    if args.video and os.path.exists(args.video):
        window.camera_panel.set_active_source_file(args.video)
        window.processor_thread.set_source_video_file(args.video)
        window.control_bar.set_camera_enabled(False)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
