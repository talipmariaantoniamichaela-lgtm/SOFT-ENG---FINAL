"""
Bottom Control Bar Component for Motion Reader.
Houses camera selection dropdown, large circular Start/Stop action button,
and real-time tracking status indicator.
"""
from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QComboBox, QPushButton, QLabel, QFrame
)
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QPolygonF, QPainterPath
from PySide6.QtCore import Qt, Signal, QRectF, QPointF

from app.ui.icons import IconFactory
from app.config import AppConfig


class CircularActionButton(QPushButton):
    """
    Large circular control button (56x56) that toggles between
    Red Stop (with white rounded square) and Blue/Green Start (with white play triangle).
    """
    def __init__(self, is_running: bool = True, parent=None):
        super().__init__(parent)
        self.is_running = is_running
        self.setFixedSize(56, 56)
        self.setCursor(Qt.PointingHandCursor)

    def set_running_state(self, running: bool) -> None:
        self.is_running = running
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        size = min(self.width(), self.height())
        center = QPointF(size / 2.0, size / 2.0)
        radius = size / 2.0

        # Background circle
        if self.is_running:
            # Red Stop State
            bg_color = QColor("#ef4444")
            if self.underMouse():
                bg_color = QColor("#dc2626")
        else:
            # Blue Start State
            bg_color = QColor("#0088ff")
            if self.underMouse():
                bg_color = QColor("#0070f3")

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawEllipse(center, radius, radius)

        # Foreground Icon
        painter.setBrush(QBrush(QColor("#ffffff")))

        if self.is_running:
            # White rounded square (Stop)
            sq_size = 18.0
            sq_rect = QRectF(center.x() - sq_size / 2.0, center.y() - sq_size / 2.0, sq_size, sq_size)
            painter.drawRoundedRect(sq_rect, 3.0, 3.0)
        else:
            # White Play Triangle
            tri = QPolygonF([
                QPointF(center.x() - 7.0, center.y() - 10.0),
                QPointF(center.x() + 11.0, center.y()),
                QPointF(center.x() - 7.0, center.y() + 10.0)
            ])
            painter.drawPolygon(tri)


class StatusDot(QWidget):
    """Small circular glowing status dot."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.status_type = "active"

    def set_status_type(self, status_type: str) -> None:
        self.status_type = status_type
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.status_type == "active":
            dot_color = QColor("#22c55e")      # Bright green
        elif self.status_type == "warning":
            dot_color = QColor("#eab308")      # Yellow / orange
        elif self.status_type == "error":
            dot_color = QColor("#ef4444")      # Red
        else:
            dot_color = QColor("#94a3b8")      # Neutral gray

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(2, 2, 10, 10)


class StatusIndicator(QWidget):
    """
    Tracking status dot and label.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status_text = "Tracking..."
        self.status_type = "active"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.dot_widget = StatusDot(self)
        self.lbl_text = QLabel(self.status_text)
        self.lbl_text.setObjectName("StatusLabel")

        layout.addWidget(self.dot_widget)
        layout.addWidget(self.lbl_text)

    def set_status(self, text: str, status_type: str = "active") -> None:
        self.status_text = text
        self.status_type = status_type
        self.lbl_text.setText(text)
        self.dot_widget.set_status_type(status_type)


class ControlBar(QWidget):
    """
    Bottom control bar matching the reference image.
    """
    camera_selected = Signal(int)
    toggle_clicked = Signal(bool)  # emits True if running, False if stopped

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setObjectName("BottomBar")

        self.is_running = True
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 0, 28, 0)
        layout.setSpacing(24)

        # Left / Center-Left: Camera selection dropdown
        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("CameraDropdown")
        self.camera_combo.setCursor(Qt.PointingHandCursor)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

        # Center: Large Action Button (Stop/Start)
        self.btn_action = CircularActionButton(is_running=True)
        self.btn_action.clicked.connect(self._on_action_clicked)

        # Center-Right: Tracking Status
        self.status_indicator = StatusIndicator()

        # Layout positioning
        layout.addStretch(1)
        layout.addWidget(self.camera_combo)
        layout.addSpacing(32)
        layout.addWidget(self.btn_action)
        layout.addSpacing(32)
        layout.addWidget(self.status_indicator)
        layout.addStretch(1)

    def populate_cameras(self, cameras: List[Dict[str, Any]]) -> None:
        """Populates camera list into dropdown."""
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for cam in cameras:
            self.camera_combo.addItem(cam["name"], cam["index"])
        self.camera_combo.blockSignals(False)

    def set_camera_enabled(self, enabled: bool) -> None:
        """Enables or disables camera selection dropdown (e.g. during video playback)."""
        self.camera_combo.setEnabled(enabled)

    def set_running_state(self, is_running: bool) -> None:
        self.is_running = is_running
        self.btn_action.set_running_state(is_running)

    def set_status(self, text: str, status_type: str = "active") -> None:
        self.status_indicator.set_status(text, status_type)

    def _on_camera_changed(self, index: int) -> None:
        cam_idx = self.camera_combo.currentData()
        if cam_idx is not None:
            self.camera_selected.emit(int(cam_idx))

    def _on_action_clicked(self) -> None:
        self.is_running = not self.is_running
        self.btn_action.set_running_state(self.is_running)
        self.toggle_clicked.emit(self.is_running)
