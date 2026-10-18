"""
Motion Terminal Panel Component for Motion Reader.
Renders real-time, color-coded movement logs, supports auto-scrolling,
and provides clear logs functionality.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser
)
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtCore import Qt, Signal

from app.services.event_manager import MovementEvent, EventManager
from app.config import AppConfig
from app.ui.icons import IconFactory


class TerminalPanel(QWidget):
    """
    Real-time Motion Terminal UI panel with color-coded event stream.
    """
    clear_requested = Signal()

    def __init__(self, config: AppConfig, event_manager: EventManager, parent=None):
        super().__init__(parent)
        self.config = config
        self.event_manager = event_manager
        self.setObjectName("TerminalCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        # 1. Header (Icon + "Motion Terminal" + "Clear" Button)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        terminal_icon_lbl = QLabel()
        terminal_icon_lbl.setPixmap(IconFactory.create_terminal_icon("#0088ff", 22).pixmap(22, 22))

        title_lbl = QLabel("Motion Terminal")
        title_lbl.setObjectName("PanelTitle")

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("ClearButton")
        self.btn_clear.setIcon(IconFactory.create_trash_icon("#94a3b8", 16))
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self._on_clear_clicked)

        header_layout.addWidget(terminal_icon_lbl)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_clear)

        layout.addLayout(header_layout)

        # 2. Terminal Text Browser
        self.terminal_browser = QTextBrowser()
        self.terminal_browser.setObjectName("TerminalView")
        self.terminal_browser.setOpenExternalLinks(False)
        self.terminal_browser.setReadOnly(True)

        # Typography
        mono_font = QFont("Consolas", 11)
        mono_font.setStyleHint(QFont.Monospace)
        self.terminal_browser.setFont(mono_font)

        layout.addWidget(self.terminal_browser, stretch=1)

    def _connect_signals(self) -> None:
        self.event_manager.event_added.connect(self.append_event)
        self.event_manager.events_cleared.connect(self.clear_display)

    def append_event(self, event: MovementEvent) -> None:
        """Appends formatted HTML line to the terminal."""
        html_line = event.formatted_html()
        self.terminal_browser.append(html_line)

        # Auto-scroll if enabled
        if self.config.terminal.auto_scroll:
            cursor = self.terminal_browser.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.terminal_browser.setTextCursor(cursor)

    def clear_display(self) -> None:
        """Clears the terminal text widget."""
        self.terminal_browser.clear()

    def _on_clear_clicked(self) -> None:
        self.event_manager.clear()
        self.clear_requested.emit()
