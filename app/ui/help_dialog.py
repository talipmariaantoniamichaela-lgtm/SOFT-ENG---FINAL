"""
Help Modal Dialog Component for Motion Reader.
Provides complete user guide, movement recognition reference, and troubleshooting.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser
)
from PySide6.QtCore import Qt


class HelpDialog(QDialog):
    """
    Help modal dialog styled consistently with dark navy & electric blue theme.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help & Documentation — Motion Reader")
        self.setFixedSize(620, 520)
        self.setModal(True)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 20)
        layout.setSpacing(16)

        # Header
        title_lbl = QLabel("Motion Reader Documentation")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #ffffff;")
        layout.addWidget(title_lbl)

        # Help Content Browser
        browser = QTextBrowser()
        browser.setStyleSheet("""
            background-color: #0d1424;
            border: 1px solid #1c273c;
            border-radius: 8px;
            color: #e2e8f0;
            padding: 16px;
            font-size: 13px;
            line-height: 1.6;
        """)

        help_html = """
        <h3 style="color: #0088ff; margin-top: 0;">1. System Overview</h3>
        <p><b>Motion Reader</b> is a real-time computer vision system that tracks human movement, classifies postures, identifies gestures, and streams timestamped actions directly into the Motion Terminal.</p>

        <h3 style="color: #0088ff;">2. Source Selection</h3>
        <ul>
            <li><b>Live Camera:</b> Select the "Live Camera" card or use the bottom camera selector dropdown. Supports any USB or integrated webcam.</li>
            <li><b>Upload Video:</b> Select the "Upload Video" card to load an MP4, MOV, AVI, or WebM video file. When in video mode, seek bar, play/pause, and restart controls appear automatically.</li>
        </ul>

        <h3 style="color: #0088ff;">3. Skeletal Tracking Overlay</h3>
        <ul>
            <li><b>Bones:</b> Electric-blue lines indicate anatomical skeletal links between joints.</li>
            <li><b>Joints:</b> Two-tone circular markers represent detected landmarks (shoulders, elbows, wrists, hips, knees, ankles).</li>
            <li><b>Hands & Face:</b> Detailed 21-point hand skeleton and facial contour are tracked whenever subjects are clearly visible.</li>
        </ul>

        <h3 style="color: #0088ff;">4. Recognized Movement Events</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; border-color: #1f2b42; width: 100%;">
            <tr style="background-color: #121b2d; color: #94a3b8;">
                <th>Category</th>
                <th>Terminal Color</th>
                <th>Supported Movements</th>
            </tr>
            <tr>
                <td><b>Person Detection</b></td>
                <td style="color: #22c55e;">Green</td>
                <td>Maria detected, Person detected (ID: 2), Mark left the frame</td>
            </tr>
            <tr>
                <td><b>Neutral Posture</b></td>
                <td style="color: #ffffff;">White</td>
                <td>Standing, Sitting, Crouching, Lying down, Idle</td>
            </tr>
            <tr>
                <td><b>Hand Movements</b></td>
                <td style="color: #38bdf8;">Cyan</td>
                <td>Right hand raised, Right hand lowered, Left hand raised, Left hand lowered, Both hands raised</td>
            </tr>
            <tr>
                <td><b>Recognized Gestures</b></td>
                <td style="color: #facc15;">Yellow</td>
                <td>Waving, Clapping, Jumping, Nodding, Shaking head</td>
            </tr>
            <tr>
                <td><b>Body Movements</b></td>
                <td style="color: #c084fc;">Purple</td>
                <td>Walking, Running, Stepping forward/backward, Turning left/right, Leaning left/right, Bending over, Arms crossed, Arms stretched out, Hands on head, Hands on hips, Touching face, Pointing left/right</td>
            </tr>
            <tr>
                <td><b>System Messages</b></td>
                <td style="color: #94a3b8;">Gray</td>
                <td>Device status, resolution, video playback progress</td>
            </tr>
        </table>

        <h3 style="color: #0088ff;">5. Duplicate Suppression & Hysteresis</h3>
        <p>The system uses stateful hysteresis and temporal sliding windows to prevent spamming duplicate events. Sustained postures (like Standing) only log when transitioning into or out of that posture.</p>

        <h3 style="color: #0088ff;">6. Troubleshooting</h3>
        <ul>
            <li><b>Camera not opening:</b> Ensure other apps (Zoom, Teams, Browser) are not locking the webcam. You can switch camera index in the bottom dropdown.</li>
            <li><b>Low FPS:</b> In Settings &rarr; Video, select 640 &times; 480 or toggle off hand/face tracking if running on low-power hardware.</li>
        </ul>
        """
        browser.setHtml(help_html)
        layout.addWidget(browser)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.setObjectName("ModalPrimaryBtn")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
