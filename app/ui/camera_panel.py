"""
Camera Feed Panel Component for Motion Reader.
Displays video feed, skeletal tracking overlay, corner brackets,
source selection buttons (Live Camera / Upload Video), and file playback controls.
"""
import math
from typing import Optional, List, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QFrame
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QImage, QPixmap, QFont,
    QPainterPath, QIcon
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF

from app.detection.pose_estimator import (
    TrackingResult, NormalizedLandmark, POSE_CONNECTIONS, HAND_CONNECTIONS,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE
)
from app.detection.person_tracker import PersonSnapshot
from app.config import AppConfig
from app.ui.icons import IconFactory


# Distinct colors for recognized people; unknown people are drawn in the default blue
_NAME_COLORS = ["#22d3ee", "#a3e635", "#f472b6", "#fbbf24", "#a78bfa", "#fb923c", "#34d399", "#f87171"]
_UNKNOWN_COLOR = "#00d2ff"


def person_color(person: PersonSnapshot) -> QColor:
    if not person.name:
        return QColor(_UNKNOWN_COLOR)
    return QColor(_NAME_COLORS[sum(map(ord, person.name)) % len(_NAME_COLORS)])


class VideoViewport(QWidget):
    """
    Hardware-accelerated custom video rendering widget with skeletal tracking,
    corner markers, and live performance badge.
    """
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setMinimumSize(480, 320)
        self.setStyleSheet("background-color: #050811; border-radius: 10px;")

        # Current frame and tracking state
        self.current_image: Optional[QImage] = None
        self.people: List[PersonSnapshot] = []
        self.fps: float = 30.0
        self.resolution_str: str = "1280 × 720"
        self.source_label: str = "LIVE"
        self.is_paused: bool = False

    def update_frame(
        self,
        rgb_data,
        people: List[PersonSnapshot],
        fps: float,
        res_str: str,
        source_label: str = "LIVE",
        is_paused: bool = False
    ) -> None:
        """Updates frame data and triggers repaint."""
        h, w, ch = rgb_data.shape
        bytes_per_line = ch * w
        self.current_image = QImage(rgb_data.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.people = list(people or [])
        self.fps = fps
        self.resolution_str = res_str
        self.source_label = source_label
        self.is_paused = is_paused
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Background fill
        painter.fillRect(self.rect(), QColor("#050811"))

        if self.current_image is None or self.current_image.isNull():
            # Idle placeholder
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for video stream...")
            return

        # 2. Compute letterbox / aspect-ratio preserved target rect
        widget_w, widget_h = self.width(), self.height()
        img_w, img_h = self.current_image.width(), self.current_image.height()

        scale = min(widget_w / img_w, widget_h / img_h)
        target_w = img_w * scale
        target_h = img_h * scale
        offset_x = (widget_w - target_w) / 2.0
        offset_y = (widget_h - target_h) / 2.0
        video_rect = QRectF(offset_x, offset_y, target_w, target_h)

        # Draw video image with rounded clip
        clip_path = QPainterPath()
        clip_path.addRoundedRect(video_rect, 8.0, 8.0)
        painter.save()
        painter.setClipPath(clip_path)
        painter.drawImage(video_rect, self.current_image)

        # 3. Skeleton, box and identity / action tag for every person
        for person in self.people:
            color = person_color(person)
            self._draw_skeleton(painter, person.tracking, color, offset_x, offset_y, target_w, target_h)
        for person in self.people:
            self._draw_person_tag(painter, person, person_color(person), offset_x, offset_y, target_w, target_h)

        painter.restore()

        # 5. Draw Corner Tracking Markers
        if self.config.display.show_corner_markers:
            self._draw_corner_markers(painter, video_rect)

        # 6. Draw Top-Left Status Badge (LIVE | FPS: 30 | 1280 × 720)
        if self.config.display.show_fps_badge:
            self._draw_status_badge(painter, video_rect)

    def _draw_person_tag(self, painter: QPainter, person: PersonSnapshot, color: QColor,
                         ox: float, oy: float, tw: float, th: float) -> None:
        """Draws the person's box, a name pill, and a line listing what they are doing."""
        bbox = person.tracking.bounding_box
        if not bbox:
            return
        min_x, min_y, max_x, max_y = bbox
        bx, by = ox + min_x * tw, oy + min_y * th
        bw, bh = (max_x - min_x) * tw, (max_y - min_y) * th

        box_color = QColor(color)
        box_color.setAlpha(150)
        painter.setPen(QPen(box_color, 1.5, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(bx, by, bw, bh), 6.0, 6.0)

        # Name pill
        if person.name:
            name_text = f"● {person.name}  {person.name_score:.2f}"
        else:
            name_text = f"● Unknown (ID {person.track_id})" if self.config.display.show_person_id else "● Unknown"
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        fm = painter.fontMetrics()
        tag_h = 24.0
        tag_w = fm.horizontalAdvance(name_text) + 18.0
        tag_x = max(ox + 8.0, min(ox + tw - tag_w - 8.0, bx))
        tag_y = max(oy + 8.0, by - tag_h - 26.0)
        pill_rect = QRectF(tag_x, tag_y, tag_w, tag_h)
        painter.setPen(QPen(color, 1.2))
        painter.setBrush(QBrush(QColor(11, 15, 25, 225)))
        painter.drawRoundedRect(pill_rect, 12.0, 12.0)
        painter.setPen(color)
        painter.drawText(pill_rect, Qt.AlignCenter, name_text)

        # Actions line under the name
        labels = person.analysis.summary_labels() if person.analysis else []
        if labels:
            painter.setFont(QFont("Segoe UI", 8))
            fm = painter.fontMetrics()
            act_text = " · ".join(dict.fromkeys(labels))
            act_text = fm.elidedText(act_text, Qt.ElideRight, int(max(160.0, min(tw * 0.6, 420.0))))
            act_w = fm.horizontalAdvance(act_text) + 16.0
            act_rect = QRectF(tag_x, tag_y + tag_h + 2.0, act_w, 20.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(11, 15, 25, 200)))
            painter.drawRoundedRect(act_rect, 10.0, 10.0)
            painter.setPen(QColor("#e2e8f0"))
            painter.drawText(act_rect, Qt.AlignCenter, act_text)

    def _draw_corner_markers(self, painter: QPainter, rect: QRectF) -> None:
        """Draws subtle cyan/blue L-brackets at the four corners of the video frame."""
        pen = QPen(QColor("#00d2ff"), 2.0, Qt.SolidLine, Qt.SquareCap)
        painter.setPen(pen)
        margin = 14.0
        arm_len = 22.0

        x1 = rect.left() + margin
        y1 = rect.top() + margin
        x2 = rect.right() - margin
        y2 = rect.bottom() - margin

        # Top-Left ┌
        painter.drawLine(int(x1), int(y1), int(x1 + arm_len), int(y1))
        painter.drawLine(int(x1), int(y1), int(x1), int(y1 + arm_len))

        # Top-Right ┐
        painter.drawLine(int(x2), int(y1), int(x2 - arm_len), int(y1))
        painter.drawLine(int(x2), int(y1), int(x2), int(y1 + arm_len))

        # Bottom-Left └
        painter.drawLine(int(x1), int(y2), int(x1 + arm_len), int(y2))
        painter.drawLine(int(x1), int(y2), int(x1), int(y2 - arm_len))

        # Bottom-Right ┘
        painter.drawLine(int(x2), int(y2), int(x2 - arm_len), int(y2))
        painter.drawLine(int(x2), int(y2), int(x2), int(y2 - arm_len))

    def _draw_status_badge(self, painter: QPainter, video_rect: QRectF) -> None:
        """Draws semi-transparent pill badge with live status, FPS, and resolution."""
        badge_x = video_rect.left() + 16.0
        badge_y = video_rect.top() + 16.0
        badge_h = 30.0

        # Badge text content
        status_txt = self.source_label if not self.is_paused else "PAUSED"
        fps_txt = f"FPS: {int(round(self.fps))}"
        res_txt = self.resolution_str

        # Measure text width
        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        txt_full = f"  {status_txt}   {fps_txt}   {res_txt}  "
        badge_w = fm.horizontalAdvance(txt_full) + 24.0

        badge_rect = QRectF(badge_x, badge_y, badge_w, badge_h)

        # Background pill
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.setBrush(QBrush(QColor(17, 24, 39, 210)))
        painter.drawRoundedRect(badge_rect, 15.0, 15.0)

        # Pulsing dot
        dot_color = QColor("#22c55e") if not self.is_paused else QColor("#eab308")
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(QRectF(badge_x + 10.0, badge_y + 10.0, 10.0, 10.0))

        # Status text
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(badge_x + 26.0, badge_y, fm.horizontalAdvance(status_txt) + 10, badge_h), Qt.AlignVCenter, status_txt)

        # Divider and metrics text
        metrics_x = badge_x + 28.0 + fm.horizontalAdvance(status_txt)
        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Segoe UI", 10))
        metrics_str = f"|  {fps_txt}  |  {res_txt}"
        painter.drawText(QRectF(metrics_x, badge_y, badge_w - (metrics_x - badge_x), badge_h), Qt.AlignVCenter, metrics_str)

    def _draw_skeleton(self, painter: QPainter, tracking: TrackingResult, color: QColor,
                       ox: float, oy: float, tw: float, th: float) -> None:
        """Draws glowing skeletal lines and circular joint nodes for one person."""
        lms = tracking.pose_landmarks
        if not lms or len(lms) < 33:
            return

        def to_screen(lm: NormalizedLandmark) -> QPointF:
            return QPointF(ox + lm.x * tw, oy + lm.y * th)

        bone_color = QColor(color)
        joint_outer_color = QColor(color).darker(130)
        joint_inner_color = QColor("#ffffff")

        # 1. Draw Bones (connected segments)
        if self.config.display.show_skeleton:
            bone_pen = QPen(bone_color, 2.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(bone_pen)

            for idx1, idx2 in POSE_CONNECTIONS:
                if idx1 < len(lms) and idx2 < len(lms):
                    p1 = lms[idx1]
                    p2 = lms[idx2]
                    if p1.visibility > 0.4 and p2.visibility > 0.4:
                        painter.drawLine(to_screen(p1), to_screen(p2))

            # Draw central spine connection (chin/mid-shoulder down to mid-hip)
            l_sh = lms[LEFT_SHOULDER]
            r_sh = lms[RIGHT_SHOULDER]
            l_hip = lms[LEFT_HIP]
            r_hip = lms[RIGHT_HIP]
            nose = lms[NOSE]

            if l_sh.visibility > 0.4 and r_sh.visibility > 0.4:
                mid_sh = NormalizedLandmark(x=(l_sh.x + r_sh.x) / 2.0, y=(l_sh.y + r_sh.y) / 2.0, z=0.0)
                if nose.visibility > 0.4:
                    painter.drawLine(to_screen(nose), to_screen(mid_sh))

                if l_hip.visibility > 0.4 and r_hip.visibility > 0.4:
                    mid_hip = NormalizedLandmark(x=(l_hip.x + r_hip.x) / 2.0, y=(l_hip.y + r_hip.y) / 2.0, z=0.0)
                    painter.drawLine(to_screen(mid_sh), to_screen(mid_hip))

            # 2. Draw Hand Landmarks & Bones (if detected)
            if self.config.display.show_hands:
                hand_pen = QPen(QColor("#38bdf8"), 2.0, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(hand_pen)
                for hands in [tracking.left_hand_landmarks, tracking.right_hand_landmarks]:
                    if hands and len(hands) == 21:
                        # Draw finger bones
                        for i1, i2 in HAND_CONNECTIONS:
                            painter.drawLine(to_screen(hands[i1]), to_screen(hands[i2]))
                        # Draw finger joints
                        for h_lm in hands:
                            pt = to_screen(h_lm)
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(QBrush(joint_outer_color))
                            painter.drawEllipse(pt, 3.2, 3.2)
                            painter.setBrush(QBrush(joint_inner_color))
                            painter.drawEllipse(pt, 1.8, 1.8)

            # 3. Draw Face Mesh / Features (if detected)
            if self.config.display.show_face and tracking.face_landmarks:
                face_pen = QPen(bone_color, 1.2, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(face_pen)
                # Draw facial oval / jawline contour from key face landmarks
                f_lms = tracking.face_landmarks
                # Selected indices for clean face contour
                oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                if len(f_lms) > 400:
                    face_path = QPainterPath()
                    first_pt = to_screen(f_lms[oval_indices[0]])
                    face_path.moveTo(first_pt)
                    for o_idx in oval_indices[1:]:
                        face_path.lineTo(to_screen(f_lms[o_idx]))
                    face_path.closeSubpath()
                    painter.drawPath(face_path)

        # 4. Draw Major Anatomical Joint Nodes
        if self.config.display.show_landmarks:
            major_joint_indices = [
                LEFT_SHOULDER, RIGHT_SHOULDER,
                LEFT_ELBOW, RIGHT_ELBOW,
                LEFT_WRIST, RIGHT_WRIST,
                LEFT_HIP, RIGHT_HIP,
                LEFT_KNEE, RIGHT_KNEE,
                LEFT_ANKLE, RIGHT_ANKLE
            ]
            for idx in major_joint_indices:
                if idx < len(lms):
                    lm = lms[idx]
                    if lm.visibility > 0.4:
                        pt = to_screen(lm)
                        # Outer electric-blue ring
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(joint_outer_color))
                        painter.drawEllipse(pt, 5.8, 5.8)
                        # Inner white core
                        painter.setBrush(QBrush(joint_inner_color))
                        painter.drawEllipse(pt, 3.4, 3.4)


class CameraPanel(QWidget):
    """
    Main Camera Feed panel comprising header, video viewport,
    source-selection cards (Live Camera & Upload Video), and file controls.
    """
    live_camera_selected = Signal()
    upload_video_selected = Signal(str)
    play_pause_toggled = Signal()
    restart_requested = Signal()
    seek_requested = Signal(float)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setObjectName("PanelCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.current_source_is_file = False
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        # 1. Panel Header (icon + "Camera Feed")
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        cam_icon_lbl = QLabel()
        cam_icon_lbl.setPixmap(IconFactory.create_camera_icon("#0088ff", 22).pixmap(22, 22))

        title_lbl = QLabel("Camera Feed")
        title_lbl.setObjectName("PanelTitle")

        header_layout.addWidget(cam_icon_lbl)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # 2. Video Viewport
        self.viewport = VideoViewport(self.config, self)
        layout.addWidget(self.viewport, stretch=1)

        # 3. File Playback Controls (visible only in file mode)
        self.file_controls_widget = QWidget()
        file_ctrl_layout = QHBoxLayout(self.file_controls_widget)
        file_ctrl_layout.setContentsMargins(0, 4, 0, 4)
        file_ctrl_layout.setSpacing(10)

        self.btn_play_pause = QPushButton("Pause")
        self.btn_play_pause.setObjectName("ClearButton")
        self.btn_play_pause.clicked.connect(self.play_pause_toggled.emit)

        self.btn_restart = QPushButton("Restart")
        self.btn_restart.setObjectName("ClearButton")
        self.btn_restart.clicked.connect(self.restart_requested.emit)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(lambda v: self.seek_requested.emit(v / 1000.0))

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color: #94a3b8; font-family: monospace; font-size: 11px;")

        file_ctrl_layout.addWidget(self.btn_play_pause)
        file_ctrl_layout.addWidget(self.btn_restart)
        file_ctrl_layout.addWidget(self.seek_slider, stretch=1)
        file_ctrl_layout.addWidget(self.lbl_time)

        self.file_controls_widget.setVisible(False)
        layout.addWidget(self.file_controls_widget)

        # 4. Source Selection Buttons (Live Camera vs Upload Video)
        source_layout = QHBoxLayout()
        source_layout.setSpacing(12)

        # Live Camera Button
        self.btn_live_camera = QPushButton()
        self.btn_live_camera.setCursor(Qt.PointingHandCursor)
        self.btn_live_camera.setObjectName("SourceButtonActive")
        self.btn_live_camera.setMinimumHeight(68)
        btn_live_layout = QHBoxLayout(self.btn_live_camera)
        btn_live_layout.setContentsMargins(16, 8, 16, 8)
        btn_live_layout.setSpacing(14)

        icon_cam = QLabel()
        icon_cam.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        icon_cam.setStyleSheet("background: transparent;")
        icon_cam.setPixmap(IconFactory.create_camera_icon("#0088ff", 28).pixmap(28, 28))

        text_cam_box = QVBoxLayout()
        text_cam_box.setSpacing(2)
        lbl_cam_title = QLabel("Live Camera")
        lbl_cam_title.setObjectName("SourceTitle")
        lbl_cam_title.setStyleSheet("background: transparent;")
        lbl_cam_title.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        lbl_cam_sub = QLabel("Use your webcam for real-time analysis")
        lbl_cam_sub.setObjectName("SourceSubtitle")
        lbl_cam_sub.setStyleSheet("background: transparent;")
        lbl_cam_sub.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        text_cam_box.addWidget(lbl_cam_title)
        text_cam_box.addWidget(lbl_cam_sub)

        btn_live_layout.addWidget(icon_cam)
        btn_live_layout.addLayout(text_cam_box)
        btn_live_layout.addStretch()

        self.btn_live_camera.clicked.connect(self._on_live_camera_clicked)

        # Upload Video Button
        self.btn_upload_video = QPushButton()
        self.btn_upload_video.setCursor(Qt.PointingHandCursor)
        self.btn_upload_video.setObjectName("SourceButtonInactive")
        self.btn_upload_video.setMinimumHeight(68)
        btn_upload_layout = QHBoxLayout(self.btn_upload_video)
        btn_upload_layout.setContentsMargins(16, 8, 16, 8)
        btn_upload_layout.setSpacing(14)

        icon_upload = QLabel()
        icon_upload.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        icon_upload.setStyleSheet("background: transparent;")
        icon_upload.setPixmap(IconFactory.create_upload_icon("#94a3b8", 28).pixmap(28, 28))

        text_upload_box = QVBoxLayout()
        text_upload_box.setSpacing(2)
        lbl_upload_title = QLabel("Upload Video")
        lbl_upload_title.setObjectName("SourceTitle")
        lbl_upload_title.setStyleSheet("background: transparent;")
        lbl_upload_title.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        lbl_upload_sub = QLabel("Analyze a video file")
        lbl_upload_sub.setObjectName("SourceSubtitle")
        lbl_upload_sub.setStyleSheet("background: transparent;")
        lbl_upload_sub.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        text_upload_box.addWidget(lbl_upload_title)
        text_upload_box.addWidget(lbl_upload_sub)

        btn_upload_layout.addWidget(icon_upload)
        btn_upload_layout.addLayout(text_upload_box)
        btn_upload_layout.addStretch()

        self.btn_upload_video.clicked.connect(self._on_upload_video_clicked)

        source_layout.addWidget(self.btn_live_camera, stretch=1)
        source_layout.addWidget(self.btn_upload_video, stretch=1)

        layout.addLayout(source_layout)

    def _on_live_camera_clicked(self) -> None:
        self.btn_live_camera.setObjectName("SourceButtonActive")
        self.btn_upload_video.setObjectName("SourceButtonInactive")
        self.btn_live_camera.style().unpolish(self.btn_live_camera)
        self.btn_live_camera.style().polish(self.btn_live_camera)
        self.btn_upload_video.style().unpolish(self.btn_upload_video)
        self.btn_upload_video.style().polish(self.btn_upload_video)
        self.current_source_is_file = False
        self.file_controls_widget.setVisible(False)
        self.live_camera_selected.emit()

    def _on_upload_video_clicked(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File to Analyze",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*.*)"
        )
        if filepath:
            self.set_active_source_file(filepath)
            self.upload_video_selected.emit(filepath)

    def set_active_source_file(self, filepath: str) -> None:
        """Updates UI styling for uploaded video mode."""
        self.btn_live_camera.setObjectName("SourceButtonInactive")
        self.btn_upload_video.setObjectName("SourceButtonActive")
        self.btn_live_camera.style().unpolish(self.btn_live_camera)
        self.btn_live_camera.style().polish(self.btn_live_camera)
        self.btn_upload_video.style().unpolish(self.btn_upload_video)
        self.btn_upload_video.style().polish(self.btn_upload_video)
        self.current_source_is_file = True
        self.file_controls_widget.setVisible(True)

    def update_video_progress(self, current_sec: float, total_sec: float, progress_pct: float) -> None:
        """Updates the seek slider and elapsed time label."""
        self.seek_slider.setValue(int(progress_pct * 1000))
        cur_m = int(current_sec) // 60
        cur_s = int(current_sec) % 60
        tot_m = int(total_sec) // 60
        tot_s = int(total_sec) % 60
        self.lbl_time.setText(f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}")
