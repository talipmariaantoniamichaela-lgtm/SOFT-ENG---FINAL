"""
Vector Icon Generator and Icon Factory for Motion Reader.
Uses QPainter vector drawing to render crisp, high-DPI icons for all UI elements.
"""
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath, QFont
from PySide6.QtCore import Qt, QRectF, QPointF


class IconFactory:
    """Creates crisp vector icons on demand."""

    @staticmethod
    def create_app_logo(size: int = 40) -> QPixmap:
        """Draws the running person logo with motion streaks in electric blue."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        blue = QColor("#0088ff")
        light_blue = QColor("#38bdf8")

        # Draw motion speed streaks
        pen_streak = QPen(light_blue, 2.5, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_streak)
        painter.drawLine(int(size * 0.08), int(size * 0.28), int(size * 0.28), int(size * 0.28))
        painter.drawLine(int(size * 0.05), int(size * 0.50), int(size * 0.25), int(size * 0.50))
        painter.drawLine(int(size * 0.12), int(size * 0.72), int(size * 0.32), int(size * 0.72))

        # Runner Head
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(blue))
        painter.drawEllipse(QRectF(size * 0.58, size * 0.12, size * 0.22, size * 0.22))

        # Runner Body and Limbs
        body_pen = QPen(blue, 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(body_pen)

        # Torso
        painter.drawLine(int(size * 0.64), int(size * 0.34), int(size * 0.52), int(size * 0.58))
        # Front arm
        painter.drawLine(int(size * 0.60), int(size * 0.38), int(size * 0.44), int(size * 0.42))
        painter.drawLine(int(size * 0.44), int(size * 0.42), int(size * 0.48), int(size * 0.28))
        # Back arm
        painter.drawLine(int(size * 0.60), int(size * 0.38), int(size * 0.76), int(size * 0.46))
        # Front leg
        painter.drawLine(int(size * 0.52), int(size * 0.58), int(size * 0.68), int(size * 0.70))
        painter.drawLine(int(size * 0.68), int(size * 0.70), int(size * 0.88), int(size * 0.66))
        # Back leg
        painter.drawLine(int(size * 0.52), int(size * 0.58), int(size * 0.38), int(size * 0.72))
        painter.drawLine(int(size * 0.38), int(size * 0.72), int(size * 0.46), int(size * 0.90))

        painter.end()
        return pixmap

    @staticmethod
    def create_camera_icon(color_hex: str = "#ffffff", size: int = 24) -> QIcon:
        """Video camera icon."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        color = QColor(color_hex)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)

        # Camera body
        painter.drawRoundedRect(QRectF(size * 0.1, size * 0.25, size * 0.55, size * 0.50), 3.0, 3.0)

        # Lens triangle
        path = QPainterPath()
        path.moveTo(size * 0.65, size * 0.38)
        path.lineTo(size * 0.90, size * 0.25)
        path.lineTo(size * 0.90, size * 0.75)
        path.lineTo(size * 0.65, size * 0.62)
        path.closeSubpath()
        painter.drawPath(path)

        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def create_upload_icon(color_hex: str = "#ffffff", size: int = 24) -> QIcon:
        """Upload tray icon with upward arrow."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(color_hex), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        # Arrow shaft and head
        painter.drawLine(int(size * 0.5), int(size * 0.65), int(size * 0.5), int(size * 0.20))
        path = QPainterPath()
        path.moveTo(size * 0.32, size * 0.38)
        path.lineTo(size * 0.50, size * 0.20)
        path.lineTo(size * 0.68, size * 0.38)
        painter.drawPath(path)

        # Tray
        tray = QPainterPath()
        tray.moveTo(size * 0.20, size * 0.58)
        tray.lineTo(size * 0.20, size * 0.80)
        tray.lineTo(size * 0.80, size * 0.80)
        tray.lineTo(size * 0.80, size * 0.58)
        painter.drawPath(tray)

        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def create_terminal_icon(color_hex: str = "#0088ff", size: int = 24) -> QIcon:
        """Terminal prompt `>_` icon."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Rounded background badge
        painter.setBrush(QBrush(QColor("#182338")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(size * 0.1, size * 0.1, size * 0.8, size * 0.8), 4.0, 4.0)

        pen = QPen(QColor(color_hex), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        # > chevron
        path = QPainterPath()
        path.moveTo(size * 0.28, size * 0.32)
        path.lineTo(size * 0.48, size * 0.50)
        path.lineTo(size * 0.28, size * 0.68)
        painter.drawPath(path)

        # _ underscore
        painter.drawLine(int(size * 0.54), int(size * 0.68), int(size * 0.72), int(size * 0.68))

        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def create_gear_icon(color_hex: str = "#94a3b8", size: int = 20) -> QIcon:
        """Settings gear icon."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(color_hex), 2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        # Outer cog circle
        painter.drawEllipse(QRectF(size * 0.22, size * 0.22, size * 0.56, size * 0.56))
        # Inner hole
        painter.drawEllipse(QRectF(size * 0.38, size * 0.38, size * 0.24, size * 0.24))

        # Teeth
        teeth_pen = QPen(QColor(color_hex), 2.2, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(teeth_pen)
        for angle in range(0, 360, 45):
            rad = angle * 3.14159 / 180.0
            x1 = size * 0.5 + size * 0.28 * 0.8 * float(f"{angle}")  # or use math
            # Simple radial teeth
            painter.save()
            painter.translate(size * 0.5, size * 0.5)
            painter.rotate(angle)
            painter.drawLine(0, -int(size * 0.46), 0, -int(size * 0.35))
            painter.restore()

        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def create_help_icon(color_hex: str = "#94a3b8", size: int = 20) -> QIcon:
        """Help question mark circle icon."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(color_hex), 1.8)
        painter.setPen(pen)
        painter.drawEllipse(QRectF(size * 0.15, size * 0.15, size * 0.70, size * 0.70))

        # Draw ? text
        font = QFont("Segoe UI", int(size * 0.42), QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, "?")

        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def create_trash_icon(color_hex: str = "#94a3b8", size: int = 16) -> QIcon:
        """Trash can icon for clear button."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(color_hex), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        # Lid
        painter.drawLine(int(size * 0.18), int(size * 0.28), int(size * 0.82), int(size * 0.28))
        painter.drawLine(int(size * 0.38), int(size * 0.20), int(size * 0.62), int(size * 0.20))

        # Can body
        path = QPainterPath()
        path.moveTo(size * 0.25, size * 0.28)
        path.lineTo(size * 0.30, size * 0.85)
        path.lineTo(size * 0.70, size * 0.85)
        path.lineTo(size * 0.75, size * 0.28)
        painter.drawPath(path)

        # Lines inside
        painter.drawLine(int(size * 0.42), int(size * 0.40), int(size * 0.42), int(size * 0.75))
        painter.drawLine(int(size * 0.58), int(size * 0.40), int(size * 0.58), int(size * 0.75))

        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def create_dataset_icon(color_hex: str = "#94a3b8", size: int = 20) -> QIcon:
        """Database / dataset icon for Dataset & Training navigation."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(color_hex), 1.6, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)

        # Layers / database records
        y_positions = [size * 0.22, size * 0.50, size * 0.78]
        for y in y_positions:
            painter.drawRoundedRect(QRectF(size * 0.18, y - size * 0.09, size * 0.64, size * 0.18), 3.0, 3.0)

        painter.end()
        return QIcon(pixmap)
