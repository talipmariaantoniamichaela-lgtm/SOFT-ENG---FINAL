"""
UI Stylesheet and Theme Definitions for Motion Reader.
Dark navy and electric-blue aesthetic closely matching the reference specification.
"""

MAIN_STYLESHEET = """
/* Global Window & Fonts */
QWidget {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    font-size: 13px;
    selection-background-color: #0070f3;
    selection-color: #ffffff;
}

/* Main Window */
QMainWindow {
    background-color: #0b0f19;
}

/* Header Bar */
#HeaderBar {
    background-color: #0d121f;
    border-bottom: 1px solid #1a2235;
    min-height: 54px;
    max-height: 54px;
    padding: 0 16px;
}

#AppTitleLabel {
    font-size: 19px;
    font-weight: 700;
    color: #ffffff;
}

#AppTitleReader {
    color: #0088ff;
}

#AppSubtitleLabel {
    font-size: 12px;
    color: #94a3b8;
    margin-left: 12px;
}

#HeaderButton {
    background-color: transparent;
    color: #94a3b8;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
}

#HeaderButton:hover {
    background-color: #1e293b;
    color: #ffffff;
}

#WindowControlButton {
    background-color: transparent;
    color: #94a3b8;
    border: none;
    border-radius: 4px;
    width: 28px;
    height: 28px;
    font-size: 14px;
}

#WindowControlButton:hover {
    background-color: #1e293b;
    color: #ffffff;
}

#WindowCloseButton:hover {
    background-color: #ef4444;
    color: #ffffff;
}

/* Left Sidebar */
#Sidebar {
    background-color: #0d121f;
    border-right: 1px solid #1a2235;
    min-width: 160px;
    max-width: 175px;
    padding: 14px 10px;
}

#SidebarLiveViewBtn, #SidebarNavBtnActive {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0070f3, stop:1 #0088ff);
    color: #ffffff;
    border: 1px solid #0099ff;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 600;
    text-align: left;
}

#SidebarNavBtnInactive {
    background-color: transparent;
    color: #94a3b8;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 500;
    text-align: left;
}

#SidebarNavBtnInactive:hover {
    background-color: #162032;
    color: #e2e8f0;
}

/* Content Panels */
#PanelCard {
    background-color: #101726;
    border: 1px solid #1c273c;
    border-radius: 12px;
}

#PanelHeader {
    padding: 12px 16px 8px 16px;
}

#PanelTitle {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
}

/* Video Feed Box */
#VideoContainer {
    background-color: #050811;
    border: 1px solid #182338;
    border-radius: 10px;
}

/* Source Selection Buttons */
#SourceButtonActive {
    background-color: #0b1a33;
    border: 2px solid #0088ff;
    border-radius: 10px;
    padding: 10px 16px;
    text-align: left;
}

#SourceButtonActive:hover {
    background-color: #0e2142;
}

#SourceButtonInactive {
    background-color: #131c2e;
    border: 1px solid #1f2b42;
    border-radius: 10px;
    padding: 10px 16px;
    text-align: left;
}

#SourceButtonInactive:hover {
    background-color: #19243b;
    border-color: #2a3b5a;
}

#SourceTitle {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
}

#SourceSubtitle {
    font-size: 11px;
    color: #94a3b8;
}

/* Terminal View */
#TerminalCard {
    background-color: #101726;
    border: 1px solid #1c273c;
    border-radius: 12px;
}

#TerminalView {
    background-color: #06090e;
    border: 1px solid #151d2d;
    border-radius: 8px;
    color: #e2e8f0;
    font-family: 'Consolas', 'Cascadia Code', 'Segoe UI Mono', monospace;
    font-size: 13px;
    line-height: 1.5;
    padding: 12px;
}

#ClearButton {
    background-color: #151e30;
    color: #94a3b8;
    border: 1px solid #202d45;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
}

#ClearButton:hover {
    background-color: #1e2b45;
    color: #ffffff;
    border-color: #2e3f60;
}

/* Bottom Control Bar */
#BottomBar {
    background-color: #090d16;
    border-top: 1px solid #161f31;
    min-height: 72px;
    max-height: 72px;
    padding: 0 24px;
}

#CameraDropdown {
    background-color: #121a2b;
    color: #ffffff;
    border: 1px solid #1e2b42;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 500;
    min-width: 170px;
}

#CameraDropdown:hover {
    border-color: #2e3f60;
    background-color: #162136;
}

#CameraDropdown::drop-down {
    border: none;
    width: 20px;
}

#CameraDropdown QAbstractItemView {
    background-color: #121a2b;
    border: 1px solid #1e2b42;
    color: #ffffff;
    selection-background-color: #0070f3;
    padding: 4px;
}

/* Large Circular Action Buttons */
#StopButton {
    background-color: #ef4444;
    border: none;
    border-radius: 28px;
    min-width: 56px;
    max-width: 56px;
    min-height: 56px;
    max-height: 56px;
}

#StopButton:hover {
    background-color: #dc2626;
}

#StartButton {
    background-color: #0088ff;
    border: none;
    border-radius: 28px;
    min-width: 56px;
    max-width: 56px;
    min-height: 56px;
    max-height: 56px;
}

#StartButton:hover {
    background-color: #0070f3;
}

/* Tracking Status Indicator */
#StatusLabel {
    font-size: 14px;
    font-weight: 500;
    color: #e2e8f0;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #090d16;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #1e293b;
    min-height: 25px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #334155;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 4px;
    background: #1e293b;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #0088ff;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #0088ff;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

/* Dialogs */
QDialog {
    background-color: #0d121f;
    border: 1px solid #1e293b;
    border-radius: 12px;
}

QTabWidget::pane {
    border: 1px solid #1c273c;
    border-radius: 8px;
    background-color: #101726;
    padding: 16px;
}

QTabBar::tab {
    background-color: #0d121f;
    color: #94a3b8;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #101726;
    color: #ffffff;
    font-weight: 600;
}

QCheckBox {
    color: #e2e8f0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #2d3b55;
    border-radius: 4px;
    background-color: #121a2b;
}

QCheckBox::indicator:checked {
    background-color: #0088ff;
    border-color: #0088ff;
    image: url(app/assets/icons/check.png);
}

QPushButton#ModalPrimaryBtn {
    background-color: #0088ff;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
}

QPushButton#ModalPrimaryBtn:hover {
    background-color: #0070f3;
}

QPushButton#ModalSecondaryBtn {
    background-color: #1a2336;
    color: #94a3b8;
    border: 1px solid #283750;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 500;
}

QPushButton#ModalSecondaryBtn:hover {
    background-color: #222e47;
    color: #ffffff;
}
"""
