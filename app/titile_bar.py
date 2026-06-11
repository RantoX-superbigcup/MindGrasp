from PyQt5.QtWidgets import (
    QMainWindow, QWidget,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PyQt5.QtGui import QIcon
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QPoint, QSize
from Resource import resource

class CustomTitleBar(QWidget):
    def __init__(self, parent_window: QMainWindow = None):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.moving = False
        self.offset = QPoint()
        self.init_title()

    def init_title(self):
        self.setFixedHeight(72)
        self.setStyleSheet("background: #FFFFFF; border-bottom: 1px solid #E6EAF0;")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(32, 0, 32, 0)
        self.layout.setSpacing(12)

        self.icon_label = QSvgWidget(":/icon/title.svg")
        self.icon_label.setFixedSize(6, 22)

        self.title_label = QLabel("MindGrasp 控制台")
        self.title_label.setFixedSize(260, 28)
        self.title_label.setStyleSheet("""
            QLabel {
                font-family: "SourceHanSansCN";
                font-weight: 600;
                font-size: 21px;
                color: #20242A;
                border: none;
                background: transparent;
            }
        """)

        self.close_button = QPushButton()
        self.close_button.setIcon(QIcon(":/icon/close_btn.svg"))
        self.close_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                padding: 0px;
            }
        """)
        self.close_button.setIconSize(QSize(20, 20))
        self.close_button.setFixedSize(20, 20)
        self.close_button.setCursor(Qt.PointingHandCursor)
        self.close_button.clicked.connect(self.on_close_btn_clicked)

        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.title_label)
        self.layout.addStretch()
        self.layout.addWidget(self.close_button)

    def on_close_btn_clicked(self):
        self.parent_window.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.moving = True
            self.offset = event.globalPos() - self.parent_window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self.moving:
            self.parent_window.move(event.globalPos() - self.offset)

    def mouseReleaseEvent(self, event):
        self.moving = False
