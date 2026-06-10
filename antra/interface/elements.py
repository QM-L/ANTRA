from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QTextEdit
from PySide6.QtCore import Signal, Qt

'''Custom elements'''

class SectionFrame(QFrame):
    '''Basic frame'''

    def __init__(self, fill=True, border=True):
        super().__init__()
        if border: self.setFrameShape(QFrame.StyledPanel)
        self.content_layout = QVBoxLayout()

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addLayout(self.content_layout)
        if fill: self.main_layout.addStretch()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
    
    def add_space(self, spacing: int):
        self.content_layout.addSpacing(spacing)

class HeaderBar(QFrame):
    setup_clicked = Signal()
    select_clicked = Signal()
    advice_clicked = Signal()

    def __init__(self, version: str, parent=None):
        super().__init__(parent)
        self.setObjectName("appHeader")

        self.setup_btn = QPushButton("Setup")
        self.select_btn = QPushButton("Ablation Zone Placement")
        self.advice_btn = QPushButton("Needle Placement")

        self.select_btn.setDisabled(True)
        self.advice_btn.setDisabled(True)

        title = QLabel("ANTRA")
        title.setObjectName("appTitle")
        ver = QLabel(version, alignment=Qt.AlignmentFlag.AlignRight)
        ver.setObjectName("appVersion")

        layout = QHBoxLayout(self)
        layout.addWidget(title, 1)
        layout.addWidget(ver, 1)
        layout.addWidget(self.setup_btn, 2)
        layout.addWidget(self.select_btn, 2)
        layout.addWidget(self.advice_btn, 2)

        self.setup_btn.clicked.connect(self.setup_clicked)
        self.select_btn.clicked.connect(self.select_clicked)
        self.advice_btn.clicked.connect(self.advice_clicked)

    def unlock_select(self):
        self.select_btn.setEnabled(True)

    def unlock_advice(self):
        self.advice_btn.setEnabled(True)

class LogPanel(SectionFrame):
    def __init__(self):
        super().__init__(False, False)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        
        self.add_widget(self.console)
    
    def write(self, text):
        self.console.append(text)