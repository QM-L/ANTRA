import sys
import easygui
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget
from PySide6.QtCore import Qt, QObject, Signal
from antra.interface.elements import SectionFrame, LogPanel, HeaderBar
from antra.interface.controls import ControlsPanel
from antra.interface.visuals import SetupPage
from antra.interface.logic import LogicHandler
from antra.interface.state import State

VERSION = "v0.0.1"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ablation Needle Trajectory Advisor")
        self.resize(1200, 700)

        self.state = State()
        self.logic = LogicHandler(self.state)

        self._build_ui()
        self._connect_signals()

        sys.stdout = EmittingStream(textWritten=self.logs.write)
        print("Successfully initialized.")

    def _build_ui(self):
        self.header = HeaderBar(VERSION)
        self.controls = ControlsPanel()
        self.logs = LogPanel()

        self.setup_page = SetupPage()
        self.select_page = SectionFrame()
        self.advice_page = SectionFrame()

        self.pages = QStackedWidget()
        self.pages.addWidget(self.setup_page)
        self.pages.addWidget(self.select_page)
        self.pages.addWidget(self.advice_page)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.controls, 4)
        left_layout.addWidget(self.logs, 1)

        body = QHBoxLayout()
        body.addWidget(left, 1)
        body.addWidget(self.pages, 3)

        root = QVBoxLayout()
        root.addWidget(self.header)
        root.addLayout(body)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _connect_signals(self):
        self.header.setup_clicked.connect(self.go_to_setup)
        self.header.select_clicked.connect(self.go_to_select)
        self.header.advice_clicked.connect(self.go_to_advice)

        # controls
        self.controls.load_seg_btn.clicked.connect(self.load_segmentation)
        self.controls.import_ct_btn.clicked.connect(self.load_ct_scan)
        self.controls.start_seg_btn.clicked.connect(self.start_seg)

        # logic handler signals
        self.logic.dicom_loaded.connect(lambda: self.controls.start_seg_btn.setEnabled(True))
        self.logic.seg_loaded.connect(lambda: self.header.select_btn.setEnabled(True))
        self.logic.seg_loading.connect(lambda: self.header.select_btn.setDisabled(True))

        # remove any plots on quit
        QApplication.instance().aboutToQuit.connect(self.setup_page.cleanup)

    def load_segmentation(self):
        folder = easygui.diropenbox(msg="ct scan / dicom selection", default="data/")
        self.logic.load_segmentation(folder)

    def load_ct_scan(self):
        folder = easygui.diropenbox(msg="ct scan / dicom selection", default="scans/")
        self.logic.load_dicom(folder)
    
    def start_seg(self):
        self.logic.generate_segmentation()

    def go_to_setup(self):
        self.pages.setCurrentWidget(self.setup_page)
        self.controls.setCurrentWidget(self.controls.controls_setup)
    def go_to_select(self):
        self.pages.setCurrentWidget(self.select_page)
        self.controls.setCurrentWidget(self.controls.controls_select)
    def go_to_advice(self):
        self.pages.setCurrentWidget(self.advice_page)
        self.controls.setCurrentWidget(self.controls.controls_advice)
        

class EmittingStream(QObject):
    '''Streams system output to the logs panel'''
    textWritten = Signal(str)
    def write(self, text):
        if text == "\n" or not text: return
        self.textWritten.emit(str(text).strip("\n"))
    
    def flush(self):
        pass

def load_stylesheet(app):
    with open(Path(__file__).parent / "style.qss", "r") as f:
        style = f.read()
        app.setStyleSheet(style)

def run_application():
    app = QApplication([])
    load_stylesheet(app)

    window = MainWindow()
    window.show()

    app.exec()