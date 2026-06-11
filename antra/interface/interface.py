import sys
import easygui
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget
from PySide6.QtCore import Qt, QObject, Signal
from antra.interface.elements import LogPanel, HeaderBar
from antra.interface.controls import ControlsPanel
from antra.interface.pages import SetupPage, SelectPage, AdvicePage
from antra.interface.logic import LogicHandler
from antra.general.state import State
from antra.visualisation.visualizer import Visualizer

VERSION = "v0.0.1"

class MainWindow(QMainWindow):
    def __init__(self, preload_seg=None):
        super().__init__()
        self.setWindowTitle("Ablation Needle Trajectory Advisor")
        self.resize(1200, 700)

        self.state = State()
        self.logic = LogicHandler(self.state)

        self._build_ui()
        self._connect_signals()

        sys.stdout = EmittingStream(textWritten=self.logs.write)
        print("Successfully initialized.")

        # debug seg shortcut
        if preload_seg: self.logic.load_segmentation(Path(__file__).parent.parent.parent / "data" / preload_seg)

    def _build_ui(self):
        self.header = HeaderBar(VERSION)
        self.controls = ControlsPanel()
        self.logs = LogPanel()

        self.setup_page = SetupPage()
        self.select_page = SelectPage()
        self.advice_page = AdvicePage()

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
        self.controls.confirm_tumor_btn.clicked.connect(self.confirm_tumor)
        self.select_page.dicom_widget.customContextMenuRequested.connect(self.update_info)

        # advice controls
        self.controls.start_raytracing_btn.clicked.connect(self.start_raytracing)
        self.controls.advice_combo.currentIndexChanged.connect(self.on_advice_selected)
        self.controls.scoring_toggle_btn.toggled.connect(self.advice_page.toggle_scoring_colors)

        # range sliders
        self.controls.theta_slider.changed.connect(self.update_range_preview)
        self.controls.phi_slider.changed.connect(self.update_range_preview)

        # logic handler signals
        self.logic.dicom_loaded.connect(self.on_dicom_loaded)
        self.logic.seg_loaded.connect(self.on_seg_loaded)
        self.logic.seg_loading.connect(lambda: self.header.select_btn.setDisabled(True))
        self.logic.raytracing_done.connect(self.on_raytracing_done)

    ### Signal Handlers

    def on_dicom_loaded(self):
        self.controls.start_seg_btn.setEnabled(True)
        # show DICOM viewer as soon as scan is loaded, before segmentation
        if self.state.dicom:
            vis = Visualizer(self.state.dicom, self.state.segmentations or {})
            self.state.visualizer = vis
            self.setup_page.show_dicom(vis)
 
    def on_seg_loaded(self):
        self.header.unlock_select()
        # rebuild Visualizer now that segmentations are available
        vis = Visualizer(self.state.dicom, self.state.segmentations)
        self.state.visualizer = vis
        self.setup_page.show_dicom(vis)
        self.setup_page.show_segmentation(vis)
        self.setup_page.show_ablation(vis)

    def on_advice_selected(self, index: int):
        if not self.state.advice: return
        self.advice_page.highlight_advice(index, self.state.advice, self.state.origin)

    def on_raytracing_done(self):
        self.header.unlock_advice()
        self.controls.populate_advice_combo(self.state.advice)
        self.advice_page.show_results(self.state.visualizer,self.state.raytracer,self.state.advice,self.controls.get_max_results())

    ### Actions

    def load_segmentation(self):
        folder = easygui.diropenbox(msg="ct scan / dicom selection", default="data/")
        self.logic.load_segmentation(folder)

    def load_ct_scan(self):
        folder = easygui.diropenbox(msg="ct scan / dicom selection", default="scans/")
        self.logic.load_dicom(folder)
    
    def start_seg(self):
        self.logic.generate_segmentation()
    
    def update_info(self):
        pos = self.select_page.dicom_widget.selected_voxel()
        data = self.state.tumor_analyzer.stats_for_origin(pos)
        self.controls.info_panel.update_data(data)
        return pos

    def confirm_tumor(self):
        pos = self.update_info()
        self.logic.confirm_ablation_center(pos)
        self.header.advice_btn.setEnabled(True)


    def start_raytracing(self):
        self.logic.run_raytracing( self.controls.get_theta_rad(), self.controls.get_phi_rad(), self.controls.get_density(), self.controls.get_max_results())

    def update_range_preview(self):
        '''update preview on advice page'''
        self.state.raytracer.set_theta_range(*self.controls.get_theta_rad())
        self.state.raytracer.set_phi_range(*self.controls.get_phi_rad())
        self.advice_page.update_range_preview(self.state.raytracer, self.controls.get_density())

    def go_to_setup(self):
        self.pages.setCurrentWidget(self.setup_page)
        self.controls.setCurrentWidget(self.controls.controls_setup)
        
        # clean other pages
        self.advice_page.cleanup()
        self.select_page.cleanup()
        self.state.page = "setup"

    def go_to_select(self):
        self.pages.setCurrentWidget(self.select_page)
        self.controls.setCurrentWidget(self.controls.controls_select)
        self.select_page.show_dicom(self.state.visualizer)
        self.logic.init_tumor_analyzer()

        # clean other pages
        self.advice_page.cleanup()
        self.setup_page.cleanup()
        self.state.page = "select"

    def go_to_advice(self):
        # set up advice page with base scene and range preview on first visit
        print("Setting up advice page, this can take a few seconds...")
        self.logic.init_raytracer()
        self.advice_page.setup(self.state.visualizer)
        self.advice_page.update_range_preview(self.state.raytracer, self.controls.get_density())
        self.pages.setCurrentWidget(self.advice_page)
        self.controls.setCurrentWidget(self.controls.controls_advice)
        
        # clean other pages
        self.select_page.cleanup()
        self.setup_page.cleanup()
        self.state.page = "advice"
    
    def closeEvent(self, event):
        '''Override closing: make sure all plots close first.'''
        self.setup_page.cleanup()
        self.select_page.cleanup()
        super().closeEvent(event)
        

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

def run_application(seg=None):
    app = QApplication([])
    load_stylesheet(app)

    window = MainWindow(seg)
    window.show()
    app.exec()