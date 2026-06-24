import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from superqt import QRangeSlider
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QTextEdit, QSlider
from PySide6.QtCore import Signal, Qt

from antra.visualisation.visualizer import Visualizer
from antra.needle_placing.scoring import SCORER_NAMES

'''Custom elements'''

class SectionFrame(QFrame):
    '''Basic frame'''

    def __init__(self, fill=True, border=True):
        super().__init__()
        self.app = QApplication.instance()
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
        self.app = QApplication.instance()
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

        # Connections
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
        self.app = QApplication.instance()
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        
        self.add_widget(self.console)
    
    def write(self, text):
        self.console.append(text)
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.console.setTextCursor(cursor)

        # scroll



class DicomSliceWidget(QWidget):
    '''DICOM viewer, update(visualizer) populates it after data is loaded.'''

    def __init__(self, selection=False, parent=None):
        super().__init__(parent)
        self.app = QApplication.instance()
        self.select_active = selection
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self.state  = None

    def update(self, visualizer: Visualizer) -> None:
        '''Build or rebuild the slice figure from a loaded Visualizer.'''
        # remove old canvas if present
        if self.canvas is not None:
            self.box.removeWidget(self.canvas)
            self.canvas.setParent(self)

        fig, self.state = visualizer.build_slice_figure(self.select_active)

        self.canvas = FigureCanvasQTAgg(fig)
        self.box.addWidget(self.canvas)
    
    def toggle_segmentations(self):
        '''Toggles whether segmentations are visible on the plot'''
        # TODO: this

    def selected_voxel(self) -> tuple[int, int, int] | None:
        '''Returns the currently selected (x, y, z) voxel, or None if not loaded.'''
        if self.state is None: return None
        return self.state['x'], self.state['y'], self.state['z']

class ValueRow(QWidget):
    '''singular status label.'''
    def __init__(self, key: str, initial_value: str = "", parent=None):
        super().__init__(parent)
        self.app = QApplication.instance()

        # Main horizontal layout
        layout = QHBoxLayout(self)

        # Key Label (Styled to be muted/secondary)
        self.key_label = QLabel(key)
        self.key_label.setStyleSheet("font-weight: bold;")

        # Value Label (Styled to stand out)
        self.value_label = QLabel(str(initial_value))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.key_label)
        layout.addStretch()
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))

class ValuePanel(QFrame):
    ''''Column of status labels that can be updated with a dict.'''
    def __init__(self, title: str =None, parent=None):
        super().__init__(parent)
        self.app = QApplication.instance()
        self.rows = {}

        self.main_layout = QVBoxLayout(self)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: bold;")
        self.main_layout.addWidget(self.title)
    
    def update_data(self, data: dict):
        '''update data and rows'''
        for key, value in data.items():
            # update already existing data
            if key in self.rows:
                self.rows[key].set_value(value)
            # append new one
            else:
                row_widget = ValueRow(key, value, self)
                self.main_layout.addWidget(row_widget)
                self.rows[key] = row_widget

class ValueSlider(QWidget):
    ''''Slider with a label next to it that shows the value.'''
    def __init__(self, init_min: int, init_max: int, init_value: int, suffix: str = "", label=None):
        super().__init__()
        self.app = QApplication.instance()

        # widgets
        self.name = QLabel(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(init_min)
        self.slider.setMaximum(init_max)
        self.slider.setValue(init_value)
        self.slider.setTickInterval(50)
        self.label = QLabel(str(init_value)+suffix)

        self.slider.valueChanged.connect(lambda v: self.label.setText(str(v)+suffix))

        # arrange
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        if label: layout.addWidget(self.name)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)

class RangeSlider(QWidget):
    '''the sliders that change the range of the raycasting'''
    changed = Signal()

    def __init__(self, init_min: int, init_max: int, init_value: tuple[int, int], suffix: str = ""):
        super().__init__()
        self.app = QApplication.instance()
        self.suffix = suffix

        # widgets
        self.slider = QRangeSlider(Qt.Horizontal)
        self.slider.setRange(init_min, init_max)
        self.slider.setValue(init_value)
        self.slider.setTickInterval(50)
        self.label = QLabel(f"{init_value[0]}-{init_value[1]}{self.suffix}")

        self.slider.valueChanged.connect(self.on_changed)

        # arrange
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def on_changed(self, val: tuple[int,int]):
        self.label.setText(f"{val[0]}-{val[1]}{self.suffix}")
        self.changed.emit()

    def get_radians(self) -> tuple[float, float]:
        range_deg = self.slider.value()
        return np.radians(range_deg[0]), np.radians(range_deg[1])

class WeightsPanel(QFrame):
    '''allows user to modify scoring weights'''
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.app = QApplication.instance()

        self.main_layout = QVBoxLayout(self)
        self.title = QLabel("Scoring Weights")
        self.title.setStyleSheet("font-weight: bold;")
        self.main_layout.addWidget(self.title)

        self.sliders = {}

        for name in SCORER_NAMES:
            slider = ValueSlider(init_min=0,init_max=300,init_value=100,suffix="%", label=name)
            self.main_layout.addWidget(slider)
            self.sliders[name] = slider
            slider.slider.valueChanged.connect(lambda x: self.changed.emit())

    def get_weights(self) -> list[float]:
        '''returns weights as floats'''
        return [self.sliders[scorer].slider.value() / 100 for scorer in SCORER_NAMES]

    def set_weights(self, weights: list[float]) -> None:
        '''load weights from config/state.'''
        for name, weight in zip(SCORER_NAMES, weights):
            self.sliders[name].slider.setValue(int(weight * 100))