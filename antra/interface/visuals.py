from PySide6.QtWidgets import QLabel, QWidget, QHBoxLayout, QVBoxLayout
from pyvistaqt import QtInteractor
from antra.interface.elements import SectionFrame


class SetupPage(SectionFrame):
    '''Setup visuals are 3 frames: z frame dicom viewer, ablation zone viewer and segmentation viewer'''
    def __init__(self, parent=None):
        super().__init__(False, parent)

        # Container widget with horizontal layout
        container = QWidget()
        layout = QHBoxLayout(container)
        small_container = QWidget()
        small_layout = QVBoxLayout(small_container)

        # Create the three plotters
        self.dicom_plotter = self.make_plotter(container, layout)
        layout.addWidget(small_container)
        self.ablation_plotter = self.make_plotter(small_container, small_layout)
        self.seg_plotter = self.make_plotter(small_container, small_layout)

        self.add_widget(container)

    def make_plotter(self, parent: QWidget, layout: QHBoxLayout) -> QtInteractor:
        plotter = QtInteractor(parent)
        layout.addWidget(plotter)
        return plotter

    def cleanup(self):
        self.dicom_plotter.close()
        self.ablation_plotter.close()
        self.seg_plotter.close()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

