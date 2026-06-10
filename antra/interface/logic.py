from pathlib import Path
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QThread

from antra.general.dicom_object import DICOM_Scan
from antra.interface.state import State
from antra.segmentation.segmentator import Segmentation
from antra.segmentation.worker import SegmentationWorker

''''Talks to backend and saves / accesses tool state.'''

class LogicHandler(QObject):
    # signals to the main window
    dicom_loaded = Signal()
    seg_loading = Signal()
    seg_loaded = Signal()
    
    def __init__(self, state: State):
        super().__init__()
        self.state = state
    
    def load_dicom(self, folder: Path):
        '''loads a dicom object into memory'''
        self.state.dicom = DICOM_Scan(folder)
        self.dicom_loaded.emit()

    def load_segmentation(self, folder: Path):
        '''loads a segmentation dict into memory from past data.'''
        self.state.segmentations = {
            "total": Segmentation(task="total", folder=folder),
            "liver_vessels": Segmentation(task="liver_vessels", folder=folder),
            "body": Segmentation(task="body", folder=folder),
        }
        self.seg_loaded.emit()
        print("succesfully loaded segmentation")
    
    def generate_segmentation(self):
        self.seg_loading.emit()
        now = datetime.now()
        folder = now.strftime("Segmentation_%Y-%m-%d_%H-%M-%S")

        self.thread = QThread()
        self.worker = SegmentationWorker(self.state.dicom, ["liver_vessels", "total", "body"], folder)
        self.worker.moveToThread(self.thread)

        # connectors
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_seg_done)
        self.worker.error.connect(lambda err: print(err))
        self.worker.notice.connect(lambda txt: print(txt))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
    
    def on_seg_done(self, segmentations):
        self.state.segmentations = segmentations
        self.seg_loaded.emit()
        print("successfully generated segmentation")