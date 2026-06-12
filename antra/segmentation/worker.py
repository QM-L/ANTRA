from PySide6.QtCore import QObject, Signal
from antra.segmentation.segmentator import Segmentation

'''This is a worker class which runs the segmentation creation, so that the main thread isn't overloaded.'''

class SegmentationWorker(QObject):
    finished = Signal(dict)
    error    = Signal(str)
    notice   = Signal(str)

    def __init__(self, dicom, tasks, folder):
        super().__init__()
        self.dicom, self.tasks, self.folder = dicom, tasks, folder

    def run(self):
        try:
            result = {}
            for task in self.tasks:
            #for task in self.tasks:
                self.notice.emit(f"trying to create seg [{task}] into {self.folder}")
                result[task] = Segmentation(dicom=self.dicom, task=task, folder=self.folder)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))