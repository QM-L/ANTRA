import copy
from pathlib import Path
from datetime import datetime
import numpy as np

from PySide6.QtCore import QObject, Signal, QThread

from antra.general.dicom_object import DICOM_Scan
from antra.general.state import State
from antra.needle_placing.scoring import TumorAnalyzer, weigh_score
from antra.needle_placing.raytracer import Raytracer
from antra.needle_placing.needle_advisor import NeedleAdvisor
from antra.segmentation.segmentator import Segmentation
from antra.segmentation.worker import SegmentationWorker

''''Talks to backend and saves / accesses tool state.'''

class LogicHandler(QObject):
    # signals to the main window
    dicom_loaded = Signal()
    seg_loading = Signal()
    seg_loaded = Signal()
    raytracing_done = Signal()
    
    def __init__(self, state: State):
        super().__init__()
        self.state = state
    
    def load_dicom(self, folder: Path):
        '''loads a dicom object into memory'''
        self.state.dicom = DICOM_Scan(folder)
        self.dicom_loaded.emit()

    def load_segmentation(self, folder: Path):
        '''loads a segmentation dict into memory from past data.'''
        dicom = None
        segmentations = {}
        for task in ["liver_vessels","total","body"]:
            seg = Segmentation(dicom=dicom, task=task, folder=folder, load=True)
            segmentations[task] = seg
            dicom = dicom or seg.dicom

        self.state.segmentations = segmentations
        self.state.dicom = dicom
        self.dicom_loaded.emit()
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
    
    def init_tumor_analyzer(self):
        self.state.tumor_analyzer = TumorAnalyzer(self.state.segmentations)
    
    def confirm_ablation_center(self, pos):
        pos_mm = self.state.tumor_analyzer.stats_for_origin(pos)["current pos (mm)"]
        self.state.origin = pos_mm
        print(f"Set ablation center at {pos_mm}")
    
    def init_raytracer(self, theta, phi):
        ablation_dist = self.state.config.getint('needle', 'ablation_center_dist')
        rt = Raytracer(ablation_dist, self.state.segmentations)
        rt.set_theta_range(*theta)
        rt.set_phi_range(*phi)
        rt.set_origin(*self.state.origin)
        self.state.raytracer = rt

    def run_raytracing(self, theta_range: tuple[float, float], phi_range: tuple[float, float], density: int, max_results: int) -> None:
        '''Runs raytracing and NeedleAdvisor, stores results in state.'''
        # ablation_center_dist: how far the needle tip extends past the origin
        ablation_dist = self.state.config.getint('needle', 'ablation_center_dist')

        raytracer = Raytracer(ablation_dist, self.state.segmentations)
        raytracer.set_origin(*self.state.origin)
        raytracer.set_theta_range(*theta_range)
        raytracer.set_phi_range(*phi_range)

        print(f"Raytracing {density} rays/srad² over θ={np.degrees(theta_range)}° and φ={np.degrees(phi_range)}°...")

        rays, results = raytracer.analyze_range(density)
        self.state.score_data = results
        
        self.raytracing_done.emit()
    
    def store_weighted_scores(self):
        '''stores weighted scores to state'''
        weighted_scores = copy.deepcopy(self.state.score_data)
        for datapoint in weighted_scores:
            datapoint["weighted_score"] = weigh_score(datapoint["scores"], self.state.weights)
        self.state.weighted_scores = weighted_scores
    
    def get_needle_advice(self):
        advice = NeedleAdvisor(self.state.config, self.state.weighted_scores).advise()
        print(f"Found {len(advice)} viable path(s)")
        return advice