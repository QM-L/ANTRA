from pathlib import Path
import easygui
import pyvista as pv
import matplotlib
import matplotlib.pyplot as plt

from antra.segmentation.segmentator import DICOM_Scan, Segmentation
from antra.visualisation.visualizer import Visualizer
from antra.needle_placing.raytracer import Raytracer
from antra.general.config import load_configs

def run_raw_workflow():
    '''runs system from front to back without a gui, standard config. Only works for dicoms in /scans/'''
    # select a segmentation or dicom
    folder = easygui.diropenbox(msg="ct dicom / segmentation selection", default=Path(__file__).parent.parent.parent)
    conf = load_configs()

    # is dicom? segment
    if Path(folder).parent.name == "scans":
        dicom = DICOM_Scan(folder)
        segmentations = {task:Segmentation(dicom,task,folder) for task in ["liver_vessels","total","body"]}

    # is segmentation? load
    elif Path(folder).parent.name == "data":
        segmentations = {task:Segmentation(None,task,folder,True) for task in ["liver_vessels","total","body"]}
        dicom = segmentations["total"].dicom
    
    # else: error
    else: print(Path(folder).parent.name, "is not a valid segmentation or dicom"); return

    # select an ablation target
    vis = Visualizer(dicom,segmentations)
    fig, state = vis.build_slice_figure(True)
    plt.show()
    origin = (state['x'],state['y'],state['z'])

    # raytrace with default settings
    ray = Raytracer(conf.getfloat("needle","ablation_center_dist"),segmentations)
    ray.set_origin(*origin)
    results = ray.analyze_range(1000)

    # show scores
    plotter = pv.Plotter()
    vis.visualize_body_scoring(plotter, origin, results, conf.gettuple('scoring','weights'))
