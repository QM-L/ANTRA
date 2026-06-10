import numpy as np
from antra.general import config
from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation
from antra.visualisation.visualizer import Visualizer
from antra.needle_placing.raytracer import Raytracer
from antra.visualisation import demos
from antra.interface.window import run_application
from antra.interface.logic import LogicHandler

DEFAULT_DICOM = 'PATIENT_DICOM_1'
SEG_MODE = False

def load_objects(dicom_path) -> tuple[DICOM_Scan, Segmentation, Visualizer, Raytracer]:
    # load config
    config_data = config.load_configs()

    # init dicom, segmentations, visualizer
    print('loading dicom...')
    dicom = DICOM_Scan(dicom_path)
    print('generating/loading segmentations...')
    segmentations = {task: Segmentation(dicom, task) for task in ['liver_vessels', 'total', 'body']} # earlier get priority
    visualizer = Visualizer(dicom, segmentations)

    # init raytracer
    dx,dy,dz = dicom.resolution
    raytracer = Raytracer(config_data.getfloat('needle','ablation_center_dist'), segmentations)
    return dicom, segmentations, visualizer, raytracer

def demo_segmentation(vis):
    # generate segmentations & visualize them
    demos.demo_3D_visualizer(vis)

def demo_raytrace(vis, ray):
    # raycast high amount of angles test
    ray.set_theta_range(0, 2*np.pi)
    ray.set_phi_range(0.2*np.pi, 0.8*np.pi)
    demos.demo_rendertime(vis, ray)

def demo_circle(vis, ray):
    # raycast high amount of angles and show each as a thin line
    ray.set_theta_range(0, 2*np.pi)
    ray.set_phi_range(0, np.pi)
    demos.demo_circle(vis, ray)

def demo_tumor_selection(img, vis, ray):
    origin = vis.tumor_selector(None)
    dx,dy,dz = img.resolution
    x, y, z = origin[0]*dx, origin[1]*dy, origin[2]*dz
    ray.set_origin(x, y, z)

if __name__ == "__main__":
    if SEG_MODE:
        print("scanning dicom...")
        dicom = DICOM_Scan("scans/"+DEFAULT_DICOM)
        folder = "Segmentation_A"
        print("starting segmentations")
        segmentations = {task: Segmentation(dicom, task, folder) for task in ['liver_vessels', 'total', 'body']}
        print("done")
    
    run_application()