import argparse
import numpy as np
from antra.general import config
from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation
from antra.visualisation.visualizer import Visualizer
from antra.needle_placing.raytracer import Raytracer
from antra.visualisation import demos

DEFAULT_DICOM = 'PATIENT_DICOM'

def load_objects(dicom_path):
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
    #demos.demo_circle(vis, ray)
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
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dicom", type=str, default=DEFAULT_DICOM, required=False)
    args = parser.parse_args()
    dicom_path = f'scans/{args.dicom}'

    # load
    img, seg, vis, ray = load_objects(dicom_path)
    ray.set_origin(339*img.resolution[0], 228*img.resolution[1], 109*img.resolution[2]) # default for dicom 2
    
    # demos
    demo_tumor_selection(img, vis, ray)
    print('raytracing...')
    demo_raytrace(vis, ray)
    