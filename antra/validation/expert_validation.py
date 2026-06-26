import numpy as np
from pathlib import Path
import pyvista as pv

from antra.visualisation.visualizer import Visualizer
from antra.validation.dicom_validation import default_raytrace
from antra.needle_placing.needle_advisor import NeedleAdvisor
from antra.needle_placing.scoring import weigh_score

CASE_08_PATH = "data/Segmentation_3D_IRCADB_01_8"
CASE_18_PATH = "data/Segmentation_3D_IRCADB_01_18"

CASE_08_EXPERT_PATH = {"entry":[167.3,63.5, 174.4],"target":[191.4,128.7,174.4]}
CASE_18_EXPERT_PATH = {"entry":[128.2,99.3,98.5],"target":[99.5,145.2,92.5]}

def plot_expert_path_over_scoring():

    for case_path, expert_path in [(CASE_08_PATH, CASE_08_EXPERT_PATH),(CASE_18_PATH, CASE_18_EXPERT_PATH)]:
        folder = Path(__file__).parent.parent.parent / case_path
        origin = expert_path["target"]
        dicom, segmentations, data, score_data, conf = default_raytrace(folder,origin)
        
        # expert trajectory
        expert_dir = unit(np.array(expert_path["entry"]) - np.array(expert_path["target"]))
        theta, phi = direction_to_angles(expert_dir)

        # weighted scores
        weights = conf.gettuple("scoring", "weights")
        weighted_scores = [{"theta": score["theta"],"phi": score["phi"],"weighted_score": weigh_score(score["scores"], weights)} for score in score_data]
        
        # expert score
        expert_ray = nearest_score(theta, phi, score_data)
        expert_score = weigh_score(expert_ray["scores"], weights)
        all_scores = np.array([s["weighted_score"] for s in weighted_scores])
        percentile = (np.sum(all_scores <= expert_score)/len(all_scores)*100)

        # advice generation
        advisor = NeedleAdvisor(conf, weighted_scores)
        advice = advisor.advise()

        # visualization
        plotter = pv.Plotter()
        vis = Visualizer(dicom, segmentations)

        vis.visualize_body_scoring(plotter, np.array(origin), score_data, weights)

        # expert trajectory
        ray_length = 120.0
        plotter.add_mesh(pv.Arrow(start=origin, direction=expert_dir, scale=ray_length,shaft_radius=0.008, tip_radius=0.025), color="cyan", name="expert")

        # top advice arrows
        for i, adv in enumerate(advice):
            t, p = adv['theta'], adv['phi']
            d = np.array([np.sin(p) * np.cos(t),np.sin(p) * np.sin(t), np.cos(p)])
            arrow = pv.Arrow(start=origin, direction=d, scale=ray_length,shaft_radius=0.008, tip_radius=0.025)
            plotter.add_mesh(arrow, color='yellow', name=f'advice_{i}')

        plotter.add_mesh(pv.PolyData([origin]), color="red", point_size=15, render_points_as_spheres=True)

        print(folder.name, f"Expert score: {expert_score:.3f}\nExpert percentile: {percentile:.1f}%\n Advice patches: {len(advice)}")
        plotter.show()

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

def direction_to_angles(d):
    x, y, z = d
    theta = np.mod(np.arctan2(y, x), 2*np.pi)
    phi = np.arccos(np.clip(z, -1, 1))
    return theta, phi

def nearest_score(theta, phi, score_data):
    idx = np.argmin([(theta - s["theta"])**2 +(phi - s["phi"])**2for s in score_data])
    return score_data[idx]