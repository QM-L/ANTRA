from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from antra.segmentation.segmentator import DICOM_Scan, Segmentation
from antra.visualisation.visualizer import Visualizer
from antra.needle_placing.raytracer import Raytracer
from antra.general.config import load_configs
from antra.needle_placing.needle_advisor import NeedleAdvisor
from antra.validation.patch_visual import advise_debug
from antra.needle_placing.scoring import weigh_score

def default_raytrace(folder, origin):
    '''runs system from front to back without a gui, standard config.'''
    conf = load_configs()
    segmentations = {task:Segmentation(None,task,folder,True) for task in ["liver_vessels","total","body"]}
    dicom = segmentations["total"].dicom

    # show base scene
    plotter= pv.Plotter()
    vis = Visualizer(dicom, segmentations)
    vis.plot_segmentation(plotter, 'total', cmap='Grays', opacity=0.5)
    vis.plot_segmentation(plotter, 'body',  cmap='Grays', opacity=0.15)
    vis.plot_segmentation(plotter, 'liver_vessels', cmap='Reds', opacity=0.9)
    plotter.show()

    # raytrace with default settings
    ray = Raytracer(conf.getfloat("needle","ablation_center_dist"),segmentations)
    ray.set_origin(*origin)
    ray.set_phi_range(0,np.pi)
    ray.set_theta_range(0,2*np.pi)
    _,score_data = ray.analyze_range(2000)

    # advice
    weighted_scores = [{"theta": r["theta"],"phi": r["phi"],"weighted_score": weigh_score(r["scores"], [1, 1, 0.2, 0.5, 0.1])} for r in score_data]
    advisor = NeedleAdvisor(load_configs(), weighted_scores)
    data = advise_debug(advisor)

    # print stat and return
    max_score = np.max([score["weighted_score"] for score in weighted_scores])
    nonzero = sum(score["weighted_score"] > 0 for score in weighted_scores)
    threshold = sum(score["weighted_score"] > 0.5*max_score for score in weighted_scores)
    percentage_nonzero = 100 * nonzero / len(weighted_scores)
    percentage_threshold = 100 * threshold / len(weighted_scores)
    print(f"NONZERO: {percentage_nonzero}%\nABOVE THRESHOLD: {percentage_threshold}")
    return dicom, segmentations, data, score_data, conf

def plot_heatmap(folder, origin):
    dicom, segmentations, data, score_data, conf = default_raytrace(folder, origin)

    fig, ax = plt.subplots(1,1,figsize=(10,8), constrained_layout=True)
    theta_ticks = [0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi]
    theta_labels = ["0",r"$\frac{\pi}{2}$",r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    phi_ticks = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]
    phi_labels = ["0",r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]

    # A) interpolated score field
    im = ax.imshow(data["grid_scores"], origin="lower",cmap="YlOrRd", extent=data["extent"])
    fig.colorbar(im, ax=ax,location='right')

    # formatting
    ax.set_xticks(theta_ticks)
    ax.set_yticks(phi_ticks)
    ax.set_xticklabels(theta_labels)
    ax.set_yticklabels(phi_labels)
    ax.set_aspect('auto')
    
    fig.supxlabel(r"Azimuth $\theta$")
    fig.supylabel(r"Elevation $\phi$")

    plt.show()
    fig.savefig("hd_fig.png",dpi=300)

    plotter= pv.Plotter()
    vis = Visualizer(dicom, segmentations)
    vis.visualize_body_scoring(plotter,origin,score_data, conf.gettuple("scoring","weights"))
    plotter.show()
