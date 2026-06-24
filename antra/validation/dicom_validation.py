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

def plot_heatmap(folder, origin):
    '''runs system from front to back without a gui, standard config. Only works for dicoms in /scans/'''
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
    else: print(Path(folder).name, "is not a valid segmentation or dicom"); return

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
    ray.set_theta_offset(0)
    _,score_data = ray.analyze_range(2000)

    # advice
    weighted_scores = [{"theta": r["theta"],"phi": r["phi"],"weighted_score": weigh_score(r["scores"], [1, 1, 0.2, 0.5, 0.1])} for r in score_data]
    advisor = NeedleAdvisor(load_configs(), weighted_scores)
    data = advise_debug(advisor)

    # plotting
    fig, ax = plt.subplots(1,1,figsize=(10,8), constrained_layout=True)
    theta_ticks = [0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi]
    #theta_ticks = [0.5*np.pi,np.pi,1.5*np.pi,2*np.pi,2.5*np.pi]
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