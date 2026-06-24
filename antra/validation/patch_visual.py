import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops

from antra.general.config import load_configs
from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation
from antra.needle_placing.raytracer import Raytracer
from antra.needle_placing.scoring import weigh_score
from antra.needle_placing.needle_advisor import NeedleAdvisor

## Plt font
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def score_heatmap():
    '''graphs final (unweighted) score on 2D heatmaps and displays:
        a) full trajectory space
        b) thresholded viable patches
        c) distance transform
        d) centers marked'''

    # setup objects
    dicom = DICOM_Scan("scans/DICOM_3D_IRCADB_01_8")
    seg = {task: Segmentation(dicom=dicom,task=task, folder="data/Segmentation_3D_IRCADB_01_8",load=True) for task in ["liver_vessels", "total","body"]}
    raytracer = Raytracer(1, seg)
    origin = (190.4, 128.3, 174.9)
    raytracer.set_origin(*origin)

    # raytrace & score
    _,score_data = raytracer.analyze_range(1000)
    weighted_scores = [{"theta": r["theta"],"phi": r["phi"],"weighted_score": weigh_score(r["scores"], [1, 1, 0.2, 0.5, 0.1])} for r in score_data]
    advisor = NeedleAdvisor(load_configs(), weighted_scores)
    data = advise_debug(advisor)

    # plotting
    fig, ax = plt.subplots(2,2,figsize=(10,8), constrained_layout=True)
    theta_ticks = [0.5*np.pi,np.pi,1.5*np.pi,2*np.pi,2.5*np.pi]
    theta_labels = ["0",r"$\frac{\pi}{2}$",r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    phi_ticks = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]
    phi_labels = ["0",r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]

    # A) interpolated score field
    im0 = ax[0,0].imshow(data["grid_scores"], origin="lower",cmap="YlOrRd", extent=data["extent"])
    fig.colorbar(im0, ax=ax[0,:],location='right')

    # B) thresholded viable region
    ax[0,1].imshow(data["binary"],origin="lower", cmap="gray", extent=data["extent"])

    # C) distance transform
    im2 = ax[1,0].imshow(data["distance"],origin="lower",cmap="inferno", extent=data["extent"])
    fig.colorbar(im2, ax=ax[1,:],location='right')

    # D - accepted patches and centers
    ax[1,1].imshow(data["binary"], origin="lower", cmap="gray", extent=data["extent"])
    for col, row in data["centers"]:
        ax[1,1].scatter(col,row,color="red",marker="x", s=100)

    # formatting
    for a in ax.flat:
        a.set_xticks(theta_ticks)
        a.set_yticks(phi_ticks)
        a.set_xticklabels(theta_labels)
        a.set_yticklabels(phi_labels)
        a.set_aspect('auto')
    for a in ax[0,:]:
        a.tick_params(labelbottom=False)
    for a in ax[:,1]:
        a.tick_params(labelleft=False)
    
    fig.supxlabel(r"Azimuth $\theta$")
    fig.supylabel(r"Elevation $\phi$")

    plt.show()
    fig.savefig("hd_fig3.png",dpi=300)

def advise_debug(advisor: NeedleAdvisor) -> list[dict]:
    '''Simplified version of advisor.advise() which returns data to plot.'''
    # generate grid with the weighted scores
    t = np.array([r['theta'] for r in advisor.results])
    p = np.array([r['phi']   for r in advisor.results])
    total  = np.array([r['weighted_score'] for r in advisor.results])

    # score a finer coordinate space with interpolation
    res    = 500
    thetas = np.linspace(t.min(), t.max(), res)
    phis   = np.linspace(p.min(), p.max(), res)
    mesh_theta, mesh_phi = np.meshgrid(thetas, phis)
    grid_scores = griddata(points=np.column_stack([t, p]), values=total, xi=(mesh_theta,mesh_phi), method='linear', fill_value=0)

    # calculate step sizes
    theta_step = (t.max()-t.min())/(res-1)
    phi_step   = (p.max()-p.min())/(res-1)

    # find patches with the threshold
    threshold_value = advisor.threshold * grid_scores.max()
    binary = grid_scores >= threshold_value
    labelled,_ = label(binary)
    centers = []

    # go through each patch
    for region in regionprops(labelled):
        # continue if region is too small, else, find center of region
        if region.num_pixels < advisor.min_patch_size: continue
        _,center = advisor.get_region_center_data(region, labelled, theta_step, phi_step)
        centers.append((thetas[center[1]],phis[center[0]]))

    return {
        "grid_scores": grid_scores,
        "binary": binary,
        "distance": distance_transform_edt(binary),
        "labelled": labelled,
        "centers": centers,
        "extent": [t.min(),t.max(),p.min(),p.max()]
    }