import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation
from antra.needle_placing.raytracer import Raytracer, RayData
from antra.visualisation.visualizer import Visualizer
from antra.general.config import get_label_opacity_maps
from scipy.ndimage import map_coordinates

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

def graph_attenuation():
    '''Graphs attenuation from start of ray to end'''
    # setup objects
    dicom = DICOM_Scan("scans/DICOM_3D_IRCADB_01_8")
    seg = {
        task: Segmentation(task=task, folder="data/Segmentation_3D_IRCADB_01_8",load=True) for task in ["liver_vessels", "total","body"]
    }
    visualizer = Visualizer(dicom, seg)
    raytracer = Raytracer(1, seg)
    raytracer.set_origin(190.4, 128.3, 174.9)

    # setup plotter and ax
    plotter = pv.Plotter(shape=(2, 2), groups=[([0,1],0)], border_color='white')
    fig, ax = plt.subplots(figsize=(12, 4))
    fig2, ax2 = plt.subplots(figsize=(12, 4))

    arrow, chart, img = [None], [None], [None]
    def select_point(point):
        vector = np.asarray(point) - raytracer.origin
        vector = vector/np.linalg.norm(vector) # normalized
        ray_data = raytracer.analyze_path(vector)

        # arrow
        plotter.subplot(0,0)
        if arrow[0] is not None: plotter.remove_actor(arrow[0])
        arrow[0] = plotter.add_mesh(pv.Arrow(start=point+vector*5, direction=-vector,scale=ray_data.total_length + 5, shaft_radius=0.01,tip_radius=0.01),color='white')

        # replace chart
        plotter.subplot(0, 1)
        if chart[0] is not None: plotter.remove_chart(chart[0])
        update_ray_attenuation_chart(ray_data, ax)
        chart[0] = pv.ChartMPL(fig)
        plotter.add_chart(chart[0])

        # replace img
        plotter.subplot(1, 1)
        if img[0] is not None: plotter.remove_chart(img[0])
        update_dicom_chart(ray_data, raytracer.origin, visualizer.image, visualizer.seg, ax2)
        img[0] = pv.ChartMPL(fig2)
        plotter.add_chart(img[0])

    # show base scene for ray selection + enable picking on the sphere only
    visualizer.plot_segmentation(plotter, 'total')
    body_clickable = visualizer.plot_segmentation(plotter, 'body',cmap="Grays", opacity=0.2)
    visualizer.plot_segmentation(plotter, 'liver_vessels', cmap ='Blues', opacity=0.9)
    #body_clickable = self.visualize_body_scoring(plotter, raytracer, 500)
    plotter.enable_surface_point_picking(callback=select_point, show_message=False)
    plotter.pickable_actors = [body_clickable]
    plotter.show()
    fig.savefig("hd_fig.pdf",dpi=300, bbox_inches="tight")
    fig2.savefig("hd_fig2.pdf",dpi=300, bbox_inches="tight")

    plt.close(fig)

def update_ray_attenuation_chart(ray_data: RayData, ax: plt.Axes):
    ax.clear()

    segments = ray_data.ordered
    opacity_map = get_label_opacity_maps()

    distances = [0.0]
    scores = [1.0]

    d = 0.0
    current_score = 1.0

    for seg in segments:
        task   = seg["task"]
        label  = seg["tissue"]
        length = seg["length"]
        alpha  = seg["alpha"]

        if task == "body": continue

        raw_opacity = opacity_map[task][label] if label != 0 else 0.0
        opacity = raw_opacity * alpha

        # visualize traversed tissue
        ax.axvspan(d,d + length,alpha=min(opacity*0.3,1),color="blue",linewidth=0)

        # score contribution of this segment
        current_score *= (1 - opacity) ** length
        d += length
        distances.append(d)
        scores.append(current_score)

    score_line = ax.plot(distances, scores, color="red", lw=2,zorder=3, linestyle='dashed', label="Attenuation")
    tissue_patch = Patch(facecolor="blue", alpha=0.5, label="Tissue traversed")

    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, d)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Traversal Score")
    ax.legend(handles=[score_line[0], tissue_patch], frameon=True, loc="lower left")

def update_dicom_chart(ray_data: RayData, origin: np.ndarray, image, seg, ax: plt.Axes):
    ''' Close-to-z-slice CT volume along ray's phi angle, with labels.'''
    ax.clear()
    rx, ry, rz = image.resolution
    array = np.transpose(image.array, (2, 1, 0))  # (X, Y, Z)

    # basis vectors
    ray_vec = ray_data.vector / np.linalg.norm(ray_data.vector)
    xy_dir = ray_vec * np.array([1,1,0])
    xy_dir = xy_dir / np.linalg.norm(xy_dir)
    u = np.array([-xy_dir[1], xy_dir[0], 0.0])
    v = ray_vec

    # sample grid
    sx, sy, sz = array.shape
    extent = np.linalg.norm([sx*rx, sy*ry, sz*rz]) / 2

    voxel_min = min(rx, ry, rz)
    n_samples = int(extent * 2 / voxel_min)

    coords     = np.linspace(-extent, extent, n_samples)
    uu, vv     = np.meshgrid(coords, coords, indexing='ij')

    # convert coords
    world = (origin[:, None, None] + uu[None] * u[:, None, None] + vv[None] * v[:, None, None])
    voxel_coords = world / np.array([rx, ry, rz])[:, None, None]

    # ct
    ct_slice = map_coordinates(array, voxel_coords, order=1, mode='constant', cval=-1000)
    ax.imshow(ct_slice.T, cmap='gray', origin='lower', extent=[-extent, extent, -extent, extent], aspect='equal')

    # ray
    ray_u = np.dot(ray_vec, u)
    ray_v = np.dot(ray_vec, v)
    length = getattr(ray_data, 'total_length', extent)
    ax.annotate('', xy=(ray_u * (length+15), ray_v * (length+15)), xytext=(-15*ray_u, -15*ray_v), arrowprops=dict(arrowstyle='-', color='red', lw=0.2))

    ax.scatter(0, 0, color='red', s=3, zorder=5)  # origin marker
    ax.set_facecolor("#000000")