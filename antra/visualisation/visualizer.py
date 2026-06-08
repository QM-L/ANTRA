import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import map_coordinates

from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation
from antra.needle_placing.scoring import *
from antra.needle_placing.raytracer import Raytracer
from antra.visualisation.tumor_selection import MplContrastHelper
from antra.general.config import get_label_opacity_maps, load_configs

class Visualizer():
    '''Arbitrary class to show results, needs to be replaced with a class that
       properly interacts with PySide6 to show plots.'''
    def __init__(self, image: DICOM_Scan, segmentations: dict[str,Segmentation]) -> None:
        self.image  = image
        self.config = load_configs()
        self.seg    = segmentations
        self.array  = np.transpose(self.image.array, (2, 1, 0))
        self.volume = self.get_3D_volume()

    def get_3D_volume(self):
        volume = pv.ImageData()
        volume.dimensions = self.array.shape
        volume.spacing = self.image.resolution
        volume.point_data["CT"] = self.array.flatten(order="F")
        volume = volume.resample(0.3,interpolation="nearest")
        return volume

    def plot_base_scene(self, plotter: pv.Plotter):
        ''''Plots the body and it's insides.'''
        self.plot_segmentation(plotter, 'total')
        self.plot_segmentation(plotter, 'body',cmap="Grays", opacity=0.2)
        self.plot_segmentation(plotter, 'liver_vessels', cmap ='Blues', opacity=0.9)

    def visualize_angles(self, plotter: pv.Plotter, raycaster: Raytracer, radius=100):
        ''''Visualize a list of raydata as arrows a sphere'''
        directions = raycaster.generate_n_directions(3000)
        origin     = raycaster.origin
        points     = directions * radius + origin
        cloud      = pv.PolyData(points)
        cloud["vectors"] = directions
        arrows     = cloud.glyph(orient="vectors", scale=False, factor=radius * 0.1)

        plotter.add_mesh(arrows, color="red")
        plotter.add_mesh(pv.PolyData([origin]), point_size=12,
                        render_points_as_spheres=True, color='r')     

    def plot_segmentation(self, plotter: pv.Plotter, seg_task='total', cmap='Reds', manual_array: np.ndarray = None, opacity: float = 0.95):
        '''Plots specified segmentation'''
        seg = self.seg.get(seg_task)
        if not seg and not manual_array: return
        array = manual_array or seg.get_array(np.float32)

        # define the masks's 3d volume
        volume = pv.ImageData()
        volume.dimensions = np.array(array.shape) + 1
        volume.spacing    = np.sqrt((seg.raw_mask.affine[:3, :3] ** 2).sum(axis=0)).tolist()
        volume.origin     = self.image.origin
        volume.cell_data["label"] = array.flatten(order="F")

        # Threshold out background (label 0)
        segmented = volume.threshold(value=0.5, scalars="label")

        # define custom colormap where liver is blue
        if seg_task == "total":
            n = int(np.floor(array).max()) + 1
            base   = plt.get_cmap(cmap, n)
            colors = [base(i / n) for i in range(n)]
            colors[5] = mcolors.to_rgba('yellow')
            cmap = mcolors.ListedColormap(colors)

        # plot
        actor = plotter.add_mesh(segmented, scalars="label", log_scale=True, cmap=cmap, opacity=opacity,show_scalar_bar=False)
        return actor

    def visualize_body_scoring(self, plotter: pv.Plotter, raytracer: Raytracer, n: int = 250) -> pv.Actor:
        '''Color the body surface mesh by ray scores from the origin.'''
        # body mask to vertex mesh
        array   = self.seg['body'].get_array(np.uint16)
        vol = pv.ImageData()
        vol.dimensions = np.array(array.shape) + 1
        vol.spacing    = np.sqrt((self.seg['body'].mask.affine[:3, :3] ** 2).sum(axis=0)).tolist()
        vol.origin     = self.image.origin
        vol.cell_data["label"] = array.flatten(order="F")
        body_mesh = vol.threshold(0.5).extract_surface(algorithm='dataset_surface').smooth(n_iter=100, relaxation_factor=0.05)

        # raycast
        directions, accumulators = raytracer.analyze_range(n)
        lengths = np.asarray([acc.total_length - raytracer.ablation_dist for acc in accumulators]).reshape(-1, 1)
        points  = directions * lengths + raytracer.origin
        scorer = Scorer(self.seg, raytracer.origin)
        scores  = [scorer.get_score(accumulator) for accumulator in accumulators]

        # Build a point cloud with score scalars
        cloud = pv.PolyData(points)
        cloud["score"] = np.array(scores)
        body_mesh = body_mesh.interpolate(cloud, radius=5)
                
        # plot
        plotter.add_mesh(cloud, scalars="score", log_scale=False,color="red", name="cloud",cmap="Oranges", render_points_as_spheres=True,point_size=5)
        return plotter.add_mesh(body_mesh, scalars="score", cmap="Oranges", log_scale=False, show_scalar_bar=True)
        
    
    def segmentation_margin_view(self, plotter: pv.Plotter):
        '''Browse organ labels live, showing tissue and margin separately.'''
        array  = self.seg['total'].get_array(np.float32)
        affine = self.seg['total'].mask.affine

        # define the masks's 3d volume
        vol = pv.ImageData()
        vol.dimensions = np.array(array.shape) + 1
        vol.spacing    = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist()
        vol.origin     = (affine[:3, 3] - np.array(self.image.origin)).tolist()
        vol.cell_data["label"] = array.flatten(order="F")

        labels = sorted(int(v) for v in np.unique(np.floor(array)) if v > 0)
        actors = {"tissue": None, "margin": None, "text": None}
        show_margin  = {'state': True}

        def show_label(label: int):
            # create meshes for tissue and margins seperately
            tissue = vol.threshold([label-0.01, label + 0.01], scalars="label").extract_surface()
            margin = vol.threshold([label, label + 0.99], scalars="label").extract_surface()
 
            # remove existing actors and create new ones
            (plotter.remove_actor(actors[key]) for key in actors if actors[key])
            actors["tissue"] = plotter.add_mesh(tissue, color="orange", opacity=0.85,show_scalar_bar=False, name="tissue")
            actors["margin"] = plotter.add_mesh(margin, color="yellow", opacity=0.4,show_scalar_bar=False, name="margin") if show_margin['state'] else None
            actors["text"] = plotter.add_text(f"Label {label}", position="upper_left",font_size=14, color="white")

        def on_slider(value: float):
            # snap float slider value to nearest valid label
            label = min(labels, key=lambda l: abs(l - value))
            show_label(label)

        def on_toggle(state: bool):
            show_margin['state'] = state

        # plot
        plotter.add_volume(self.volume, scalars="CT", cmap="bone", opacity=[0,0,0,0,0,0,0.01,0.01,0.01,0.01,0.01,0.01])
        plotter.add_slider_widget(on_slider,rng=[labels[0], labels[-1]],value=labels[0], title="Label", pointa=(0.1, 0.06), pointb=(0.9, 0.06))
        plotter.add_checkbox_button_widget(on_toggle,value=True,position=(10, 10),size=30,color_on="yellow",color_off="grey")
        plotter.add_axes()
        plotter.show()

    def tumor_selector(self, plotter: pv.Plotter) -> tuple[int]:
        '''Matplotlib tumor selector'''
        fig = plt.figure(layout='constrained')
        axes = fig.subplot_mosaic("aab\naac")
        rx, ry, rz = self.image.resolution
        state = {"x": self.array.shape[0] // 2,"y": self.array.shape[1] // 2,"z": self.array.shape[2] // 2, "show_seg": True}

        # intial image initialisation
        x, y, z = state["x"], state["y"], state["z"]

        img_a = axes['a'].imshow(self.array[:,:,z].T, cmap="gray", origin="lower", aspect=ry/rx)
        img_b = axes['b'].imshow(self.array[:,y,:].T, cmap="gray", origin="lower", aspect=rz/rx)
        img_c = axes['c'].imshow(self.array[x,:,:].T, cmap="gray", origin="lower", aspect=rz/ry)

        line_ah = axes['a'].axhline(y, color="red", lw=0.8, alpha=0.3)
        line_bh = axes['b'].axhline(z, color="red", lw=0.8, alpha=0.3)
        line_ch = axes['c'].axhline(z, color="red", lw=0.8, alpha=0.3)

        line_av = axes['a'].axvline(x, color="red", lw=0.8, alpha=0.3)
        line_bv = axes['b'].axvline(x, color="red", lw=0.8, alpha=0.3)
        line_cv = axes['c'].axvline(y, color="red", lw=0.8, alpha=0.3)

        # get segmentations
        tumor_array = np.asarray(self.seg['liver_vessels'].raw_mask.dataobj)
        liver_array   = np.asarray(self.seg['total'].raw_mask.dataobj)
        contours = {"tumor": [], "liver": []}

        axes['a'].invert_yaxis()

        def refresh():
            # change data in the images based on new xyz values
            x, y, z = state["x"], state["y"], state["z"]
            change_images(x,y,z)
            change_lines(x,y,z)
            change_contours(z)
            fig.canvas.draw_idle()
        
        def change_images(x,y,z):
            img_a.set_data(self.array[:,:,z].T)
            img_b.set_data(self.array[:,y,:].T)
            img_c.set_data(self.array[x,:,:].T)

        def change_lines(x,y,z):
            line_ah.set_ydata([y, y]); line_av.set_xdata([x, x])
            line_bh.set_ydata([z, z]); line_bv.set_xdata([x, x])
            line_ch.set_ydata([z, z]); line_cv.set_xdata([y, y])

        def change_contours(z):
            # remove old contours
            for key in contours:
                for cs in contours[key]:
                    cs.remove()
                contours[key] = []
            
            if not state['show_seg']: return

            tumor_slice = (tumor_array[:, :, z] == 2).T
            liver_slice   = (liver_array[:, :, z]   == 5).T

            # only draw if the label is present in this slice
            if tumor_slice.any():
                cs = axes['a'].contour(tumor_slice, levels=[0.5], colors=["cyan"], linewidths=0.8)
                contours["tumor"] = [cs]

            if liver_slice.any():
                cs = axes['a'].contour(liver_slice,   levels=[0.5], colors=["yellow"], linewidths=0.8)
                contours["liver"]   = [cs]

        def on_click(event):
            if event.button == 1: return # left click is dragging
            if event.inaxes != axes['a']: return
            state["x"], state["y"] = int(event.xdata), int(event.ydata)
            refresh()

        fig.canvas.mpl_connect("button_press_event", on_click)

        # scroll to change slice
        def on_scroll(event):
            delta = 1 if event.button == "up" else -1
            if event.inaxes != axes['a']: return
            state["z"] = np.clip(state["z"] + delta, 0, self.array.shape[2]-1)
            refresh()

        fig.canvas.mpl_connect("scroll_event", on_scroll)

        def on_key(event):
            # toggle contours
            if event.key != "t": return
            state["show_seg"] = not state["show_seg"]
            change_contours(state["z"])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)

        # connect HU handling functions to the chart objects
        HU_handlers = MplContrastHelper(img_a), MplContrastHelper(img_b), MplContrastHelper(img_c)

        refresh()
        plt.show()

        # return state when done
        return state["x"], state["y"], state["z"]
    
    def graph_attenuation(self, raytracer: Raytracer):
        '''Graphs attenuation from start of ray to end'''
        # setup plotter and ax
        plotter = pv.Plotter(shape=(2, 2), groups=[([0,1],0)], window_size=(1000, 1200), border_color='white')
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
            update_fixed_theta_chart(ray_data, raytracer.origin, self.image, self.seg, ax2)
            img[0] = pv.ChartMPL(fig2)
            plotter.add_chart(img[0])

        # show base scene for ray selection + enable picking on the sphere only
        self.plot_segmentation(plotter, 'total')
        body_clickable = self.plot_segmentation(plotter, 'body',cmap="Grays", opacity=0.2)
        self.plot_segmentation(plotter, 'liver_vessels', cmap ='Blues', opacity=0.9)
        #body_clickable = self.visualize_body_scoring(plotter, raytracer, 500)
        plotter.enable_surface_point_picking(callback=select_point, show_message=False)
        plotter.pickable_actors = [body_clickable]
        plotter.show()
        plt.close(fig)

def update_ray_attenuation_chart(ray_data: RayData, ax: plt.Axes):
    ax.clear()
    segments = ray_data.ordered
    opacity_map = get_label_opacity_maps()

    # cumulative distance breakpoints and attenuation values
    distances    = [0.0]
    attenuations = [1.0]
    total_opacity, d = 0.0, 0.0

    for seg in segments:
        task   = seg["task"]
        label  = seg["tissue"]
        length = seg["length"]
        alpha  = seg["alpha"]

        if task == "body": continue
        rav_opacity = opacity_map[task][label] if label != 0 else 0.0
        opacity = rav_opacity * alpha if rav_opacity != 1 else np.inf 

        ax.axvspan(d, d + length, alpha = min(opacity, 1), color = 'blue', linewidth = 0)
        distances.append(d)
        total_opacity += opacity * length
        attenuations.append(np.exp(-total_opacity))
        d += length

    # attenuation curve on top
    ax.axvspan(14.9, 15.1, alpha = 0.5, color = 'red', linewidth = 0)
    ax.plot(distances, attenuations, color="red", lw=2, zorder=3)
    ax.set_xlim(0, d)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Illumination")
    ax.set_facecolor("#111111")
    ax.set_title("Ray attenuation", pad=10)

def update_fixed_theta_chart(ray_data: RayData, origin: np.ndarray, image, seg, ax: plt.Axes):
    ''' Vertical CT volume along ray's theta angle, with labels.'''
    ax.clear()
    rx, ry, rz = image.resolution
    array = np.transpose(image.array, (2, 1, 0))  # (X, Y, Z)

    # basis vectors
    theta = np.arctan2(ray_data.vector[1], ray_data.vector[0])
    u     = np.array([np.cos(theta), np.sin(theta), 0])
    v     = np.array([0, 0, 1])

    # sample grid
    sx, sy, sz = array.shape
    r_extent   = np.linalg.norm([sx*rx, sy*ry]) / 2
    z_extent   = sz * rz / 2

    r_samples  = int(r_extent * 2 / min(rx, ry))
    z_samples  = int(z_extent * 2 / rz)

    r_coords   = np.linspace(-r_extent, r_extent, r_samples)
    z_coords   = np.linspace(-z_extent, z_extent, z_samples)
    rr, zz     = np.meshgrid(r_coords, z_coords, indexing='ij')

    # convert coords
    world = (origin[:, None, None] + rr[None] * u[:, None, None] + zz[None] * v[:, None, None])
    voxel_coords = world / np.array([rx, ry, rz])[:, None, None]

    # ct
    ct_slice = map_coordinates(array, voxel_coords, order=1, mode='constant', cval=-1000)
    ax.imshow(ct_slice.T, cmap='gray', origin='lower',extent=[-r_extent, r_extent, -z_extent, z_extent],aspect='equal')

    # ray
    ray_r = np.dot(ray_data.vector, u)   # projection onto plane axes
    ray_z = np.dot(ray_data.vector, v)
    length = getattr(ray_data, 'total_length', r_extent)
    ax.annotate('', xy=(ray_r * (length+15), ray_z * (length+15)), xytext=(-15*ray_r, -15*ray_z),arrowprops=dict(arrowstyle='<-', color='red', lw=1.5))

    ax.scatter(0, 0, color='red', s=30, zorder=5)  # origin marker
    ax.set_xlabel("Radial distance (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(f"θ = {np.degrees(theta):.1f}°")
    ax.set_facecolor("#111111")