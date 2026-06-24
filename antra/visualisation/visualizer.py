import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import map_coordinates

from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation
from antra.needle_placing.scoring import *
from antra.needle_placing.raytracer import Raytracer
from antra.needle_placing.needle_advisor import NeedleAdvisor
from antra.visualisation.tumor_selection import MplContrastHelper
from antra.general.config import get_label_opacity_maps, load_configs

class Visualizer():
    '''Class containing all functions to visualize something to the user'''
    def __init__(self, image: DICOM_Scan, segmentations: dict[str,Segmentation]) -> None:
        self.image  = image
        self.config = load_configs()
        self.seg    = segmentations
        self.array  = np.transpose(self.image.array, (2, 1, 0))
        self.volume = self.get_3D_volume()
        self._range_actor = None

    def get_3D_volume(self) -> pv.ImageData:
        volume = pv.ImageData()
        volume.dimensions = self.array.shape
        volume.spacing = self.image.resolution
        volume.point_data["CT"] = self.array.flatten(order="F")
        volume = volume.resample(0.3,interpolation="nearest")
        return volume

    def plot_base_scene(self, plotter: pv.Plotter) -> tuple[pv.Actor,pv.Actor,pv.Actor]:
        ''''Plots the body and it's insides.'''
        actors = []
        actors.append(self.plot_segmentation(plotter, 'total', cmap='Reds', opacity=0.5))
        actors.append(self.plot_segmentation(plotter, 'body',  cmap='Grays', opacity=0.15))
        actors.append(self.plot_segmentation(plotter, 'liver_vessels', cmap='Blues', opacity=0.95))
        plotter.reset_camera()
        return actors

    def plot_ablation_zone(self, plotter: pv.Plotter) -> None:
        '''plots the ablation zone centered around origin'''
        r_r = self.config.getfloat('ablation', 'radius_r')
        r_z = self.config.getfloat('ablation', 'radius_z')

        sphere = pv.Sphere(radius=1.0, theta_resolution=32, phi_resolution=32)
        sphere.points *= np.array([r_r, r_r, r_z])

        plotter.add_mesh(sphere, style='wireframe', color='red', line_width=1)
        plotter.add_mesh(sphere, color='cyan', opacity=0.12)
        plotter.reset_camera()

    def plot_segmentation(self, plotter: pv.Plotter, seg_task='total', cmap='Reds', opacity: float = 0.95) -> pv.Actor:
        '''Plots specified segmentation'''
        seg = self.seg.get(seg_task)
        array = np.asarray(seg.raw_mask.dataobj)
    
        # define the masks's 3d volume
        volume = pv.ImageData()
        volume.dimensions = np.array(array.shape) + 1
        volume.spacing    = np.sqrt((seg.raw_mask.affine[:3, :3] ** 2).sum(axis=0)).tolist()
        volume.origin     = self.image.origin
        volume.cell_data["label"] = array.flatten(order="F")

        # Threshold out background (label 0)
        segmented = volume.threshold(value=0.5, scalars="label")

        # define custom colormap where liver is blue
        n = int(np.floor(array).max()) + 1
        if seg_task == "total" and n > 5:
            base   = plt.get_cmap(cmap, n)
            colors = [base(i / n) for i in range(n)]
            colors[5] = mcolors.to_rgba('yellow')
            cmap = mcolors.ListedColormap(colors)

        # plot
        actor = plotter.add_mesh(segmented, scalars="label", log_scale=True, cmap=cmap, opacity=opacity,show_scalar_bar=False)
        return actor

    def build_slice_figure(self, select=False) -> tuple[plt.Figure, tuple]:
        '''Build the linked three-view Matplotlib figure for embedding.
        Returns (fig, state) so the caller can read the selected voxel.'''
        fig = plt.figure(layout='constrained')
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot()
        rx, ry, rz = self.image.resolution
        state = {"x": self.array.shape[0] // 2,"y": self.array.shape[1] // 2,"z": self.array.shape[2] // 2, "show_seg": True}

        # intial image initialisation
        x, y, z = state["x"], state["y"], state["z"]

        img = ax.imshow(self.array[:,:,z].T, cmap="gray", origin="lower", aspect=ry/rx)

        alpha = 0.3 if select else 0
        line_h = ax.axhline(y, color="red", lw=0.8, alpha=alpha)
        line_v = ax.axvline(x, color="red", lw=0.8, alpha=alpha)

        # get segmentations
        tumor_array = liver_array = np.zeros_like(self.array)
        if self.seg:
            tumor_array = np.asarray(self.seg['liver_vessels'].raw_mask.dataobj)
            liver_array   = np.asarray(self.seg['total'].raw_mask.dataobj)
        contours = {"tumor": [], "liver": []}

        ax.invert_yaxis()

        def refresh():
            # change data in the images based on new xyz values
            x, y, z = state["x"], state["y"], state["z"]
            img.set_data(self.array[:,:,z].T)
            line_h.set_ydata([y, y])
            line_v.set_xdata([x, x])
            change_contours(z)
            fig.canvas.draw_idle()

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
                cs = ax.contour(tumor_slice, levels=[0.5], colors=["cyan"], linewidths=0.8)
                contours["tumor"] = [cs]

            if liver_slice.any():
                cs = ax.contour(liver_slice,   levels=[0.5], colors=["yellow"], linewidths=0.8)
                contours["liver"]   = [cs]

        def on_click(event):
            if event.button == 1: return # left click is dragging
            if event.inaxes != ax: return
            state["x"], state["y"] = int(event.xdata), int(event.ydata)
            refresh()

        fig.canvas.mpl_connect("button_press_event", on_click)

        # scroll to change slice
        def on_scroll(event):
            delta = 1 if event.button == "up" else -1
            if event.inaxes != ax: return
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
        HU_handlers = MplContrastHelper(img)

        refresh()
        return fig, state

    def update_range_preview(self, plotter: pv.Plotter, raytracer: Raytracer, density: int):
        # remove old preview
        if self._range_actor is not None:
            plotter.remove_actor(self._range_actor)
            self._range_actor = None

        # generate directions for current range
        directions, _ = raytracer.generate_n_directions(int(density * raytracer.get_area()))

        # approximation of smallest radius that'll exit all rays from the body
        radius = float(np.linalg.norm( np.array(raytracer.segmentations['body'].dicom.dimensions) * np.array(raytracer.voxel_size))) * 0.4
        origin = np.array(raytracer.origin, dtype=float)
        points = directions * radius + origin

        cloud = pv.PolyData(points)
        self._range_actor = plotter.add_mesh(cloud, color='dodgerblue', opacity=1, point_size=4, render_points_as_spheres=True, name='range_preview')
        plotter.render()
        return self._range_actor

    def visualize_body_scoring(self, plotter: pv.Plotter, origin: list[float], score_data: list[dict], weights: list[float], interpolation_radius: float = 5):
        '''Color the body surface mesh by ray scores from the origin.'''
        # body mask to vertex mesh
        array   = self.seg['body'].get_array(np.uint16)
        vol = pv.ImageData()
        vol.dimensions = np.array(array.shape) + 1
        vol.spacing    = np.sqrt((self.seg['body'].mask.affine[:3, :3] ** 2).sum(axis=0)).tolist()
        vol.origin     = self.image.origin
        vol.cell_data["label"] = array.flatten(order="F")
        body_mesh = vol.threshold(0.5).extract_surface(algorithm='dataset_surface').smooth(n_iter=100, relaxation_factor=0.05)

        # get scores and corresponding points
        ablation_center_dist = self.config.getfloat("needle", "ablation_center_dist")
        raw_scores = np.array([data["scores"] for data in score_data])
        directions = np.array([data["dir"] for data in score_data])
        lengths = np.array([data["length"] - ablation_center_dist for data in score_data])
        scores = np.prod(raw_scores ** np.array(weights), axis=1)
        points = directions * lengths[:, np.newaxis] + np.array(origin)

        # Build a point cloud with score scalars
        cloud = pv.PolyData(points)
        cloud["score"] = np.array(scores)
        body_mesh = body_mesh.interpolate(cloud, radius=interpolation_radius)
        
        # plot
        scoring_actor = plotter.add_mesh(body_mesh, name="body_scoring", scalars="score", cmap="Oranges", log_scale=False, show_scalar_bar=True, scalar_bar_args={"color":"white"})
        return scoring_actor