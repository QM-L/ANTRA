import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Qt

from antra.interface.elements import SectionFrame, DicomSliceWidget
from antra.visualisation.visualizer import Visualizer


class SetupPage(SectionFrame):
    '''Setup visuals: linked DICOM slice viewer (left),
       segmentation surface viewer and ablation ellipsoid (right column).'''

    def __init__(self, parent=None):
        super().__init__(False, parent)

        container    = QWidget()
        main_layout  = QHBoxLayout(container)

        right_col        = QWidget()
        right_layout     = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Left: embedded Matplotlib DICOM viewer
        self.dicom_widget = DicomSliceWidget()

        # Right top: segmentation surface plotter
        self.seg_plotter = QtInteractor(right_col)
        self.seg_plotter.set_background("#1a1a1a")

        # Right bottom: ablation ellipsoid plotter
        self.ablation_plotter = QtInteractor(right_col)
        self.ablation_plotter.set_background("#1a1a1a")

        right_layout.addWidget(self.seg_plotter, 1)
        right_layout.addWidget(self.ablation_plotter, 1)

        main_layout.addWidget(self.dicom_widget, 2)
        main_layout.addWidget(right_col, 1)

        self.add_widget(container)

        self.visualizer = None

    ### rendering

    def show_dicom(self, visualizer: Visualizer) -> None:
        '''Populate the DICOM slice viewer from a loaded Visualizer.'''
        self.visualizer = visualizer
        self.dicom_widget.update(visualizer)

    def show_segmentation(self, visualizer: Visualizer) -> None:
        '''Render total + liver_vessels segmentations in the seg plotter.'''
        self.seg_plotter.clear()
        visualizer.plot_base_scene(self.seg_plotter)
        self.seg_plotter.render()

    def show_ablation(self, visualizer: Visualizer) -> None:
        '''Render the ablation ellipsoid wireframe in the ablation plotter.'''
        self.ablation_plotter.clear()
        visualizer.plot_ablation_zone(self.ablation_plotter)
        self.ablation_plotter.render()

    def selected_voxel(self) -> tuple[int, int, int] | None:
        '''Returns the voxel currently selected in the slice viewer, or None.'''
        return self.dicom_widget.selected_voxel()

    def cleanup(self):
        self.seg_plotter.close()
        self.ablation_plotter.close()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)


class SelectPage(SectionFrame):
    '''Select visuals: linked DICOM slice viewer with selector enabled.'''

    def __init__(self, parent=None):
        super().__init__(False, parent)

        # embedded Matplotlib DICOM viewer
        self.dicom_widget = DicomSliceWidget(selection=True)
        self.dicom_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.add_widget(self.dicom_widget)
        self.visualizer = None
    
    def show_dicom(self, visualizer: Visualizer) -> None:
        '''Populate the DICOM slice viewer from a loaded Visualizer.'''
        self.visualizer = visualizer
        self.dicom_widget.update(visualizer)

    def toggle_segmentations(self):
        self.dicom_widget.toggle_segmentations()

    def selected_voxel(self) -> tuple[int, int, int] | None:
        '''Returns the voxel currently selected in the slice viewer, or None.'''
        return self.dicom_widget.selected_voxel()

    def cleanup(self):
        if self.dicom_widget.canvas is not None:
            self.dicom_widget.box.removeWidget(self.dicom_widget.canvas)

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)


class AdvicePage(SectionFrame):
    '''Needle placement page: single plotter with range preview
       mode before raytracing and scoring/advice mode after.'''

    def __init__(self, parent=None):
        super().__init__(False, parent)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1a1a1a")
        self.add_widget(self.plotter)

        self.visualizer    = None
        self._range_actor   = None   # blue range preview point cloud
        self._scoring_actor = None   # body scoring mesh actor
        self._arrow_actors  = []     # advice arrow actors
        self._showing_scores = True

    ### Range Preview

    def setup(self, visualizer: Visualizer) -> None:
        '''Initial setup: render base scene and show default range preview.'''
        self.visualizer = visualizer
        visualizer.plot_base_scene(self.plotter)
        self.plotter.reset_camera()
        self.plotter.render()

    def update_range_preview(self, raytracer, density) -> None:
        '''Highlight the currently selected angular range as a blue point cloud
           projected onto the body surface at a fixed radius.'''
        self.visualizer.update_range_preview(self.plotter, raytracer, density)

    ### scoring / advice results

    def show_results(self, visualizer: Visualizer, raytracer, advice: list[dict],
                     max_results: int) -> None:
        '''Switch to results mode: scoring mesh + top N advice arrows.'''
        # clear range preview
        if self._range_actor is not None:
            self.plotter.remove_actor(self._range_actor)
            self._range_actor = None

        # body scoring mesh
        self._scoring_actor = visualizer.visualize_body_scoring(
            self.plotter, raytracer)

        # draw top N advice arrows
        self._draw_advice_arrows(advice[:max_results], raytracer.origin)
        self.plotter.reset_camera()
        self.plotter.render()

    def _draw_advice_arrows(self, advice: list[dict], origin) -> None:
        '''Render an arrow + label for each advice entry.'''
        # remove old arrows
        for actor in self._arrow_actors:
            self.plotter.remove_actor(actor)
        self._arrow_actors = []

        origin     = np.array(origin, dtype=float)
        ray_length = 120.0

        for i, adv in enumerate(advice):
            t, p = adv['theta'], adv['phi']
            d    = np.array([np.sin(p) * np.cos(t),np.sin(p) * np.sin(t), np.cos(p)])
            arrow = pv.Arrow(
                start=origin, direction=d, scale=ray_length,
                shaft_radius=0.008, tip_radius=0.025
            )
            actor = self.plotter.add_mesh(arrow, color='yellow', name=f'advice_{i}')
            self._arrow_actors.append(actor)

            self.plotter.add_point_labels(
                [origin + d * ray_length],
                [f"#{i+1}  {adv['score']:.3f}"],
                font_size=12, text_color='yellow',
                fill_shape=False, shape_opacity=0,
            )

    def highlight_advice(self, index: int, advice: list[dict], origin) -> None:
        '''Redraw arrows with the selected one highlighted in a different color.'''
        for actor in self._arrow_actors:
            self.plotter.remove_actor(actor)
        self._arrow_actors = []

        origin     = np.array(origin, dtype=float)
        ray_length = 120.0

        for i, adv in enumerate(advice):
            t, p  = adv['theta'], adv['phi']
            d     = np.array([np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)])
            color = 'cyan' if i == index else 'yellow'
            arrow = pv.Arrow(start=origin, direction=d, scale=ray_length,
                             shaft_radius=0.008, tip_radius=0.025)
            actor = self.plotter.add_mesh(arrow, color=color, name=f'advice_{i}')
            self._arrow_actors.append(actor)

        self.plotter.render()

    def toggle_scoring_colors(self, visible: bool) -> None:
        '''Show or hide the scoring color mesh.'''
        if self._scoring_actor is not None:
            self._scoring_actor.SetVisibility(visible)
            self.plotter.render()
    
    def cleanup(self):
        # reset plotter
        self.plotter.clear()
        self.plotter.set_background("#1a1a1a")
        
        self.visualizer = None
        self._range_actor = None
        self._scoring_actor = None
        self._arrow_actors = []

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)