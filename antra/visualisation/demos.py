import numpy as np
import pyvista as pv

from antra.visualisation.visualizer import Visualizer
from antra.needle_placing.raytracer import Raytracer

''''Demos that demonstrate features, only meant for testing.'''

def demo_seg_margin(visualizer: Visualizer):
    '''Visualization of segmentation margins'''
    visualizer.segmentation_margin_view()

def demo_rendertime(visualizer: Visualizer, raytracer: Raytracer):
    '''Visualization of checking all rays'''

    # base scene
    plotter = pv.Plotter()
    visualizer.visualize_body_scoring(plotter,raytracer, 1000)
    plotter.show()

def demo_circle(visualizer: Visualizer, raytracer: Raytracer):
    '''Visualization of checking all rays'''

    # base scene
    plotter = pv.Plotter()
    visualizer.plot_base_scene(plotter)
    visualizer.visualize_angles(plotter, raytracer, 100)
    plotter.show()

def demo_3D_visualizer(visualizer: Visualizer):
    '''show a 3D image of the raw DICOM ct-scan with some extra stuff'''
    # base scene
    plotter = pv.Plotter()
    visualizer.plot_base_scene(plotter)
    plotter.show()