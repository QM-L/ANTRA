import numpy as np
import pyvista as pv
from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation

def show_body():
    '''Plots a smooth+shaded but less accurate body segmentation.'''
    # setup objects
    dicom = DICOM_Scan("scans/DICOM_3D_IRCADB_01_8")
    seg = Segmentation(dicom=dicom, task="body", folder="data/Segmentation_3D_IRCADB_01_8",load=True)
    plotter = pv.Plotter(border_color='white')
    array = np.pad(np.asarray(seg.raw_mask.dataobj), pad_width=1, mode='constant', constant_values=0)

    # define the masks's 3d volume
    volume = pv.ImageData()
    volume.dimensions = np.array(array.shape) + 1
    volume.spacing    = np.sqrt((seg.raw_mask.affine[:3, :3] ** 2).sum(axis=0)).tolist()
    volume.origin     = dicom.origin
    volume.cell_data["label"] = array.flatten(order="F")

    # Convert cell data to point data (required for contour/marching cubes)
    volume_pts = volume.cell_data_to_point_data()

    # Extract isosurface via Marching Cubes at the 0.5 boundary
    surface = volume_pts.contour(isosurfaces=[0.5], scalars="label")

    # Smooth the surface (Laplacian smoothing)
    smoothed = surface.smooth(n_iter=250, relaxation_factor=0.1)

    # Optional: recompute normals for better shading
    smoothed.compute_normals(cell_normals=False, point_normals=True, inplace=True)

    # Plot
    plotter.add_mesh(smoothed, color="#ffd8ae", opacity=1.0, smooth_shading=True, show_scalar_bar=False)
    plotter.show()