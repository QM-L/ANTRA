import numpy as np
import pyvista as pv
from SimpleITK import Image, sitkInt16
from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.segmentator import Segmentation
from antra.needle_placing.raytracer import Raytracer
from antra.needle_placing.scoring import Scorer
from antra.visualisation.visualizer import Visualizer

# For most of these tests, ablation_center_dist should be set to 0

def run_length_validation() -> None:
    '''Generates phantom segmentations and uses them to verify basic functionality'''

    # generate empty dicom image with anisotropic resolution 1.0,1.0,1.5mm
    shape = (256,256,256)
    res = (1.0, 1.0, 1.5)
    manual_image = Image(*shape, sitkInt16)
    manual_image.SetSpacing(res)
    dicom = DICOM_Scan(manual_img=manual_image)

    # generate a r=10cm sphere of body, 
    # and a nested sphere with layers 5mm of tumor, 2.5 cm of liver vessel, 5cm of liver
    # all centered on 11,11,11cm
    ray_origin = (110,110,110)
    body = generate_sphere(shape, res, 100, ray_origin, 1)
    tumor = generate_sphere(shape, res, 5, ray_origin, 2)
    liver_vessels = generate_sphere(shape, res, 30, ray_origin, 1)
    combined = np.where(tumor == 0, liver_vessels, tumor)
    liver = generate_sphere(shape, res, 80, ray_origin, 5)

    phantoms = {
        "liver_vessels": Segmentation(dicom, "liver_vessels", manual_array=combined),
        "total": Segmentation(dicom, "total", manual_array=liver),
        "body": Segmentation(dicom, "body", manual_array=body)
        }

    # setup raytracer with origin in center
    ray_tracer = Raytracer(0, phantoms)
    ray_tracer.set_origin(*ray_origin)

    # trace and get lengths
    ray_data,_ = ray_tracer.analyze_range(1000)
    total_lengths = np.array([ray.total_length for ray in ray_data])

    # collect data
    tissue_lengths = {"tumor":[],"vessel":[],"liver":[],"body":[], "total": []}
    for ray in ray_data:
        tissue_lengths["tumor"].append(ray.lengths["liver_vessels"][2])
        tissue_lengths["vessel"].append(ray.lengths["liver_vessels"][1])
        tissue_lengths["liver"].append(ray.lengths["total"][5])
        tissue_lengths["body"].append(ray.lengths["body"][1]),
        tissue_lengths["total"].append(total_lengths)

    # parse
    expected = {"tumor": 5, "vessel": 25, "liver": 50, "body": 20, "total": 100}
    tissue_length_data = {tissue: [
        expected[tissue],
        float(np.mean(lengths)),
        float(np.std(lengths)), 
        float(np.max(np.abs(np.array(lengths) - expected[tissue])))
        ] for tissue,lengths in tissue_lengths.items()}

    print("TYPE, EXPECTED, MEAN, STD, MAX ERROR:")
    print(*[f"{tissue} | {data}\n" for tissue, data in tissue_length_data.items()])

def run_criterion_validation() -> None:
    '''Generates a few segmentations with clear ground truth scoring results for each criterion
       and shows the scoring mesh for it.'''

    # generate empty dicom image
    shape = (256,256,256)
    res = (1.0, 1.0, 1.0)
    manual_image = Image(*shape, sitkInt16)
    manual_image.SetSpacing(res)
    dicom = DICOM_Scan(manual_img=manual_image)
    middle = (128,128,128)

    # reusable segmentation dict and base masks
    seg = {"liver_vessels": None, "total": None, "body": None}
    body = generate_sphere(shape, res, 100, middle, 1)
    tumor = generate_ellipsoid(shape, res, (5,5,10), middle, 2)
    liver = generate_sphere(shape, res, 50, middle, 5)
    no_liver = generate_sphere(shape, res, 1, (250,250,250), 5)

    # 1 tissue traversal
    vessels = np.where(tumor == 0, generate_cylinder(shape, res, 1, 200, (111, 200, 128), 2, 1.99), tumor)
    rib = generate_cylinder(shape, res, 1, 200, (145, 200, 128), 2, 1)
    seg["liver_vessels"] = Segmentation(dicom, "liver_vessels", manual_array=vessels)
    seg["total"] = Segmentation(dicom, "total", manual_array=rib)
    seg["body"] = Segmentation(dicom, "body", manual_array=body)
    show_criterion(seg, middle, [1,0,0,0,0])

    # 2 skin angle
    high_tumor = generate_ellipsoid(shape, res, (5,5,10), (128,128,220), 2)
    seg["liver_vessels"] = Segmentation(dicom, "liver_vessels", manual_array=high_tumor)
    seg["total"] = Segmentation(dicom, "total", manual_array=no_liver)
    show_criterion(seg, (128,128,220), [0,1,0,0,0])

    # 3 score length
    show_criterion(seg, (128,128,220), [0,0,1,0,0])

    # 4 liver entrance
    seg["liver_vessels"] = Segmentation(dicom, "liver_vessels", manual_array=generate_ellipsoid(shape, res, (5,5,10), (128,128,175), 2))
    seg["total"] = Segmentation(dicom, "total", manual_array=liver)
    show_criterion(seg, (128,128,175), [0,0,0,1,0])

    # 5 ablation zone coverage
    seg["liver_vessels"] = Segmentation(dicom, "liver_vessels", manual_array=generate_ellipsoid(shape, res, (5,5,10), (128,128,128), 2))
    seg["total"] = Segmentation(dicom, "total", manual_array=no_liver)
    show_criterion(seg, (128,128,128), [0,0,0,0,1])


def generate_sphere(shape: tuple, resolution: tuple, radius_mm: float, origin_mm: tuple, label: int =1) -> np.ndarray:
    '''generates a spherical numpy mask'''
    return generate_ellipsoid(shape, resolution, (radius_mm,radius_mm,radius_mm),origin_mm, label)

def generate_ellipsoid(shape: tuple, resolution: tuple, radii_mm: tuple, origin_mm: tuple, label: int = 1) -> np.ndarray:
    '''generates an ellipsoidal numpy mask'''
    # generate mm coordinate grid of shape
    grids = [np.arange(shape[i]) * resolution[i] for i in range(3)]
    zz, yy, xx = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')

    # normalized squared distance
    distance = (((zz-origin_mm[0])/radii_mm[0])**2 + ((yy-origin_mm[1])/radii_mm[1])**2 + ((xx-origin_mm[2])/radii_mm[2])**2)
    return (distance <= 1.0) * label

def generate_cylinder(shape: tuple, resolution: tuple, radius_mm: float, length_mm: float, origin_mm: tuple, axis: int = 0, label: int = 1) -> np.ndarray:
    '''generates a cylindrical numpy mask aligned along a given axis (0=z, 1=y, 2=x)'''
    grids = [np.arange(shape[i]) * resolution[i] for i in range(3)]
    zz, yy, xx = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
    coords = [zz, yy, xx]

    # cylindrical axis constraint
    axial = coords[axis] - origin_mm[axis]
    within_length = axial ** 2 <= (length_mm / 2) ** 2

    # radial constraint
    radial_axes = [i for i in range(3) if i != axis]
    radial = sum(((coords[i] - origin_mm[i]) ** 2) for i in radial_axes)
    within_radius = radial <= radius_mm ** 2

    return (within_length & within_radius) * label


def show_criterion(seg, origin, weights):
    # ray trace
    rt = Raytracer(0, seg)
    rt.set_origin(*origin)
    rays, _ = rt.analyze_range(1000)

    # score
    scorer = Scorer(seg, origin)
    score_data = [{"length": ray.total_length, "dir": ray.vector, "scores": scorer.get_scores(ray)} for ray in rays]

    # visualize
    vis = Visualizer(seg["body"].dicom, seg)
    plotter = pv.Plotter(shape=(1,2))

    plotter.subplot(0,0)
    vis.plot_segmentation( plotter, "body", cmap="Grays", opacity=0.2)
    vis.plot_segmentation( plotter, "liver_vessels", cmap="autumn", opacity=0.9)
    vis.plot_segmentation( plotter, "total", cmap="Blues", opacity=0.9)
    plotter.add_mesh( pv.PolyData([origin]), color="red", point_size=15, render_points_as_spheres=True)
    


    plotter.subplot(0,1)
    vis.visualize_body_scoring(plotter, np.array(origin), score_data, weights)
    
    plotter.show()