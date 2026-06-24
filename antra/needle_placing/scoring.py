import numpy as np
from scipy.ndimage import gaussian_filter, label

from antra.segmentation.segmentator import Segmentation
from antra.needle_placing.ray_data import RayData
from antra.general import config

'''functions pertaining to ray score calculation'''

SCORER_NAMES = ["Intensity", "Skin Angle", "Needle Length", "Liver Entrance","Ablation Coverage"]


def weigh_score(scores: list, weights: list) -> list:
    '''Compiles scorer scores into weighted scores'''
    return np.prod([score**weights[i] for i, score in enumerate(scores)])


class Scorer():
    '''Handles scoring a single path based on specified path scoring.'''
    def __init__(self, segmentations: dict[str, Segmentation], origin: tuple[float,float,float]):
        # load all values required to score rays
        self.config = config.load_configs()

        # intensity-related
        self.opacities = {tissue: float(resist) for tissue,resist in self.config.items(section='opacity')}
        self.opacity_map = config.get_label_opacity_maps()

        # body entry angle mask
        self.body_gradient = self.compute_body_gradient(segmentations)

        # ablation zone
        ablation_scorer = AblationScorer(segmentations, origin)

        # actual scoring parameters (functions and weights)
        self.weights = self.config.gettuple('scoring', 'weights')
        self.scoring_functions = [
            self.score_intensity, 
            self.score_skin_angle,
            self.score_length,
            self.score_liver_entrance,
            ablation_scorer.get_score
            ]

    def get_scores(self, ray_data: RayData) -> tuple[float]:
        return [func(ray_data) for func in self.scoring_functions]

    ### Scoring functions 

    def score_intensity(self, ray: RayData) -> float:
        '''Turn a completed acculumator into a path intensity value.'''
        total_intensity = 1
        for task in ray.tasks:
            lengths = ray.lengths[task]
            task_translucency = np.array([self.get_translucency(task, i, length) for i, length in enumerate(lengths)])
            total_intensity *= np.prod(task_translucency)
        return total_intensity

    def score_skin_angle(self, ray: RayData) -> float:
        '''Find the angle between the skin and needle at enterance and score it.'''
        angle = 90 - abs(self.get_entry_angle(ray.entry_voxel, ray.vector))
        return min(angle / self.config.getfloat('scoring', 'angle_threshold'), 1)

    def score_length(self, ray: RayData) -> bool:
        '''Whether or not the ray is a viable length (smaller than config effective needle length)'''
        return max(0, 1 - ray.total_length / self.config.getint('needle','length'))
    
    def score_liver_entrance(self, ray: RayData):
        ''''Requires the needle to enter a certain amount of healthy liver tissue first.'''
        liver_tissue_traversed = 0
        for data in reversed(ray.ordered):
            # end checking when we reach the liver tumor
            if data['task'] == 'liver_vessels' and data['tissue'] == 2: break

            if data['task'] != 'total': continue # add length if healthy liver tissue
            if data['tissue'] != 5: continue     #
            liver_tissue_traversed += data['length']
        return max(0, min(liver_tissue_traversed - self.config.getint('scoring', 'healthy_entrance_min_mm'), 1))

    ### Scoring helper functions

    def get_translucency(self, task: str, tissue_label: int, length: float = 1) -> float:
        '''get the attenuation for this length of tissue'''
        if length == 0 or tissue_label == 0: return 1
        translucency = 1 - self.opacity_map[task][tissue_label]
        return translucency ** length

    def get_entry_angle(self, voxel: tuple[int], vector: float) -> float:
        '''angle in degrees between ray and skin normal'''
        normal = self.body_gradient[voxel[0], voxel[1], voxel[2]]
        size = np.linalg.norm(normal)
        if size < 1e-6: return 90 # entry voxel has no gradient, likely edge of ct (and thus not viable eitherway)
                         
        normal  = normal / size
        cos_a   = np.clip(abs(np.dot(vector, normal)), 0, 1)
        return float(np.degrees(np.arccos(cos_a)))

    ### Other

    def compute_body_gradient(self, segmentations: dict[str, Segmentation]) -> np.ndarray:
        # load body array, smooth to make directions more correct
        body_array = np.asarray(segmentations['body'].raw_mask.dataobj, dtype=np.float32)
        voxel_size = segmentations['body'].dicom.resolution
        smoothed   = gaussian_filter(body_array, sigma=[2/voxel_size[0], 2/voxel_size[1], 2/voxel_size[2]])

        # turn into gradient
        gx, gy, gz = np.gradient(smoothed, voxel_size[0], voxel_size[1], voxel_size[2])
        return np.stack([gx, gy, gz], axis=-1)


class AblationScorer():
    '''Seperate class to evaluate the ablation coverage'''
    def __init__(self, segmentations: dict[str, Segmentation], ablation_origin: tuple[float,float,float]):
        self.config = config.load_configs()

        # ablation zone as ellipsoid with 2 radii
        self.origin    = np.array(ablation_origin, dtype=float)
        self.radius_r  = self.config.getfloat('ablation', 'radius_r')
        self.radius_z  = self.config.getfloat('ablation', 'radius_z')
        margin         = self.config.getfloat('ablation', 'tumor_margin')
        self.radius_rm = self.radius_r - margin
        self.radius_zm = self.radius_z - margin

        # tumor mask array 
        seg               = segmentations['liver_vessels']
        self.seg_array    = np.asarray(seg.raw_mask.dataobj, dtype=np.uint8)
        self.resolution   = np.array(seg.dicom.resolution)
        self.tumor_points = TumorAnalyzer(segmentations).nearest_tumor_points(ablation_origin)

        # margin
        tumor_radius    = np.linalg.norm(self.tumor_points, axis=1).max()
        self.max_margin = min(self.radius_r, self.radius_z) - tumor_radius

    def get_score(self, ray: RayData) -> float:
        '''Score this specific ray's angle'''
        margin = self.min_margin(ray.vector)
        return max(0,min(margin/self.max_margin - 1, 1))

    def min_margin(self, ray_dir: np.ndarray) -> float:
        '''minimum margin between tumor and ellipsoid edge in mm.'''
        rot        = self.rotation_matrix(ray_dir)
        local      = self.tumor_points @ rot
        normalized = np.sqrt((local[:, 0] / self.radius_r) ** 2 +(local[:, 1] / self.radius_r) ** 2 + (local[:, 2] / self.radius_z) ** 2)
        return float((1.0 - normalized.max()) * min(self.radius_r, self.radius_z))

    def rotation_matrix(self, ray_dir: np.ndarray) -> np.ndarray:
        '''compute rotation matrix that maps z-axis to the ray direction.'''
        # new z-basis vector is the ray direction
        z = np.array(ray_dir, dtype=float)
        z = z / np.linalg.norm(z)

        # pick arbitrary vector as x and y basis vectors
        vec = np.array([1,0,0]) if abs(z[0]) < 0.9 else np.array([0,1,0])
        x = np.cross(z, vec);  
        x = x/np.linalg.norm(x)
        y = np.cross(z, x)

        return np.column_stack([x, y, z])


class TumorAnalyzer():
    ''''Seperate class which analyzes tumors in the segmentation for info. used during selection to give user info.'''
    TUMOR_LABEL = 2
    TUMOR_TASK = "liver_vessels"

    def __init__(self, segmentations: dict[str, Segmentation]):
        seg = segmentations[self.TUMOR_TASK]
        raw = np.asarray(seg.raw_mask.dataobj, dtype=np.uint8)
        self.resolution = np.array(seg.dicom.resolution, dtype=float)

        # label each connected tumor region separately
        labelled, self.count = label(raw == self.TUMOR_LABEL)

        # pre-compute mm coordinates and centroid for every tumor
        self.tumors = []
        for i in range(1, self.count + 1):
            voxels = np.argwhere(labelled == i).astype(float)
            mm = voxels * self.resolution
            centroid = mm.mean(axis=0)
            self.tumors.append({"mm": mm, "center": centroid})

    def stats_for_origin(self, origin_vox: np.ndarray) -> dict:
        '''returns stats for the tumor nearest to origin_mm.'''
        if self.count == 0: return {"nearest": None, "size_mm": 0.0, "max_radius_mm": 0.0}
        origin_mm = np.array(origin_vox) * self.resolution

        # find nearest tumor by centroid distance
        nearest = self.nearest_tumor(origin_vox)
        pts = nearest["mm"]

        # max radius from center and volume
        center = nearest["center"]
        max_diameter = np.linalg.norm(pts - center, axis=1).max() * 2
        volume_mm3 = len(pts) * np.prod(self.resolution)

        # distance from center
        distance = np.linalg.norm(center - origin_mm)

        return {
            "# tumors": self.count,
            "current pos (mm)": np.round(origin_mm,1),
            "nearest tumor (mm)": np.round(center, 1),
            "dist from center (mm)": round(distance, 1),
            "volume (mm³)": round(float(volume_mm3)),
            "max radius (mm)": round(float(max_diameter), 1)
        }

    def nearest_tumor_points(self, origin_mm: np.ndarray) -> np.ndarray:
        '''returns mm coordinates of the nearest tumor's voxels, relative to origin.'''
        if self.count == 0: raise ValueError("No tumor voxels found")
        origin = np.array(origin_mm, dtype=float)
        nearest = min(self.tumors, key=lambda t: np.linalg.norm(t["center"] - origin))
        return nearest["mm"] - origin
    
    def nearest_tumor(self, origin_vox):
        '''returns the center of the nearest tumor to a coordinate'''
        origin_mm = np.array(origin_vox) * self.resolution
        return min(self.tumors,key=lambda t: np.linalg.norm(t["center"] - origin_mm))