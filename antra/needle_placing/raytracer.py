import numpy as np
from antra.segmentation.segmentator import Segmentation
from totalsegmentator.map_to_binary import class_map

class RayData:
    '''stores ray traversal data for all segmentation masks.'''
    def __init__(self, vector: tuple[float], tasks: list[str]):
        # ordered raydata: [{task: str, tissue: int, length: float}]
        self.ordered = []
        self.total_length = 0
        self.entry_voxel = None

        # per-mask label lengths: {"total": array[128], "body": array[2] ...}
        self.lengths = {task: np.zeros(len(class_map[task])+1, dtype=np.float32) for task in tasks}
        self.vector  = vector
        self.tasks   = tasks

    def accumulate(self, mask_name: str, value: float, length: float) -> None:
        '''add length to mask's length accumulator.'''
        # get label and relative(!) alpha from the value
        label = int(value)
        frac  = value - label
        alpha = 1.0 - frac
        self.lengths[mask_name][label] += length * alpha

        # add total length
        self.total_length += length

        # add ordered info
        if self.ordered is None: return
        self.ordered.append({"task":mask_name,"tissue": label,"length":length,"alpha":alpha})

class Raytracer():
    '''Raytracing using Amanatides-Woo.'''

    def __init__(self, ablation_center_dist: int, segmentations: dict[str,Segmentation]):
        # ray data defaults
        self.theta_range   = [0,2*np.pi] # offset, angle (both pos)
        self.phi_range     = [0,  np.pi] # offset, angle (both pos)

        # segmentation data 
        self.segmentations = segmentations
        self.tasks         = list(segmentations.keys())
        self.arrays        = {task: segmentations[task].get_array(np.float32) for task in self.tasks}
        self.bounds        = {"min":np.array([0,0,0]),"max":np.array(segmentations['total'].dicom.dimensions)}
        self.voxel_size    = segmentations['total'].dicom.resolution

        # needle / tumor data
        self.ablation_dist = ablation_center_dist
        self.origin        = (0,0,0)

    # setters for settings + ranges
    def set_origin(self, x, y, z) -> None:
        self.origin = np.array((x,y,z), dtype=float) # in mm

    def set_theta_range(self, lower: float, upper: float) -> None:
        self.theta_range = [lower, upper - lower]

    def set_phi_range(self, lower: float, upper: float) -> None:
        self.phi_range = [lower, upper - lower]

    def analyze_range(self, density: float = 250) -> list[RayData]:
        '''Analyzes paths fairly spread over the specified area. Density in rays / rad^2'''
        # generate directions
        srad = 4*np.pi * (self.theta_range[1]*self.phi_range[1]) / (np.pi**2*2)  
        directions = self.generate_n_directions(int(density * srad))
        results = [self.analyze_path(direction) for direction in directions]
        return directions, results

    def generate_n_directions(self, n: int) -> np.ndarray:
        '''Distribute n points over the allowed range of a sphere.'''
        fractional_range = (self.theta_range[1]) / (np.pi*2) # theta fraction
        total_num_points = int(n / fractional_range)
        golden_angle     = (3 - np.sqrt(5)) * np.pi                                                                           

        # z range is defined by phi
        z_min = np.cos(self.phi_range[1] + self.phi_range[0])
        z_max = np.cos(self.phi_range[0])

        # fibonnaci sphere math                            
        theta = np.mod(golden_angle * np.arange(total_num_points), 2 * np.pi)
        phi   = np.arccos(np.linspace(z_min, z_max, total_num_points))                             
  
        # # remove out of range
        angle_mask = (theta <= self.theta_range[1])
        theta = theta[angle_mask] + self.theta_range[0]
        phi = phi[angle_mask]

        # turn into vector
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack((x, y, z))

    def is_in_body(self, voxel: np.ndarray) -> bool:
        is_in_bounds  = (np.all(voxel >= self.bounds['min']) and np.all(voxel <  self.bounds['max']))
        if not is_in_bounds: return False
        return self.arrays['body'][voxel[0], voxel[1], voxel[2]] != 0

    def analyze_path(self, dir = np.ndarray) -> RayData:
        '''Traces a ray's traversed tissue types, using Amanatides-Woo.
           Returns a RayData object with all needed data for scoring.'''
        # initial values
        origin  = self.origin - dir * self.ablation_dist # accounts for needle extra length
        voxel   = np.floor(origin / self.voxel_size).astype(int)
        steps   = np.sign(dir).astype(int)
        bounds  = (voxel + (dir > 0).astype(int)) * self.voxel_size
        with np.errstate(divide='ignore', invalid='ignore'):
            t_max   = np.where(dir != 0, (bounds - origin)/dir,    np.inf)
            t_delta = np.where(dir != 0, self.voxel_size / np.abs(dir), np.inf)
        t = t_next = 0

        # accumulate voxels
        accumulator = RayData(dir, self.tasks)
        while(self.is_in_body(voxel)):
            # save entry voxel as last valid voxel
            accumulator.entry_voxel = voxel.copy()

            # pass last segment to accumulator
            t_next = np.min(t_max)
            length = t_next - t

            # pass to accumulator if in body
            self.accumulate(accumulator, voxel, length)

            # find next voxel using amanatides-woo
            if t_max[0] <= t_max[1] and t_max[0] <= t_max[2]: i = 0
            elif t_max[1] <= t_max[2]: i = 1
            else: i = 2
            t_next = t_max[i]
            voxel[i] += steps[i]
            t_max[i] += t_delta[i]

            t = t_next # update t

        return accumulator

    def accumulate(self, accumulator: RayData, voxel: tuple[int], length: float) -> None:
        '''Updates a ray's traversal list'''
        for task in self.tasks:
            # priority parsing, only the first non-zero value at this voxel is counted
            value = self.arrays[task][voxel[0],voxel[1],voxel[2]]
            if value == 0: continue
            accumulator.accumulate(task, value, length)
            break
        else:
            accumulator.accumulate('total', 0, length)