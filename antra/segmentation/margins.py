import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, find_objects
from totalsegmentator.map_to_binary import class_map

class MarginGenerator():
    '''expands a segmentation mask's labels by their configured margins.'''

    def __init__(self, task: str, cfg):
        self.task = task
        self.config = cfg
        self.margins = self.config.items(section='margins')
        self.window = {}

    def get_spherical_window(self, margin_mm: float, voxel_sizes: np.ndarray) -> np.ndarray:
        '''Build a boolean window that approximates a sphere
        of radius `margin_mm` in physical space around a point.'''
        # check if cached
        if margin_mm in self.window: return self.window[margin_mm]

        # find literal voxel margin (unique in each direction)
        voxel_margins = np.ceil(margin_mm / voxel_sizes).astype(int) + 1

        # build grid of literal voxel offsets from the middle
        ranges = [np.arange(-r, r + 1) for r in voxel_margins]
        grids = np.meshgrid(*ranges, indexing="ij")

        # convert voxel offsets to physical distance of nearest point and return a boolean mask
        nearest_points = sum((np.maximum(0, np.abs(grid) - 0.5) * size) ** 2 for grid, size in zip(grids, voxel_sizes))
        mask = nearest_points <= margin_mm ** 2
        self.window[margin_mm] = mask
        return mask

    def get_tissue_margins(self, tissue_label: int) -> tuple[int, float]:
        '''returns the tissue's margin and relative opacity'''
        # get the tissue's name
        tissue_name = class_map[self.task][tissue_label]  # full names (right_lung, rib_1)
        short_names = self.config.options('margins')  # shorter names (lung, rib)

        # body (special case) is not counted
        if 'body' in tissue_name: return 0, 0
        name = max([opt for opt in short_names if opt in tissue_name], key=len)

        return self.config.getmargin('margins', name)

    def apply_margins(self, img: nib.Nifti1Image) -> nib.Nifti1Image:
        '''returns a version of the mask with labels expanded by their given margin as set in margins.ini'''
        array = np.asarray(img.dataobj).astype(np.uint16)
        voxel_sizes = np.array(img.header.get_zooms()[:3])
        labels = set(np.unique(array)) - {0}
        objects = find_objects(array)  # scipy objects

        result = np.zeros_like(array, dtype=np.float32)
        for label in labels:
            # get this tissue's margin and it's relative opacity, check if valid
            margin, opacity = self.get_tissue_margins(label)
            if opacity == 0: continue
            if margin == 0: continue

            # get the margin mask and set it to the label plus the alpha's inverse
            margin_window = self.get_spherical_window(margin, voxel_sizes)
            obj_slice = objects[label - 1]

            # expand bounding box by the margins
            radius = np.array(margin_window.shape) // 2
            padded = []
            for dim_slice, r, dim_size in zip(obj_slice, radius, array.shape):
                start = max(0, dim_slice.start - r)
                stop  = min(dim_size, dim_slice.stop + r)
                padded.append(slice(start, stop))
            padded = tuple(padded)
            local_mask = (array[padded] == label)

            # dilate using created window
            dilated = binary_dilation(local_mask, structure=margin_window)
            result[padded][dilated] = label + 1 - opacity

        # Restore every voxel that had a label originally
        original_mask = (array != 0)
        result[original_mask] = array[original_mask]

        return nib.Nifti1Image(result, img.affine, img.header)
