import numpy as np

from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops

class NeedleAdvisor():
    '''Actually goes through the scores and finds the best needle path'''

    def __init__(self, config, weighted_scores: list[dict]):
        # load relevant config values
        self.threshold = config.getfloat('scoring', 'score_threshold')
        self.min_patch_size = config.getint('scoring', 'min_patch_size')
        self.results = weighted_scores

    def advise(self) -> list[dict]:
        '''turns list of {"theta", "phi", "scores", "entry_voxel"} for each ray
           into a list of {"dir", "score"} for each viable patch center'''

        # generate grid with the weighted scores
        t = np.array([r['theta'] for r in self.results])
        p = np.array([r['phi']   for r in self.results])
        total  = np.array([r['weighted_score'] for r in self.results])

        # score a finer coordinate space with interpolation
        res    = 500
        thetas = np.linspace(t.min(), t.max(), res)
        phis   = np.linspace(p.min(), p.max(), res)
        mesh_theta, mesh_phi = np.meshgrid(thetas, phis)
        grid_scores = griddata(points=np.column_stack([t, p]), values=total, xi=(mesh_theta,mesh_phi), method='linear', fill_value=0)

        # find patches with the threshold
        threshold_value = self.threshold * grid_scores.max()
        binary = grid_scores >= threshold_value
        labelled,_ = label(binary)
        advice      = []

        # go through each patch
        for region in regionprops(labelled):
            # continue if region is too small, else, find center of region
            if region.num_pixels < self.min_patch_size: continue
            center = self.get_region_center(region, labelled)

            advice.append({
                "theta": float(thetas[center[1]]),
                "phi": float(phis[center[0]]),
                "score": float(grid_scores[center]),
                "area": float(region.num_pixels/res)
            })

        return sorted(advice, key=lambda a: a["area"], reverse=True)

    def get_region_center(self, region, labelled_array):
        # each coordinate is valued with distance from edge
        patch  = labelled_array == region.label
        dist   = distance_transform_edt(patch)

        # get patch with largest distance and append
        return np.unravel_index(dist.argmax(), dist.shape)
        