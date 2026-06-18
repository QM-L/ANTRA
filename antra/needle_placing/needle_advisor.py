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

        # calculate step sizes
        theta_step = (t.max()-t.min())/(res-1)
        phi_step   = (p.max()-p.min())/(res-1)

        # find patches with the threshold
        threshold_value = self.threshold * grid_scores.max()
        binary = grid_scores >= threshold_value
        labelled,_ = label(binary)
        advice      = []

        # go through each patch
        for region in regionprops(labelled):
            # continue if region is too small, else, find center of region
            if region.num_pixels < self.min_patch_size: continue
            radius, center = self.get_region_center_data(region, labelled, theta_step, phi_step)
            # convert pixel radius into angle

            advice.append({
                "theta": float(thetas[center[1]]),
                "phi": float(phis[center[0]]),
                "score": float(grid_scores[center]),
                "angle": radius
            })

        return sorted(advice, key=lambda a: a["angle"], reverse=True)

    def get_region_center_data(self, region, labelled_array, theta_step, phi_step):
        # each coordinate is valued with distance from edge
        patch  = labelled_array == region.label
        dist   = distance_transform_edt(patch, sampling=(phi_step, theta_step))

        # get position with largest coordinate and radius
        radius = dist.max()
        position = np.unravel_index(dist.argmax(), dist.shape)
        return radius, position