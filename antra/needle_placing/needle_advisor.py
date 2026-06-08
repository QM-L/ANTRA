import numpy as np

from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops

from antra.needle_placing.raytracer import RayData

class NeedleAdvisor():
    '''Collector for unmodified ray scores, stores all methods for
       advising of needle directions.'''

    def __init__(self, config):
        # load relevant config values
        self.weights   = config.gettuple('scoring', 'weights')
        self.threshold = config.getfloat('scoring', 'score_threshold')
        self.min_patch_size = config.getint('scoring', 'min_patch_size')

    def advise(self, ray_results: list[RayData]) -> list[dict]:
        '''turns a list of RayData, each with .theta, .phi and per-scorer scores
           into a list of {"theta", "phi", "score"} for each viable patch center'''

        # generate grid with the scores
        thetas = np.array([r.theta for r in ray_results])
        phis   = np.array([r.phi   for r in ray_results])
        total  = np.zeros(len(ray_results))

        weight_sum = sum(self.weights.values())
        for key, w in self.weights.items():
            scores = np.array([r.scores[key] for r in ray_results])
            total += (w / weight_sum) * scores

        # score a finer coordinate space with interpolation
        res    = 500
        thetas = np.linspace(thetas.min(), thetas.max(), res)
        phis   = np.linspace(phis.min(),   phis.max(),   res)
        mesh_theta, mesh_phi = np.meshgrid(thetas, phis)
        grid_scores = griddata(points = np.column_stack([thetas, phis]), values = total, xi = (mesh_theta, mesh_phi), method = 'linear', fill_value = 0)

        # find patches with the threshold
        threshold_value = self.threshold * grid_scores.max()
        binary = grid_scores >= threshold_value
        labelled,_ = label(binary)
        advice      = []

        # go through each patch
        for region in regionprops(labelled):
            if region.num_pixels < self.min_patch_size: continue

            # each coordinate is valued with distance from edge
            patch  = labelled == region.label
            dist   = distance_transform_edt(patch)

            # get patch with largest distance and append
            center = np.unravel_index(dist.argmax(), dist.ashape)
            advice.append({"theta": float(thetas[center[1]]),"phi":   float(phis[center[0]]),"score": float(grid_scores[center])})

        return sorted(advice, key=lambda a: a["score"], reverse=True)