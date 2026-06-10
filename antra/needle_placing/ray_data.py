import numpy as np
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