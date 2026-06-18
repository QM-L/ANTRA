import json
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.orientations as orient
import torch

from totalsegmentator.python_api import totalsegmentator
from antra.general import config
from antra.general.dicom_object import DICOM_Scan
from antra.segmentation.margins import MarginGenerator

class Segmentation():
    '''Creates a mask of a specified segmentation'''
    def __init__(self, dicom=None, task: str = "total", folder: str = None, load=False):
        self.task = task
        self.config = config.load_configs()
        self.dicom = dicom
        self.margin_generator = MarginGenerator(self.task, self.config)

        # load data: no dicom, try to load from folder
        if not load: self.mask = self.generate_mask(folder)
        elif folder: self.mask, self.dicom = self.load_mask(folder)
        else: raise(ValueError("no dicom or folder specified for segmentation"))

        self.raw_mask = self.floored_mask()

    def generate_mask(self, folder_name) -> nib.Nifti1Image:
        '''generates a segmentation and exports it to data folder.'''

        # generate file name and location
        folder    = Path(__file__).parent.parent.parent / f'data/{folder_name}'
        file_path = folder / f'mask_{self.task}.nii.gz'
        meta_path = folder / f'dicom_data.json'

        # generate
        file_path.parent.mkdir(parents=True, exist_ok=True)
        device = "gpu" if torch.cuda.is_available() else "cpu"
        raw_image = totalsegmentator(input=self.dicom.path, task=self.task, output=None, ml=False, device=device)
        processed_image = self.margin_generator.apply_margins(self.ensure_lps(raw_image))
        nib.save(processed_image, file_path.absolute())

        # json saved
        with open(meta_path, 'w') as f:
            json.dump({"dicom": self.dicom.path}, f, indent=2)

        return processed_image

    def load_mask(self, folder: str) -> tuple[nib.Nifti1Image, DICOM_Scan]:
        '''Load mask and dicom properties from a folder.'''
        folder    = Path(folder)
        mask_path = folder / f'mask_{self.task}.nii.gz'
        meta_path = folder / f'dicom_data.json'

        if not mask_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No full segmentation data found in {folder}, please rerun the segmentation.")

        with open(meta_path) as f:
            meta = json.load(f)

        # return mask AND either the existing dicom or a new dicom if none was loaded yet
        return (nib.load(mask_path), self.dicom or DICOM_Scan(meta['dicom']))

    def ensure_lps(self, mask: nib.Nifti1Image) -> nib.Nifti1Image:
        '''converts image to LPS orientation'''
        data = mask.get_fdata()
        affine = mask.affine

        # compute transform from orientations
        start_orient  = orient.io_orientation(affine)
        target_orient = orient.axcodes2ornt(('L', 'P', 'S'))
        transform     = orient.ornt_transform(start_orient, target_orient)

        # reorient data & update affine
        reoriented = orient.apply_orientation(data, transform)
        new_affine = affine @ orient.inv_ornt_aff(transform, data.shape)

        return nib.Nifti1Image(reoriented, new_affine)

    def get_array(self, valtype: type = np.uint16) -> np.ndarray:
        return np.asarray(self.mask.dataobj).astype(valtype)

    def set_array(self, new_array) -> nib.Nifti1Image:
        '''manually edits the array of the nifti1Image. only used for debugging'''
        if not (new_array.shape == self.get_array().shape): raise IndexError(f"New array shape {self.get_array().shape} does not match old array shape {new_array.shape}.")
        self.mask = nib.Nifti1Image(new_array, self.mask.affine, self.mask.header)
        return self.mask

    def floored_mask(self) -> nib.Nifti1Image:
        '''returns the floored version of the mask i.e. for visualization.'''
        array = np.asarray(self.mask.dataobj).astype(np.float32)
        result = np.floor(array).astype(np.uint16)
        return nib.Nifti1Image(result, self.mask.affine, self.mask.header)