from pathlib import Path
import hashlib
import numpy as np
import SimpleITK as itk
from SimpleITK import Image
import nibabel as nib

class DICOM_Scan:
    # small reading / dataclass for reading and returning the ct scan and it's properties

    def __init__(self, path: str):
        # get image and properties
        self.path       = path
        self.image      = self.read_DICOM(path)
        self.array      = itk.GetArrayFromImage(self.image)
        self.origin     = self.image.GetOrigin()
        self.dimensions = self.image.GetSize()
        self.resolution = self.image.GetSpacing()
        self.id         = self.image_hash()

        print(f'loaded dicom with hash {self.id}.')

    def read_DICOM(self, input_path: str) -> Image:
        # read dicom file / files
        reader = itk.ImageSeriesReader()
        path = Path(input_path)

        if path.is_dir():
            dicom_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_names)
            return reader.Execute()
        elif path.is_file():
            return itk.ReadImage(input_path)
        else:
            raise FileNotFoundError(f"No DICOM found at {path.absolute()}")

    def image_hash(self) -> str:
        '''Generates a unique hash (16 characters) which works as identifier for caching.'''
        id = hashlib.md5(self.array.tobytes(), usedforsecurity=False)
        return id.hexdigest()[:16]