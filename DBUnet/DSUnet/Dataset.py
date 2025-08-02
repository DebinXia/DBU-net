from torch.utils import data
import torch
import os
import numpy as np
from Registration.DL_Alignment.VM.VoxelMorph.Datasets.Preprocess import data_augment
import SimpleITK as sitk


class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, image_root):
        'Initialization'
        self.list_IDs = list_IDs
        self.data_root = image_root
        self.transformers = data_augment()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        ID = self.list_IDs[index]
        fixed_image = torch.Tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, ID + '_1.tif'))).astype(np.float64))
        moving_image = torch.Tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_root, ID + '_2.tif'))).astype(np.float64))

        fixed_image = fixed_image.unsqueeze(dim=0)  # self.transformers(fixed_image.unsqueeze(dim=0))
        moving_image = moving_image.unsqueeze(dim=0)  # self.transformers(moving_image.unsqueeze(dim=0))

        return fixed_image, moving_image, # None  # .permute([3, 0, 1, 2])

