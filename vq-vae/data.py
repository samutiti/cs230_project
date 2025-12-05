# Dataset & Loaders
# Authors: Samantha Mutiti & Rong Chi
import torch
import torch.nn as nn
import os, math
import tifffile as tf
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils import *

class CropDataset(Dataset):
    def __init__(self, file_dir:str, type:str='train'):
        """
        Custom Dataset for loading crop data.

        Args:
            data (list of tuples): List where each tuple contains (image, filename).
            data_directory (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        EXPECTED FILE FORMATTING:
            - file_dir/
                - train/
                    - image1.tif
                    - image2.tif
                    - ...
                - val/
                    - image1.tif
                    - image2.tif
                    - ...
                - test/
                    - image1.tif
                    - image2.tif
                    - ...
        """
        
        self.type = type
        self.data_directory = os.path.join(file_dir, type)
        self.file_list = os.listdir(self.data_directory)
        # for now I will hard code transforms :)
         # normalization will occur in dataloader such that it can be applied to entire training set post-split
    
    def __len__(self):
        ''' returns length of the dataset '''
        return len(self.file_list)

    def __getitem__(self, idx, crop_dim=300):
        ''' retrieves item at idx
        Parameters:
            idx(int): index of data item to return'''
        image_filename = self.file_list[idx]
        image = np.array(tf.imread(os.path.join(self.data_directory, image_filename)))
        
        # Handle different data types - convert uint16 to float32
        if image.dtype == np.uint16:
            # Convert uint16 to float32 and normalize to [0, 1] range
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            # Convert uint8 to float32 and normalize to [0, 1] range
            image = image.astype(np.float32) / 255.0
        else:
            # Ensure float32 for other types
            image = image.astype(np.float32)
        
        # Convert to tensor manually (avoid ToTensor issues with uint16)
        image = torch.from_numpy(image)
        
        if image.shape[0] > 5: # ensure CxHxW format (no ims should have dimension 4 or 5 in H or W)
            image = image.permute(2, 0, 1) # reshape to CxHxW
        # Perform center padding to 256x256
        pad_dims = max(crop_dim - image.shape[2], 0), max(crop_dim - image.shape[1], 0)
        pad_dims = math.ceil(pad_dims[0] / 2), math.ceil(pad_dims[1] / 2) # ceil the division so that we can do center padding
        image = nn.functional.pad(image, (pad_dims[0], pad_dims[0], pad_dims[1], pad_dims[1], 0, 0))  # CxHxW images Pad to 256x256
        image = image[:4, :crop_dim, :crop_dim]  # Crop to 256x256 if larger (none from this dataset should be, if there are any we may adjust the default size)
        mask = image[-1, :crop_dim, :crop_dim]
        return image.to(dtype=torch.float32), mask, image_filename