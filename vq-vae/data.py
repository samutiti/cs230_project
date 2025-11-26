# Dataset & Loaders
# Authors: Samantha Mutiti & Rong Chi
import torch
import torch.nn as nn
import json, os
import tifffile as tf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from utils import *

class CropDataset(Dataset):
    def __init__(self, file_dir:str, type:str='train', transforms:nn.Sequential=None):
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
        self.transforms = transforms # normalization will occur in dataloader such that it can be applied to entire training set post-split
    
    def __len__(self):
        ''' returns length of the dataset '''
        return len(self.file_list)

    def __getitem__(self, idx):
        ''' retrieves item at idx
        Parameters:
            idx(int): index of data item to return'''
        image_filename = self.file_list[idx]
        image = np.array(tf.imread(os.path.join(self.data_directory, image_filename)))
        if self.transforms:
            image = self.transforms(image)
        return image, image_filename