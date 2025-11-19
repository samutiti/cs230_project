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
    def __init__(self, label_file_path:str, data_directory:str, transforms:nn.Sequential=None):
        """
        Custom Dataset for loading crop data.

        Args:
            data (list of tuples): List where each tuple contains (image, filename).
            data_directory (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = self.read_data_json(label_file_path)
        self.data_directory = data_directory
        self.transforms = transforms # normalization will occur in dataloader such that it can be applied to entire training set post-split

    def read_data_json(self, file_path):
        ''' reads data from json file
        Parameters:
            file_path(str): string of path where label json is stored
            ****EXPECTED FORMAT OF JSON FILE****
                    {file_name(str): labels(list)}
        '''
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        ''' returns length of the dataset '''
        return len(self.data)

    def __getitem__(self, idx):
        ''' retrieves item at idx
        Parameters:
            idx(int): index of data item to return'''
        image_filename, label = self.data[idx]
        image_array = np.array(tf.imread(os.path.join(self.data_directory, image_filename)))
        if self.transforms:
            image = self.transforms(image_array)
        return image, label

class CropDataLoader(DataLoader):
    def __init__(self, dataset, bacth_size, type:int, normalize:bool=True):
        ''' Dataloader
        Parameters:
            type: (int) = 0 (train), 1(validation), or 2(test) 
        '''
        self.dataset = normalize_input(dataset) if normalize else dataset
        self.type = ['train', 'val', 'test'][type]
