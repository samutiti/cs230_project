# Utilities Code
# Authors: Samantha Mutiti & Rong Chi
import torch
from torch import nn

train_dataset_mean = 0
train_dataset_std = 1 # need to compute and replace with true values

def normalize_input(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize the input tensor to have zero mean and unit variance.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = x.mean(dim=0, keepdim=True) # we may need to change the dim to get a single value
    std = x.std(dim=0, keepdim=True) + eps
    return (x - mean) / std

def normalize_input_01(x:torch.Tensor):
    """ 
    Normalize the input from 0 to 1
    Args:
        x (torch.Tensor): input tensor of shape (batch, channels, h, w)
    """
    x = x.float()
    x = x - torch.min(x) # now ranges from 0 to max - min
    x = x / torch.max(x) # now ranges from 0 to 1
    return x

def normalize_input_zscore(x:torch.Tensor):
    """ 
    Z-score normalization using precomputed dataset mean and std
    Args:
        x (torch.Tensor): input tensor of shape (batch, channels, h, w)
    """
    global train_dataset_mean, train_dataset_std
    x = x.float()
    x = (x - train_dataset_mean) / train_dataset_std
    return x


def compute_dataset_statistics(dataset: torch.utils.data.Dataset):
    """
    Compute the mean and standard deviation of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to compute statistics for """
    all_data = [] # dataset returns image, mask, filename
    for i in range(len(dataset)):
        image, _, _ = dataset[i]
        all_data.append(image)
    all_data = torch.stack(all_data)
    mean = torch.mean(all_data, dim=[0, 2, 3]) # mean over batch, height, width
    std = torch.std(all_data, dim=[0, 2, 3]) # std over batch, height, width
    return mean, std