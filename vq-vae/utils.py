# Utilities Code
# Authors: Samantha Mutiti & Rong Chi
import torch
from torch import nn

# For Macrophage_dataset (R. Chi and S. Mutiti, 2025)
## Dataset Mean: tensor([ 24.3902, 101.4538,  14.4291,  14.5543]), Dataset Std: tensor([223.8065, 701.5453, 135.7977, 105.5642])
## Per channel mean and stdev
train_dataset_mean = torch.tensor([ 24.3902, 101.4538,  14.4291,  14.5543]).view(1, -1, 1, 1)
train_dataset_std = torch.tensor([223.8065, 701.5453, 135.7977, 105.5642]).view(1, -1, 1, 1)

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


def compute_dataset_statistics(dataset: torch.utils.data.Dataset, batch_size: int = 32):
    """
    Compute the mean and standard deviation of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to compute statistics for """
    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    total_pixels = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for images, _, _ in dataloader:
        # Sum all pixel values across batch, height, width
        sum_pixels += torch.sum(images, dim=[0, 2, 3])
        sum_squared_pixels += torch.sum(images**2, dim=[0, 2, 3])
        total_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    
    # Compute final statistics
    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean**2)
    std = torch.sqrt(variance)
    
    return mean, std