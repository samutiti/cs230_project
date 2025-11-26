# Utilities Code
# Authors: Samantha Mutiti & Rong Chi
import torch
from torch import nn


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
    x = x - torch.min(x) # now ranges from 0 to max - min
    x = x / torch.max(x) # now ranges from 0 to 1
    return x
