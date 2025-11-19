# Model Code
# Authors: Samantha Mutiti & Rong Chi
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# Notes:
# Crop Statistics (dimensions)
### r --> max: 214 min: 15
### c --> max: 202 and min: 14
### ideal padding ~ 256
# Architecture Design
''' Iteration 1  --> test with basic architecture
num layers = 3, latet dim = 512
'''

activation_dict = {'relu': F.relu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}

class Encoder(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        if activation not in ['relu', 'tanh', 'sigmoid']:
            raise ValueError(f'activation {activation} not supported')
        activation = activation_dict[activation]
        # inputs = 4x256x256
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, ...)
        self.conv_layers = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, ...)
            nn.Conv2d(4, 8, 3), # new size = 8x254x254
            nn.MaxPool2d(2, 2), # new size = 8x127x127
            activation,
            nn.Conv2d(8, 16, 5), # 16x123x123
            nn.MaxPool2d(3, 3), # new size = 16x41x141
            activation,
            nn.Conv2d(16, 32, 7), # 32x35x35
            nn.MaxPool2d(5, 5), # new size = 32x7x7 (1568 features)
            # torch.nn.Linear(in_features, out_features, ...)
            activation,
        )
        self.body = nn.Sequential(
            nn.Linear(32 * 7 * 7, 784),
            activation,
            nn.Linear(784, 512)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor for the linear layers
        x = self.body(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(512, 784),
            nn.ReLU(),
            nn.Linear(784, 32 * 7 * 7),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
            nn.ConvTranspose2d(32, 16, 7, stride=5), # new size = 16x35x35
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3), # new size = 8x123x123
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2), # new size = 4x247x247 (close to 256x256)
            nn.Sigmoid() # to ensure output is between [0, 1]
        )
    
    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 32, 7, 7)  # Reshape to (batch_size, 32, 7, 7)
        x = self.deconv_layers(x)
        return x
