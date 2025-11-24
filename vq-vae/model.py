# Model Code
# Authors: Samantha Mutiti & Rong Chi
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

activation_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}

# Encoder Class
class Encoder(nn.Module):
    def __init__(self, activation='relu', embed_dim=512):
        super().__init__()
        if activation not in activation_dict:
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
            nn.Linear(784, embed_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor for the linear layers
        x = self.body(x)
        return x

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, activation='relu', embedding_dim=512):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        activation = activation_dict[activation]
        self.body = nn.Sequential(
            nn.Linear(embedding_dim, 784),
            activation,
            nn.Linear(784, 32 * 7 * 7),
            activation
        )
        self.deconv_layers = nn.Sequential(
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
            nn.ConvTranspose2d(32, 16, 7, stride=5), # new size = 16x35x35
            activation,
            nn.ConvTranspose2d(16, 8, 5, stride=3), # new size = 8x123x123
            activation,
            nn.ConvTranspose2d(8, 4, 3, stride=2), # new size = 4x247x247 (close to 256x256)
            nn.Sigmoid() # to ensure output is between [0, 1]
        )
    
    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 32, 7, 7)  # Reshape to (batch_size, 32, 7, 7)
        x = self.deconv_layers(x)
        return x

# Vector Quantizer Class
class VectorQuantizer(nn.Module):
    def __init__(self, num_e, embed_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_e
        self.embedding_dim = embed_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings) # uniform distribution w/ mean of 0

    def forward(self, x):
        ''' x: (batch_size, embedding_dim) - from Encoder output 
        embeddings: (num_embeddings, embedding_dim) - embedding weights'''
        # need to calulate distanced between x and embedding weights
        # Calculate distances
        x = torch.tensor(x, dtype=self.embeddings.weight.dtype)
        x_norm = torch.sum(x**2, dim=1, keepdim=True)  # (batch_size, 1)
        e_norm = torch.sum(self.embeddings.weight**2, dim=1)  # (num_embeddings,)
        dis_mat = x_norm + e_norm - 2 * torch.matmul(x, self.embeddings.weight.t())  # (batch_size, num_embeddings)
        # now argmin to get closest embedding
        embed_inds = torch.argmin(dis_mat, dim=1)  # (batch_size,)
        x_quantized = self.embeddings(embed_inds)  # (batch_size, embedding_dim)
        return x_quantized, self.compute_loss(x, x_quantized), embed_inds
    
    def compute_loss(self, x, x_quantized):
        # TODO: need to write loss function @samutiti
        pass
    
# full class for VQ-VAE   
class CellVQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=512, commitment_cost=0.25, activation='relu'):
        super().__init__()
        # what is a good number of embeddings
        self.encoder = Encoder(activation=activation, embed_dim=embedding_dim)
        self.decoder = Decoder(activation=activation,embedding_dim=embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        x_encoded = self.encoder(x)  # Encode input to latent space
        x_q, vq_loss, encoding_indices = self.vq_layer(x_encoded)  # Vector quantization
        x_reconstructed = self.decoder(x_q)  # Decode quantized latent vectors
        return x_reconstructed, vq_loss, encoding_indices
    
# references:
## GitHub Implementation: https://github.com/MishaLaskin/vqvae
## Original Paper: https://arxiv.org/abs/1711.00937
## Reference Article: https://huggingface.co/blog/ariG23498/understand-vq
