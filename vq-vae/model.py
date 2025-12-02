# Model Code
# Authors: Samantha Mutiti & Rong Chi
import torch
import torch.nn as nn

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
    def __init__(self, activation='relu', embed_dim=512, input_size=300):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        activation = activation_dict[activation]
        self.input_size = input_size
        # inputs = 4x256x256
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, ...)
        self.conv_layers = nn.Sequential(
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, ...)
            nn.Conv2d(4, 8, 3), # new size = 8x254x254
            nn.MaxPool2d(2, 2), # new size = 8x127x127
            nn.BatchNorm2d(8),
            activation,
            nn.Conv2d(8, 16, 5), # 16x123x123
            nn.MaxPool2d(3, 3), # new size = 16x41x141
            nn.BatchNorm2d(16),
            activation,
            nn.Conv2d(16, 32, 7), # 32x35x35
            nn.MaxPool2d(5, 5), # new size = 32x7x7 (1568 features)
            nn.BatchNorm2d(32),
            activation,
        )
        self.final_spatial_size = self._calculate_conv_output_size(input_size)
        self.flattened_size = 32 * self.final_spatial_size * self.final_spatial_size
        # torch.nn.Linear(in_features, out_features, ...)
        self.body = nn.Sequential(
            nn.Linear(self.flattened_size, 784),
            activation,
            nn.Linear(784, embed_dim)
        )
    
    def _calculate_conv_output_size(self, input_size):
        """Calculate spatial size after all conv and pooling operations"""
        size = input_size
        size = size - 2  # First Conv2d(3x3)
        size = size // 2  # First MaxPool2d(2, 2)
        size = size - 4  # Second Conv2d(5x5)
        size = size // 3  # Second MaxPool2d(3, 3)
        size = size - 6  # Third Conv2d(7x7)
        size = size // 5  # Third MaxPool2d(5, 5)
        return size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flattened_size)  # Flatten the tensor for the linear layers
        x = self.body(x)
        return x

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, activation='relu', embedding_dim=512, flattened_size=1568):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        activation = activation_dict[activation]
        self.body = nn.Sequential(
            nn.Linear(embedding_dim, 784),
            activation,
            nn.Linear(784, flattened_size),
            activation
        )
        self.deconv_layers =  nn.Sequential(
            nn.ConvTranspose2d(32, 16, 7, stride=5, output_padding=4), 
            nn.BatchNorm2d(16), 
            activation, 
            nn.ConvTranspose2d(16, 8, 5, stride=3, output_padding=2), 
            nn.BatchNorm2d(8), 
            activation, 
            nn.ConvTranspose2d(8, 4, 3, stride=2, output_padding=1), 
            nn.Sigmoid())
    
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
        x = x.to(self.embeddings.weight.dtype)
        x_norm = torch.sum(x**2, dim=1, keepdim=True)  # (batch_size, 1)
        e_norm = torch.sum(self.embeddings.weight**2, dim=1)  # (num_embeddings,)
        dis_mat = x_norm + e_norm - 2 * torch.matmul(x, self.embeddings.weight.t())  # (batch_size, num_embeddings)
        # now argmin to get closest embedding
        embed_inds = torch.argmin(dis_mat, dim=1)  # (batch_size,)
        x_quantized = self.embeddings(embed_inds)  # (batch_size, embedding_dim)
        # compute losses
        codebook_loss = nn.MSELoss()(x_quantized.detach(), x)
        commitment_loss = nn.MSELoss()(x_quantized, x.detach()) * self.commitment_cost
        return x_quantized, codebook_loss + commitment_loss, embed_inds
    
    
# full class for VQ-VAE   
class CellVQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=512, commitment_cost=0.25, activation='relu'):
        super().__init__()
        # what is a good number of embeddings
        self.encoder = Encoder(activation=activation, embed_dim=embedding_dim)
        self.decoder = Decoder(activation=activation,embedding_dim=embedding_dim, flattened_size=self.encoder.flattened_size)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        x_encoded = self.encoder(x)  # Encode input to latent space
        x_q, vq_loss, _ = self.vq_layer(x_encoded)  # Vector quantization
        x_reconstructed = self.decoder(x_q)  # Decode quantized latent vectors
        return x_reconstructed, vq_loss, x_encoded
    
    def train(self, mode=True):
        super().train(mode)

    def train_step(self, x, optimizer, masks=None):
        self.train()
        optimizer.zero_grad()

        # forward pass
        x_reconstructed, vq_loss, _ = self.forward(x)
        
        # loss computation
        # loss = reconstruction loss + vq loss + commitment loss
        if masks is not None:
            reconstruction_loss = nn.MSELoss()(x_reconstructed * masks, x * masks) # add mask here (hopefully shapes work out :D)
        else:
            reconstruction_loss = nn.MSELoss()(x_reconstructed, x)

        loss = vq_loss + reconstruction_loss
        loss.backward()
        
        # optimizer step
        optimizer.step()

        return loss


# Understanding the loss computation for VQ-VAE
## Codebook Loss: This term updates the codebook embeddings (the \(e_{k}\) vectors). 
### pulls the chosen codebook vector (vq embedding) toward that output, so the
### codebook learns to better represent the encoder's outputs

## Commitment Loss: This term is crucial for training the encoder network. It uses a 
### stop-gradient operator on the vq embeddings and penalizes the encoder when 
### its output  drifts too far from the selected codebook entry. This encourages 
### the encoder to "commit" to specific, stable regions in the latent space

## Reconstruction Loss:
### typical encoder loss for properly reconstructing the images
    
# References:
### GitHub Implementation: https://github.com/MishaLaskin/vqvae
### Original Paper: https://arxiv.org/abs/1711.00937
### Reference Article: https://huggingface.co/blog/ariG23498/understand-vq
