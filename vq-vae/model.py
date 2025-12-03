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
            # Layer 1: 
            nn.Conv2d(4, 8, kernel_size=7, stride=3, padding=1),
            nn.MaxPool2d(1, 1),  # No additional pooling
            nn.BatchNorm2d(8),
            activation,
            
            # Layer 2:
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(1, 1),  # No additional pooling
            nn.BatchNorm2d(16),
            activation,
            
            # Layer 3:
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(1, 1),  # No additional pooling
            nn.BatchNorm2d(24),
            activation,
            
            # Layer 4:
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=0),
            nn.MaxPool2d(1, 1),  # No additional pooling
            nn.BatchNorm2d(32),
            activation,
            
            # Layer 5:
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(1, 1),  # No additional pooling
            nn.BatchNorm2d(64),
            activation,
        )

        self.final_spatial_size = self._calculate_conv_output_size(input_size)
        self.flattened_size = 64 * self.final_spatial_size * self.final_spatial_size
        # torch.nn.Linear(in_features, out_features, ...)
        self.body = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            activation,
            nn.Dropout(0.2), # added dropout layer
            nn.Linear(1024, 784), # extra linear layer
            activation,
            nn.Linear(784, embed_dim)
        )
    
    def _calculate_conv_output_size(self, input_size):
        """Calculate spatial size after all conv and pooling operations"""
        size = input_size
        size = (size - 7 + 2*1) // 3 + 1  # Layer 1: 300 -> 100
        size = (size - 3 + 2*1) // 2 + 1  # Layer 2: 100 -> 50
        size = (size - 3 + 2*1) // 2 + 1  # Layer 3: 50 -> 25
        size = (size - 3 + 2*0) // 2 + 1  # Layer 4: 25 -> 12
        size = (size - 3 + 2*1) // 2 + 1  # Layer 5: 12 -> 6

        return size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flattened_size)  # Flatten the tensor for the linear layers
        x = self.body(x)
        return x

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, activation='relu', embedding_dim=512, flattened_size=2304, final_spatial_size=6):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        activation = activation_dict[activation]
        self.flattened_size = flattened_size
        self.final_spatial_size = final_spatial_size
        self.body = nn.Sequential(
            nn.Linear(embedding_dim, 784),
            activation,
            nn.Dropout(0.2), # added dropout layer regularizer
            nn.Linear(784, 1024),
            activation,
            nn.Linear(1024, self.flattened_size),
            activation
        )
        print('INFO: Decoder configured to produce 300x300 output from 6x6 input')
        self.deconv_layers = nn.Sequential(
            # 6x6 → 12x12
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            activation,
            
            # 12x12 → 25x25
            nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(24),
            activation,
            
            # 25x25 → 50x50
            nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            activation,
            
            # 50x50 → 99x99
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            activation,
            
            # 99x99 → 300x300
            nn.ConvTranspose2d(8, 4, kernel_size=7, stride=3, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 64, self.final_spatial_size, self.final_spatial_size)  # Reshape to (batch_size, 64, 7, 7)
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
        self.decoder = Decoder(activation=activation,
                               embedding_dim=embedding_dim, 
                               # transfer encoder data
                               flattened_size=self.encoder.flattened_size, 
                               final_spatial_size=self.encoder.final_spatial_size)
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
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)  # add channel dimension
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
