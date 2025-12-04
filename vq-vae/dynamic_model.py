# Dynamic Model Code with Variable Input Size Handling
# Authors: Samantha Mutiti & Rong Chi (Modified for dynamic sizing)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

activation_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}

# Dynamic Encoder Class - handles variable input sizes
class DynamicEncoder(nn.Module):
    def __init__(self, activation='relu', embed_dim=512):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        self.activation = activation_dict[activation]
        
        # Dynamic convolutional layers
        self.conv1 = nn.Conv2d(4, 8, kernel_size=7, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        
        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Always pool to 4x4
        
        # Fixed size after adaptive pooling: 64 * 4 * 4 = 1024
        self.body = nn.Sequential(
            nn.Linear(1024, 784),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(784, 512),
            self.activation,
            nn.Linear(512, embed_dim)
        )
    
    def forward(self, x):
        # Apply conv layers with conditional batch norm
        x = self.conv1(x)
        if x.size(2) > 1 and x.size(3) > 1:  # Only apply BN if spatial size > 1
            x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn3(x)
        x = self.activation(x)
        
        x = self.conv4(x)
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn4(x)
        x = self.activation(x)
        
        x = self.conv5(x)
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn5(x)
        x = self.activation(x)
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)  # Always outputs (batch, 64, 4, 4)
        
        # Flatten and apply linear layers
        x = x.view(-1, 1024)
        x = self.body(x)
        return x

# Dynamic Decoder Class - reconstructs to original input size
class DynamicDecoder(nn.Module):
    def __init__(self, activation='relu', embedding_dim=512):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        self.activation = activation_dict[activation]
        
        # Linear layers to expand embedding
        self.body = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(512, 784),
            self.activation,
            nn.Linear(784, 1024),  # 64 * 4 * 4
            self.activation
        )
        
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.deconv2 = nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        
        self.deconv3 = nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(8)
        
        self.final_conv = nn.Conv2d(8, 4, kernel_size=3, padding=1)
    
    def forward(self, x, target_size):
        # Expand embedding
        x = self.body(x)
        x = x.view(-1, 64, 4, 4)
        
        # Apply deconvolutional layers with conditional batch norm
        x = self.deconv1(x)  # 4x4 -> 8x8
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn1(x)
        x = self.activation(x)
        
        x = self.deconv2(x)  # 8x8 -> 16x16
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn2(x)
        x = self.activation(x)
        
        x = self.deconv3(x)  # 16x16 -> 32x32
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn3(x)
        x = self.activation(x)
        
        x = self.deconv4(x)  # 32x32 -> 64x64
        if x.size(2) > 1 and x.size(3) > 1:
            x = self.bn4(x)
        x = self.activation(x)
        
        # Use interpolation to match target size exactly
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Final convolution to get 4 channels
        x = self.final_conv(x)
        return x

# Vector Quantizer Class (same as original)
class VectorQuantizer(nn.Module):
    def __init__(self, num_e, embed_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_e
        self.embedding_dim = embed_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        x = x.to(self.embeddings.weight.dtype)
        x_norm = torch.sum(x**2, dim=1, keepdim=True)
        e_norm = torch.sum(self.embeddings.weight**2, dim=1)
        dis_mat = x_norm + e_norm - 2 * torch.matmul(x, self.embeddings.weight.t())
        embed_inds = torch.argmin(dis_mat, dim=1)
        x_quantized = self.embeddings(embed_inds)
        codebook_loss = nn.MSELoss()(x_quantized.detach(), x)
        commitment_loss = nn.MSELoss()(x_quantized, x.detach()) * self.commitment_cost
        return x_quantized, codebook_loss + commitment_loss, embed_inds

# Dynamic VQ-VAE with adaptive padding and mask-weighted loss
class DynamicCellVQVAE(nn.Module):
    def __init__(self, activation='relu', embedding_dim=512, commitment_cost=0.25, 
                 reconstruction_loss_weight=1.0, num_embeddings=1024, min_size=32):
        super().__init__()
        self.encoder = DynamicEncoder(activation=activation, embed_dim=embedding_dim)
        self.decoder = DynamicDecoder(activation=activation, embedding_dim=embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.commitment_cost = commitment_cost
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.min_size = min_size
        
    def adaptive_padding(self, x, masks=None):
        """Apply minimal padding to make input size efficient for convolutions"""
        batch_size, channels, h, w = x.shape
        
        # Calculate next power of 2 or minimum size
        target_h = max(self.min_size, 2**math.ceil(math.log2(h)))
        target_w = max(self.min_size, 2**math.ceil(math.log2(w)))
        
        # Calculate padding needed
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Apply center padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad input
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Pad masks if provided
        masks_padded = None
        if masks is not None:
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            masks_padded = F.pad(masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Store original size and padding info for reconstruction
        padding_info = {
            'original_size': (h, w),
            'target_size': (target_h, target_w),
            'padding': (pad_top, pad_bottom, pad_left, pad_right)
        }
        
        return x_padded, masks_padded, padding_info
    
    def remove_padding(self, x, padding_info):
        """Remove padding to return to original size"""
        pad_top, pad_bottom, pad_left, pad_right = padding_info['padding']
        h_orig, w_orig = padding_info['original_size']
        
        # Remove padding
        x = x[:, :, pad_top:pad_top+h_orig, pad_left:pad_left+w_orig]
        return x
    
    def forward(self, x):
        # Store original size for reconstruction
        original_size = x.shape[2:]
        
        # Apply adaptive padding
        x_padded, _, padding_info = self.adaptive_padding(x)
        target_size = x_padded.shape[2:]
        
        # Encode
        x_encoded = self.encoder(x_padded)
        
        # Vector quantization
        x_q, vq_loss, embed_inds = self.vq_layer(x_encoded)
        
        # Decode to target size, then remove padding
        x_reconstructed = self.decoder(x_q, target_size)
        x_reconstructed = self.remove_padding(x_reconstructed, padding_info)
        
        return x_reconstructed, vq_loss, x_encoded, embed_inds
    
    def create_content_mask(self, masks, padding_info):
        """Create a content mask that emphasizes cell regions and de-emphasizes padding"""
        if masks is None:
            return None
            
        # Ensure masks are float type for comparison
        masks = masks.float()
        
        # Create binary mask for cell content (any cell vs background)
        cell_mask = (masks > 0).float()
        
        # Calculate content density to weight loss appropriately
        content_ratio = cell_mask.mean()
        
        # Avoid division by zero and create balanced weighting
        content_weight = torch.where(cell_mask > 0, 1.0 / (content_ratio + 1e-8), 0.1)
        
        return content_weight
    
    def train_step(self, x, optimizer, masks=None, gradient_clip_norm=None, scheduler=None):
        self.train()
        optimizer.zero_grad()
        
        # Apply adaptive padding
        x_padded, masks_padded, padding_info = self.adaptive_padding(x, masks)
        
        # Forward pass
        x_reconstructed, vq_loss, _, _ = self.forward(x)
        
        # Create content-aware loss weighting
        if masks is not None:
            content_weight = self.create_content_mask(masks, padding_info)
            
            # Apply content weighting to reconstruction loss
            weighted_diff = (x_reconstructed - x) * content_weight
            reconstruction_loss = torch.mean(weighted_diff ** 2)
        else:
            reconstruction_loss = F.mse_loss(x_reconstructed, x)
        
        # Apply weighting to balance loss terms
        weighted_reconstruction_loss = self.reconstruction_loss_weight * reconstruction_loss
        loss = weighted_reconstruction_loss + vq_loss
        
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        return loss, reconstruction_loss, vq_loss