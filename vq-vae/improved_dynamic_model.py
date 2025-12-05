# Improved Dynamic Model with Enhanced Architecture and Loss Computation
# Authors: Samantha Mutiti & Rong Chi (Enhanced version)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

activation_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'gelu': nn.GELU()}

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and skip connections"""
    def __init__(self, channels, activation='relu'):
        super().__init__()
        self.activation = activation_dict[activation]
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.activation(x)
        return x

class AttentionBlock(nn.Module):
    """Self-attention block for better feature preservation"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        
        # Attention
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Skip connection with learnable weight
        out = self.gamma * out + x
        return out

class ImprovedDynamicEncoder(nn.Module):
    """Enhanced encoder with skip connections and attention"""
    def __init__(self, activation='relu', embed_dim=256):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        self.activation = activation_dict[activation]
        
        # Less aggressive downsampling to preserve information
        self.conv1 = nn.Conv2d(4, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32, activation)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = ResidualBlock(64, activation)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.res3 = ResidualBlock(128, activation)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Attention for better feature preservation
        self.attention = AttentionBlock(256)
        
        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Linear layers with proper dimensionality
        self.body = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim)
        )
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.res1(x)
        skip_connections.append(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.res2(x)
        skip_connections.append(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.res3(x)
        skip_connections.append(x)
        
        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        
        # Attention
        x = self.attention(x)
        
        # Adaptive pooling and linear layers
        x = self.adaptive_pool(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.body(x)
        
        return x, skip_connections

class ImprovedDynamicDecoder(nn.Module):
    """Enhanced decoder with skip connections"""
    def __init__(self, activation='relu', embedding_dim=256):
        super().__init__()
        if activation not in activation_dict:
            raise ValueError(f'activation {activation} not supported')
        self.activation = activation_dict[activation]
        
        # Linear layers to expand embedding
        self.body = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(1024, 256 * 4 * 4),
            self.activation
        )
        
        # Attention
        self.attention = AttentionBlock(256)
        
        # Deconvolutional layers with skip connection processing
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.skip_conv1 = nn.Conv2d(128, 128, kernel_size=1)  # For skip connection from conv3 (128 channels)
        self.res1 = ResidualBlock(128, activation)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.skip_conv2 = nn.Conv2d(64, 64, kernel_size=1)  # For skip connection from conv2 (64 channels)
        self.res2 = ResidualBlock(64, activation)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.skip_conv3 = nn.Conv2d(32, 32, kernel_size=1)  # For skip connection from conv1 (32 channels)
        self.res3 = ResidualBlock(32, activation)
        
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        self.final_conv = nn.Conv2d(16, 4, kernel_size=3, padding=1)
    
    def forward(self, x, target_size, skip_connections=None):
        # Expand embedding
        x = self.body(x)
        x = x.view(-1, 256, 4, 4)
        x = self.attention(x)
        
        # Deconv 1
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        if skip_connections is not None and len(skip_connections) >= 3:
            skip = F.interpolate(skip_connections[2], size=x.shape[2:], mode='bilinear', align_corners=False)
            skip = self.skip_conv1(skip)
            x = x + skip
        x = self.res1(x)
        
        # Deconv 2
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        if skip_connections is not None and len(skip_connections) >= 2:
            skip = F.interpolate(skip_connections[1], size=x.shape[2:], mode='bilinear', align_corners=False)
            skip = self.skip_conv2(skip)
            x = x + skip
        x = self.res2(x)
        
        # Deconv 3
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        if skip_connections is not None and len(skip_connections) >= 1:
            skip = F.interpolate(skip_connections[0], size=x.shape[2:], mode='bilinear', align_corners=False)
            skip = self.skip_conv3(skip)
            x = x + skip
        x = self.res3(x)
        
        # Deconv 4
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        
        # Interpolate to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Final convolution
        x = self.final_conv(x)
        return x

class ImprovedVectorQuantizer(nn.Module):
    """Enhanced vector quantizer with better initialization and EMA updates"""
    def __init__(self, num_embeddings, embed_dim, commitment_cost, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Better initialization
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.normal_(0, 1/self.num_embeddings)
        
        # EMA parameters
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embeddings.weight.data.clone())
        
    def forward(self, x):
        x = x.to(self.embeddings.weight.dtype)
        
        # Calculate distances
        x_norm = torch.sum(x**2, dim=1, keepdim=True)
        e_norm = torch.sum(self.embeddings.weight**2, dim=1)
        distances = x_norm + e_norm - 2 * torch.matmul(x, self.embeddings.weight.t())
        
        # Find closest embeddings
        embed_inds = torch.argmin(distances, dim=1)
        x_quantized = self.embeddings(embed_inds)
        
        # EMA update during training
        if self.training:
            # Update cluster sizes
            embed_onehot = F.one_hot(embed_inds, self.num_embeddings).float()
            cluster_size = embed_onehot.sum(0)
            
            # EMA update
            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            
            # Update embeddings
            embed_sum = torch.matmul(embed_onehot.t(), x)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            # Normalize
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embeddings.weight.data.copy_(embed_normalized)
        
        # Compute losses
        codebook_loss = F.mse_loss(x_quantized.detach(), x)
        commitment_loss = F.mse_loss(x_quantized, x.detach()) * self.commitment_cost
        
        return x_quantized, codebook_loss + commitment_loss, embed_inds

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self, feature_layers=[0, 5, 10, 19, 28]):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        self.features = nn.ModuleList()
        
        prev_layer = 0
        for layer in feature_layers:
            self.features.append(nn.Sequential(*list(vgg.children())[prev_layer:layer+1]))
            prev_layer = layer + 1
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        # Convert single channel to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        elif x.shape[1] == 4:
            # Use first 3 channels for perceptual loss
            x = x[:, :3, :, :]
            y = y[:, :3, :, :]
        
        loss = 0
        for feature_extractor in self.features:
            x = feature_extractor(x)
            y = feature_extractor(y)
            loss += F.mse_loss(x, y)
        
        return loss

class ImprovedDynamicCellVQVAE(nn.Module):
    """Improved VQ-VAE with enhanced architecture and training"""
    def __init__(self, activation='relu', embedding_dim=256, commitment_cost=0.25, 
                 reconstruction_loss_weight=1.0, num_embeddings=512, min_size=32,
                 use_perceptual_loss=True, perceptual_loss_weight=0.1):
        super().__init__()
        self.encoder = ImprovedDynamicEncoder(activation=activation, embed_dim=embedding_dim)
        self.decoder = ImprovedDynamicDecoder(activation=activation, embedding_dim=embedding_dim)
        self.vq_layer = ImprovedVectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.commitment_cost = commitment_cost
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.min_size = min_size
        
        # Perceptual loss
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_loss_weight = perceptual_loss_weight
        if use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss()
    
    def adaptive_padding(self, x, masks=None):
        """Apply minimal padding to make input size efficient for convolutions"""
        batch_size, channels, h, w = x.shape
        
        # Calculate target size (multiple of 16 for 4 downsampling layers)
        target_h = max(self.min_size, ((h + 15) // 16) * 16)
        target_w = max(self.min_size, ((w + 15) // 16) * 16)
        
        # Calculate padding needed
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Apply center padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad input
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        
        # Pad masks if provided
        masks_padded = None
        if masks is not None:
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            masks_padded = F.pad(masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Store padding info
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
        # Store original size
        original_size = x.shape[2:]
        
        # Apply adaptive padding
        x_padded, _, padding_info = self.adaptive_padding(x)
        target_size = x_padded.shape[2:]
        
        # Encode with skip connections
        x_encoded, skip_connections = self.encoder(x_padded)
        
        # Vector quantization
        x_q, vq_loss, embed_inds = self.vq_layer(x_encoded)
        
        # Straight-through estimator
        x_q_st = x_encoded + (x_q - x_encoded).detach()
        
        # Decode with skip connections
        x_reconstructed = self.decoder(x_q_st, target_size, skip_connections)
        
        # Remove padding
        x_reconstructed = self.remove_padding(x_reconstructed, padding_info)
        
        return x_reconstructed, vq_loss, x_encoded, embed_inds
    
    def compute_loss(self, x_reconstructed, x_original, vq_loss, masks=None):
        """Compute balanced loss with proper scaling"""
        batch_size = x_original.shape[0]
        
        # Basic reconstruction loss
        if masks is not None:
            # Content-aware weighting
            content_mask = (masks.float() > 0).float()
            if content_mask.dim() == 3:
                content_mask = content_mask.unsqueeze(1)
            
            # Balance foreground and background
            fg_ratio = content_mask.mean()
            bg_ratio = 1 - fg_ratio
            
            # Weighted MSE loss
            fg_weight = 1.0 / (fg_ratio + 1e-8)
            bg_weight = 0.1 / (bg_ratio + 1e-8)
            
            weights = torch.where(content_mask > 0, fg_weight, bg_weight)
            weights = torch.clamp(weights, 0.1, 10.0)
            
            reconstruction_loss = F.mse_loss(x_reconstructed * weights, x_original * weights)
        else:
            reconstruction_loss = F.mse_loss(x_reconstructed, x_original)
        
        # Perceptual loss
        perceptual_loss = 0
        if self.use_perceptual_loss and hasattr(self, 'perceptual_loss'):
            try:
                perceptual_loss = self.perceptual_loss(x_reconstructed, x_original)
            except:
                perceptual_loss = 0
        
        # Total loss with proper weighting
        total_loss = (self.reconstruction_loss_weight * reconstruction_loss + 
                     vq_loss + 
                     self.perceptual_loss_weight * perceptual_loss)
        
        return total_loss, reconstruction_loss, perceptual_loss
    
    def train_step(self, x, optimizer, masks=None, gradient_clip_norm=None, scheduler=None):
        """Enhanced training step with proper loss computation"""
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        x_reconstructed, vq_loss, _, _ = self.forward(x)
        
        # Compute loss
        total_loss, reconstruction_loss, perceptual_loss = self.compute_loss(
            x_reconstructed, x, vq_loss, masks
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        return total_loss, reconstruction_loss, vq_loss, perceptual_loss