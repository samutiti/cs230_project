# Improved Dynamic Training Script with Enhanced Features
# Authors: Samantha Mutiti & Rong Chi (Enhanced version)

import torch
from torch.utils.data import DataLoader
import json, argparse
from tqdm import tqdm
import os
import numpy as np
import tifffile as tf
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from collections import defaultdict
import time

from improved_dynamic_model import ImprovedDynamicCellVQVAE
from data import CropDataset
from utils import *
from smart_batching import (
    create_smart_dataloader,
    AccumulatedGradientTrainer,
    SingleImageOptimizer,
    analyze_dataset_sizes
)

optimizer_dict = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'adamw': torch.optim.AdamW}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_config(config):
    required_keys = ['data_directory', 'save_directory', 'batch_size', 'learning_rate', 
                    'optimizer', 'embedding_dim', 'activation', 'epochs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Missing required config key: {key}')

class ImprovedDynamicCropDataset(CropDataset):
    """Enhanced dataset with better normalization and preprocessing"""
    def __init__(self, file_dir, type='train', normalize_method='robust'):
        super().__init__(file_dir, type)
        self.normalize_method = normalize_method
    
    def __getitem__(self, idx):
        """Enhanced preprocessing with better normalization and data type handling"""
        image_filename = self.file_list[idx]
        image = np.array(tf.imread(os.path.join(self.data_directory, image_filename)))
        
        # Handle different data types - convert uint16 to float32
        if image.dtype == np.uint16:
            # Convert uint16 to float32 and normalize to [0, 1] range
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            # Convert uint8 to float32 and normalize to [0, 1] range
            image = image.astype(np.float32) / 255.0
        else:
            # Ensure float32 for other types
            image = image.astype(np.float32)
        
        # Convert to tensor manually (avoid ToTensor issues with uint16)
        image = torch.from_numpy(image)
        
        if image.shape[0] > 5:  # ensure CxHxW format
            image = image.permute(2, 0, 1)
        
        # Take only first 4 channels
        image = image[:4, :, :]
        mask = image[-1, :, :] if image.shape[0] == 4 else torch.zeros(image.shape[1], image.shape[2])
        
        # Apply normalization based on method
        if self.normalize_method == 'robust':
            image = self.robust_normalize(image)
        elif self.normalize_method == 'zscore':
            image = normalize_input_zscore(image.unsqueeze(0)).squeeze(0)
        elif self.normalize_method == 'minmax':
            image = self.minmax_normalize(image)
        
        return image.to(dtype=torch.float32), mask, image_filename
    
    def robust_normalize(self, image):
        """Robust normalization using percentiles"""
        normalized = torch.zeros_like(image)
        for c in range(image.shape[0]):
            channel = image[c]
            # Use 5th and 95th percentiles for robust normalization
            p5 = torch.quantile(channel, 0.05)
            p95 = torch.quantile(channel, 0.95)
            
            # Avoid division by zero
            if p95 - p5 > 1e-8:
                normalized[c] = torch.clamp((channel - p5) / (p95 - p5), 0, 1)
            else:
                normalized[c] = channel
        
        # Scale to [-1, 1] for better training
        return normalized * 2.0 - 1.0
    
    def minmax_normalize(self, image):
        """Min-max normalization per channel"""
        normalized = torch.zeros_like(image)
        for c in range(image.shape[0]):
            channel = image[c]
            min_val = channel.min()
            max_val = channel.max()
            
            if max_val - min_val > 1e-8:
                normalized[c] = (channel - min_val) / (max_val - min_val)
            else:
                normalized[c] = channel
        
        # Scale to [-1, 1]
        return normalized * 2.0 - 1.0

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

def validate_model(model, val_dataloader, device):
    """Validation function"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perceptual_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks, _ in val_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            x_reconstructed, vq_loss, _, _ = model(images)
            
            # Compute losses
            total_loss_batch, recon_loss, perceptual_loss = model.compute_loss(
                x_reconstructed, images, vq_loss, masks
            )
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perceptual_loss += perceptual_loss if isinstance(perceptual_loss, (int, float)) else perceptual_loss.item()
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'perceptual_loss': total_perceptual_loss / num_batches
    }

def save_sample_reconstructions(model, dataloader, save_dir, epoch, device, num_samples=4):
    """Save sample reconstructions for visual inspection"""
    model.eval()
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            images = images.to(device)
            x_reconstructed, _, _, _ = model(images)
            
            # Move to CPU for plotting
            original = images[0].cpu().numpy()
            reconstructed = x_reconstructed[0].cpu().numpy()
            
            # Plot first 3 channels
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            for i in range(3):
                # Original
                axes[0, i].imshow(original[i], cmap='gray')
                axes[0, i].set_title(f'Original Ch{i+1}')
                axes[0, i].axis('off')
                
                # Reconstructed
                axes[1, i].imshow(reconstructed[i], cmap='gray')
                axes[1, i].set_title(f'Reconstructed Ch{i+1}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'samples', f'epoch_{epoch}_sample_{batch_idx}.png'))
            plt.close()

def train(config, augment_epoch=-1):
    """Enhanced training function with validation and monitoring"""
    
    # Model parameters
    commitment_cost = config.get('commitment_cost', 0.25)
    reconstruction_loss_weight = config.get('reconstruction_loss_weight', 1.0)
    num_embeddings = config.get('num_embeddings', 512)
    use_perceptual_loss = config.get('use_perceptual_loss', True)
    perceptual_loss_weight = config.get('perceptual_loss_weight', 0.1)
    normalize_method = config.get('normalize_method', 'robust')
    
    # Create model
    model = ImprovedDynamicCellVQVAE(
        activation=config['activation'], 
        embedding_dim=int(config['embedding_dim']), 
        commitment_cost=commitment_cost,
        reconstruction_loss_weight=reconstruction_loss_weight,
        num_embeddings=num_embeddings,
        use_perceptual_loss=use_perceptual_loss,
        perceptual_loss_weight=perceptual_loss_weight
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer with weight decay
    optimizer_class = optimizer_dict[config['optimizer']]
    if config['optimizer'] == 'adamw':
        optimizer = optimizer_class(model.parameters(), 
                                  lr=float(config['learning_rate']),
                                  weight_decay=config.get('weight_decay', 1e-4))
    else:
        optimizer = optimizer_class(model.parameters(), lr=float(config['learning_rate']))
    
    # Enhanced scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler_type = config.get('scheduler_type', 'step')
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=config.get('scheduler_step_size', 10),
                gamma=config.get('scheduler_gamma', 0.8)
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs']
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
    
    gradient_clip_norm = config.get('gradient_clip_norm', None)
    
    # Load datasets
    train_dataset = ImprovedDynamicCropDataset(
        file_dir=config['data_directory'],
        type='train',
        normalize_method=normalize_method
    )
    
    # Analyze dataset sizes for optimal batching strategy
    dataset_stats = analyze_dataset_sizes(train_dataset, max_samples=50)
    
    # Choose batching strategy based on dataset characteristics
    use_smart_batching = config.get('use_smart_batching', True)
    use_gradient_accumulation = config.get('use_gradient_accumulation', False)
    accumulation_steps = config.get('accumulation_steps', 4)
    
    if use_smart_batching and dataset_stats and dataset_stats['unique_sizes'] < 100:
        print("Using smart batching strategy...")
        train_dataloader = create_smart_dataloader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            size_tolerance=config.get('size_tolerance', 0.3),
            pin_memory=True
        )
    elif config['batch_size'] == 1:
        print("Using optimized single-image batching...")
        # Optimize model for single-batch training
        model = SingleImageOptimizer.optimize_model_for_single_batch(model)
        train_dataloader = SingleImageOptimizer.create_efficient_single_dataloader(
            train_dataset,
            num_workers=min(2, config.get('num_workers', 4))
        )
    else:
        print("Using standard batching with gradient accumulation...")
        use_gradient_accumulation = True
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,  # Use batch_size=1 with accumulation
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
    
    # Validation dataset
    val_dataloader = None
    if os.path.exists(os.path.join(config['data_directory'], 'val')):
        val_dataset = ImprovedDynamicCropDataset(
            file_dir=config['data_directory'],
            type='val',
            normalize_method=normalize_method
        )
        
        # Use smart batching for validation too, but with smaller batch size
        val_batch_size = min(config['batch_size'], 4)
        if use_smart_batching:
            val_dataloader = create_smart_dataloader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=config.get('num_workers', 4),
                size_tolerance=config.get('size_tolerance', 0.3),
                pin_memory=True
            )
        else:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=config.get('num_workers', 4),
                pin_memory=True
            )
        print(f"Validation dataset loaded with {len(val_dataset)} samples")
    
    # Early stopping
    early_stopping = None
    if config.get('use_early_stopping', False) and val_dataloader:
        early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 1e-6)
        )
    
    # Training setup
    epochs = config['epochs']
    train_history = defaultdict(list)
    val_history = defaultdict(list)
    
    # Setup gradient accumulation trainer if needed
    gradient_trainer = None
    if use_gradient_accumulation:
        gradient_trainer = AccumulatedGradientTrainer(
            model, optimizer, accumulation_steps=accumulation_steps
        )
        print(f"Using gradient accumulation with {accumulation_steps} steps")
    
    print(f"Starting training with {len(train_dataset)} training samples")
    print(f"Device: {device}")
    print(f"Effective batch size: {config['batch_size'] * (accumulation_steps if use_gradient_accumulation else 1)}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batching strategy: {'Smart batching' if use_smart_batching else 'Standard batching'}")
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, masks, filenames) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Training step - use gradient accumulation if enabled
            if gradient_trainer:
                total_loss, recon_loss, vq_loss, perceptual_loss = gradient_trainer.train_step(
                    images, masks, gradient_clip_norm
                )
            else:
                total_loss, recon_loss, vq_loss, perceptual_loss = model.train_step(
                    images, optimizer, masks, gradient_clip_norm
                )
            
            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['reconstruction_loss'] += recon_loss.item()
            epoch_losses['vq_loss'] += vq_loss.item()
            epoch_losses['perceptual_loss'] += perceptual_loss if isinstance(perceptual_loss, (int, float)) else perceptual_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
                'VQ': f"{vq_loss.item():.6f}",
                'Batch': f"{images.shape[0]}"
            })
            
            # Check for max batches per epoch
            if config.get('max_batches_per_epoch') and batch_idx >= config['max_batches_per_epoch']:
                break
        
        # Finalize gradient accumulation if used
        if gradient_trainer:
            gradient_trainer.finalize_step(gradient_clip_norm)
        
        # Calculate average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            train_history[key].append(epoch_losses[key])
        
        # Validation phase
        val_losses = {}
        if val_dataloader:
            val_losses = validate_model(model, val_dataloader, device)
            for key, value in val_losses.items():
                val_history[key].append(value)
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses.get('total_loss', epoch_losses['total_loss']))
            else:
                scheduler.step()
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {epoch_losses['total_loss']:.4f}, "
              f"Recon: {epoch_losses['reconstruction_loss']:.4f}, "
              f"VQ: {epoch_losses['vq_loss']:.6f}, "
              f"Perceptual: {epoch_losses['perceptual_loss']:.6f}")
        
        if val_losses:
            print(f"Val   - Loss: {val_losses['total_loss']:.4f}, "
                  f"Recon: {val_losses['reconstruction_loss']:.4f}, "
                  f"VQ: {val_losses['vq_loss']:.6f}, "
                  f"Perceptual: {val_losses['perceptual_loss']:.6f}")
        
        print(f"LR: {current_lr:.6f}")
        
        # Save sample reconstructions
        if (epoch + 1) % config.get('save_samples_every', 10) == 0:
            save_sample_reconstructions(model, train_dataloader, config['save_directory'], epoch + 1, device)
        
        # Early stopping check
        if early_stopping:
            if early_stopping(val_losses['total_loss'], model):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    # Save final model and training history
    save_dir = config['save_directory']
    os.makedirs(save_dir, exist_ok=True)
    
    model_save_path = os.path.join(save_dir, 'improved_vq_vae_dynamic_model.pth')
    torch.save(model.state_dict(), model_save_path)
    
    # Save training history
    history = {
        'train': dict(train_history),
        'val': dict(val_history),
        'config': config
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, save_dir)
    
    print(f"Model saved to {model_save_path}")
    return model, history

def plot_training_curves(history, save_dir):
    """Plot and save training curves"""
    train_history = history['train']
    val_history = history['val']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(train_history['total_loss'], label='Train')
    if val_history.get('total_loss'):
        axes[0, 0].plot(val_history['total_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(train_history['reconstruction_loss'], label='Train')
    if val_history.get('reconstruction_loss'):
        axes[0, 1].plot(val_history['reconstruction_loss'], label='Val')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # VQ loss
    axes[1, 0].plot(train_history['vq_loss'], label='Train')
    if val_history.get('vq_loss'):
        axes[1, 0].plot(val_history['vq_loss'], label='Val')
    axes[1, 0].set_title('VQ Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Perceptual loss
    axes[1, 1].plot(train_history['perceptual_loss'], label='Train')
    if val_history.get('perceptual_loss'):
        axes[1, 1].plot(val_history['perceptual_loss'], label='Val')
    axes[1, 1].set_title('Perceptual Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Improved Dynamic VQ-VAE Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--augment_epoch', type=int, default=-1, help='Epoch to start data augmentation')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    check_config(config)

    # Print configuration
    print("Training configuration:")
    for key, value in config.items():
        print(f"\t{key}: {value}")

    train(config, augment_epoch=args.augment_epoch)