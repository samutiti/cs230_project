# Dynamic Training Script for Variable Input Sizes
# Authors: Samantha Mutiti & Rong Chi (Modified for dynamic model)

import torch
from torch.utils.data import DataLoader
import json, argparse
from tqdm import tqdm
import os
import numpy as np
import tifffile as tf
from torchvision.transforms import ToTensor

from dynamic_model import DynamicCellVQVAE
from data import CropDataset
from utils import *

optimizer_dict = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_config(config):
    required_keys = ['data_directory', 'save_directory', 'batch_size', 'learning_rate', 'optimizer', 'embedding_dim', 'activation', 'epochs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Missing required config key: {key}')

# Custom Dataset for Dynamic Model (minimal padding)
class DynamicCropDataset(CropDataset):
    def __getitem__(self, idx):
        """Override to use minimal padding instead of fixed 300x300"""
        image_filename = self.file_list[idx]
        image = np.array(tf.imread(os.path.join(self.data_directory, image_filename)))
        
        # Convert to tensor
        image = ToTensor()(image)
        if image.shape[0] > 5:  # ensure CxHxW format
            image = image.permute(2, 0, 1)
        
        # NO PADDING HERE - let the dynamic model handle it
        image = image[:4, :, :]  # Take only first 4 channels
        mask = image[-1, :, :] if image.shape[0] == 4 else torch.zeros(image.shape[1], image.shape[2])
        
        return image.to(dtype=torch.float32), mask, image_filename

def train(config, augment_epoch=-1):
    commitment_cost = config.get('commitment_cost', 0.25)
    reconstruction_loss_weight = config.get('reconstruction_loss_weight', 1.0)
    num_embeddings = config.get('num_embeddings', 1024)
    
    model = DynamicCellVQVAE(
        activation=config['activation'], 
        embedding_dim=int(config['embedding_dim']), 
        commitment_cost=commitment_cost,
        reconstruction_loss_weight=reconstruction_loss_weight,
        num_embeddings=num_embeddings
    ).to(device)
    
    optimizer = optimizer_dict[config['optimizer']](model.parameters(), lr=float(config['learning_rate']))
    
    # Setup scheduler if specified
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get('scheduler_step_size', 10),
            gamma=config.get('scheduler_gamma', 0.8)
        )
    
    gradient_clip_norm = config.get('gradient_clip_norm', None)
    
    # Load data with dynamic dataset
    dataset = DynamicCropDataset(file_dir=config['data_directory'], type='train')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)  # Use batch_size=1 for variable sizes
    
    # Training setup
    epochs = config['epochs']
    train_epoch = {}
    
    print(f"Starting training with {len(dataset)} samples")
    
    # Training loop
    for epoch in range(epochs):
        if epoch == augment_epoch:
            pass  # TODO: implement data augmentation
        
        total_loss = 0
        reconstruction_loss_total = 0
        vq_loss_total = 0
        num_batches = 0
        
        for batch_idx, (images, masks, filenames) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
            images = normalize_input_zscore(images)
            images = images.to(device)
            masks = masks.to(device)
            
            # Handle batch size = 1 for variable sizes
            for i in range(images.shape[0]):
                single_image = images[i:i+1]  # Keep batch dimension
                single_mask = masks[i:i+1] if masks is not None else None
                
                loss, recon_loss, vq_loss = model.train_step(
                    single_image, optimizer, single_mask, gradient_clip_norm
                )
                
                total_loss += loss.item()
                reconstruction_loss_total += recon_loss.item()
                vq_loss_total += vq_loss.item()
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_recon_loss = reconstruction_loss_total / num_batches
        avg_vq_loss = vq_loss_total / num_batches
        
        train_epoch[epoch] = {
            'total_loss': avg_loss,
            'reconstruction_loss': avg_recon_loss,
            'vq_loss': avg_vq_loss
        }
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, VQ: {avg_vq_loss:.6f}, LR: {current_lr:.6f}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Save the trained model and training loss
    save_dir = config['save_directory']
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    
    model_save_path = os.path.join(save_dir, 'vq_vae_dynamic_model.pth')
    torch.save(model.state_dict(), model_save_path)
    with open(os.path.join(save_dir, 'train_loss.json'), 'w') as f:
        json.dump(train_epoch, f)
    
    print(f"Model saved to {model_save_path}")
    return model, train_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dynamic VQ-VAE Model')
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