# training code for model will go here
# Authors: Samantha Mutiti & Rong Chi

import torch
from torch.utils.data import DataLoader
import json, argparse
from tqdm import tqdm
import os

from model import CellVQVAE
from data import CropDataset
from utils import *

optimizer_dict = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_config(config):
    required_keys = ['data_directory', 'save_directory', 'batch_size', 'learning_rate', 'optimizer', 'embedding_dim', 'activation', 'epochs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Missing required config key: {key}')

def train(config, augment_epoch=-1):
    commitment_cost = config.get('commitment_cost', 0.25)
    reconstruction_loss_weight = config.get('reconstruction_loss_weight', 1.0)
    num_embeddings = config.get('num_embeddings', 1024)
    model = CellVQVAE(activation=config['activation'], 
                      embedding_dim=int(config['embedding_dim']), 
                      commitment_cost=commitment_cost,
                      reconstruction_loss_weight=reconstruction_loss_weight,
                      num_embeddings=num_embeddings).to(device) # send model to device
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
    transforms = config.get('transforms', None) # this will be a string need to convert to nn.Sequential if not None
    if transforms is not None:
        pass # TODO: put str --> nn.Sequential code here
    # NOTE: somewhere can we schedule learning rate decay and data augmentation policies?
    # load data
    dataset = CropDataset(file_dir=config['data_directory'], type='train')
    dataloader = DataLoader(dataset, batch_size=int(config['batch_size']), shuffle=True)
    # prep training
    epochs = config['epochs']
    train_epoch = {}
    # training loop
    for epoch in range(epochs):
        if epoch == augment_epoch:
            pass # TODO: implement data augmentation start here
        loss = 0
        reconstruction_loss_total = 0
        vq_loss_total = 0
        
        for batch_idx, (images, masks, _) in enumerate(tqdm(dataloader, desc="Training Progress Epoch {}".format(epoch+1))):
            images = normalize_input_zscore(images) # normalize the data
            images = images.to(device) # send to device
            masks = masks.to(device)
            
            total_loss, recon_loss, vq_loss = model.train_step(images, optimizer, masks, gradient_clip_norm)
            loss += total_loss
            reconstruction_loss_total += recon_loss
            vq_loss_total += vq_loss
            
        avg_loss = loss / (batch_idx + 1)
        avg_recon_loss = reconstruction_loss_total / (batch_idx + 1)
        avg_vq_loss = vq_loss_total / (batch_idx + 1)
        
        train_epoch[epoch] = {
            'total_loss': avg_loss.item(),
            'reconstruction_loss': avg_recon_loss.item(), 
            'vq_loss': avg_vq_loss.item()
        }
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, VQ: {avg_vq_loss:.6f}, LR: {current_lr:.6f}")
        
        # Step scheduler at end of epoch
        if scheduler is not None:
            scheduler.step()
    
    # Save the trained model and training loss
    save_dir = config['save_directory']
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    model_save_path = os.path.join(save_dir, 'vq_vae_model.pth')
    torch.save(model.state_dict(), model_save_path)
    with open(os.path.join(save_dir, 'train_loss.json'), 'w') as f:
        json.dump(train_epoch, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQ-VAE Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--augment_epoch', type=int, default=-1, help='Epoch to start data augmentation')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    check_config(config) # check config before model loading/training

    # Example: Print out the configuration
    print("Training configuration:")
    for key, value in config.items():
        print(f"\t{key}: {value}")

    train(config, augment_epoch=args.augment_epoch)