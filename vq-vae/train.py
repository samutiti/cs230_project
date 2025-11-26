# training code for model will go here
# Authors: Samantha Mutiti & Rong Chi

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json, os, argparse
from tqdm import tqdm

from model import CellVQVAE
from data import CropDataset

optimizer_dict = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}

def train(config, augment_epoch=-1):
    model = CellVQVAE(activation=config['activation'], embedding_dim=config['embedding_dim'])
    optimizer = optimizer_dict[config['optimizer']](model.parameters(), lr=config['learning_rate'])
    transforms = config.get('transforms', None) # this will be a string need to convert to nn.Sequential if not None
    if transforms is not None:
        pass # TODO: put str --> nn.Sequential code here
    # NOTE: somewhere can we schedule learning rate decay and data augmentation policies?
    # load data
    dataset = CropDataset(file_dir=config['data_directory'], type='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    # prep training
    epochs = config['epochs']
    train_epoch = {}
    # training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        if epoch == augment_epoch:
            pass # TODO: implement data augmentation start here
        loss = 0
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Training Progress")):
            # TODO: normalize the batch data?
            loss += model.train_step(images, optimizer)
        avg_loss = loss / batch_idx
        train_epoch[epoch] = avg_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    # Save the trained model
    model_save_path = config.get('model_save_path', 'vq_vae_model.pth')
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQ-VAE Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--augment_epoch', type=int, default=-1, help='Epoch to start data augmentation')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Example: Print out the configuration
    print("Training configuration:")
    for key, value in config.items():
        print(f"\t{key}: {value}")

    train(config, argument_epoch=args.augment_epoch)