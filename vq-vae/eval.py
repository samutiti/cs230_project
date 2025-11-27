# Evaluation Code
# Authors: Samantha Mutiti & Rong Chi
import json, os, argparse
import matplotlib.pyplot as plt

import torch
from model import CellVQVAE
from data import CropDataset

def plot_loss(filepath, show:bool = True, save:bool = True, save_dir:str = os.getcwd()):
    ''' plots training loss from json 
    EXPECTED FORMAT:
    --- epoch: average loss
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    keys = list(data.keys())
    key_list = [int(k) for k in keys]
    val_list = []
    for key in keys:
        val_list.append(data[key]) # preserve order
    
    plt.scatter(key_list, val_list, linestyle='-')
    plt.title('Training Loss over Epochs')
    
    if save:
        save_path = os.path.join(save_dir, 'training_loss.png')
        plt.savefig(save_path)
    
    if show: plt.show()

def model_inference(config_file, image_directory):
    ''' performs inference 
    Args:
        config_file (str): path to the config file used to train the model
        image_directory (str): path to the directory containing images 
    Returns:
        results (tuple): (reconstructions, losses, embeddings)
        filenames (list): list of filenames
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = CellVQVAE(activation=config['activation'], embedding_dim=int(config['embedding_dim'])).to(device)
    model_load_path = os.path.join(config['save_directory'], 'vq_vae_model.pth') # assuming model weights are specified here
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval()

    # load images
    dataset = CropDataset(file_dir=image_directory, type='test')
    inputs, filenames = dataset[:]
    inputs = inputs.to(device)
    results = model(inputs) # format (reconstructions, losses, embeddings)
    return results, filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # will add more options for eval
    parser.add_argument('--plot_loss', action='store_true', help='plot training loss from json file')
    parser.add_argument('--loss_filepath', type=str, default='./', help='path to training loss json file')
    parser.add_argument('--no_show', action='store_false', default=True, help='block showing the plot')
    parser.add_argument('--no_save', action='store_false', default=True, help='block saving the plot')
    parser.add_argument('--save_dir', type=str, default=os.getcwd(), help='directory to save the plot')
    args = parser.parse_args()

    if args.plot_loss:
        plot_loss(args.loss_filepath, show=args.no_show, save=args.no_save, save_dir=args.save_dir)