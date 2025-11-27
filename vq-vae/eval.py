# Evaluation Code
# Authors: Samantha Mutiti & Rong Chi
import json, os, argparse
import matplotlib.pyplot as plt

def plot_loss(filepath, show:bool = True, save:bool = True, save_dir:str = os.getcwd()):
    ''' plots training loss from json 
    EXPECTED FORMAT:
    --- epoch: average loss
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    key_list = list(data.keys())
    val_list = []
    for key in key_list:
        val_list.append(data[key]) # preserve order
    
    plt.scatter(key_list, val_list)
    plt.title('Training Loss over Epochs')
    
    if save:
        save_path = os.path.join(save_dir, 'training_loss.png')
        plt.savefig(save_path)
    
    if show: plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # will add more options for eval
    parser.add_argument('--plot_loss', action='store_true', help='plot training loss from json file')
    parser.add_argument('--loss_filepath', type=str, default='./', help='path to training loss json file')
    parser.add_argument('--show', action='store_true', default=True, help='show the plot')
    parser.add_argument('--save', action='store_true', default=True, help='save the plot')
    parser.add_argument('--save_dir', type=str, default=os.getcwd(), help='directory to save the plot')
    args = parser.parse_args()

    if args.plot_loss:
        plot_loss(args.loss_filepath, show=args.show, save=args.save, save_dir=args.save_dir)