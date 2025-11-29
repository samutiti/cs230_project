# Authors: Samantha Mutiti & Rong Chi
from cellpose import models
import numpy as np
import tifffile as tf
import os, argparse
from tqdm import tqdm
import torch
import re
from collections import defaultdict


def save_cell_crops(mask, image, save_prefix, save_dir, buffer:int=10):
    # copy from sherlock code
    # store mask and crop cell image - name mask as save_prefix_M.tif
    '''crop cells and save cell image and mask
    Args:
        mask (array): the cellpose mask
        image (array): the original image (np.array) HxWxC
        save_prefix (str): the prefix underwhich to save the crop and mask crop
        save_dir (str): directory to save data
        buffer (int): the amount of spave to add to the edge of the crops
    '''
    pass
    cell_indices = np.unique(mask)
    cell_indices = cell_indices[cell_indices != 0]
    for cell_id in tqdm(cell_indices, desc=f'Cropping cells for {save_prefix}'):
        y_coords, x_coords = np.where(mask == cell_id)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        if (
            x_min == 0 or y_min == 0 or x_max == image.shape[1] or y_max == image.shape[0]
        ): # skipping cells that may be cut off by the edge of the image
            continue
        x_start, x_end = max(x_min - buffer, 0), min(x_max + buffer, image.shape[1])
        y_start, y_end = max(y_min - buffer, 0), min(y_max + buffer, image.shape[1])
        cell_crop = image[y_start:y_end, x_start:x_end, :]
        mask_crop = mask[y_start:y_end, x_start:x_end]
        mask_crop = mask_crop.astype(np.uint8)
        # append mask as last channel of cell data
        cell_crop = np.concatenate(
            (cell_crop, np.expand_dims(mask_crop, axis=-1)), axis=-1
        )
        # write data
        with open(f'{save_dir}/{save_prefix}_{cell_id}.tif', 'w') as f:
            tf.imwrite(f, cell_crop)

def load_multichannel_image(input_dir, base_name):
    """Load all 4 channels for a given base name and combine them into a single multi-channel image.
    
    Args:
        input_dir (str): directory containing tif images
        base_name (str): base name without channel suffix (e.g., "Region 2_t001_s099")
    
    Returns:
        np.array: Combined multi-channel image with shape (H, W, 4)
    """
    channels = []
    for ch in range(4):
        channel_file = f"{base_name}_ch{ch:02d}.tif"
        channel_path = os.path.join(input_dir, channel_file)
        
        if not os.path.exists(channel_path):
            raise FileNotFoundError(f"Channel file not found: {channel_path}")
        
        channel_image = tf.imread(channel_path)
        # Ensure channel is 2D (H, W)
        if len(channel_image.shape) == 3:
            channel_image = channel_image[:, :, 0]  # Take first channel if it's RGB
        channels.append(channel_image)
    
    # Stack channels along the last axis to create (H, W, 4) array
    multichannel_image = np.stack(channels, axis=-1)
    return multichannel_image

def group_files_by_base_name(file_list):
    """Group files by their base name (without channel suffix).
    
    Args:
        file_list (list): List of filenames
    
    Returns:
        dict: Dictionary mapping base names to lists of channel files
    """
    # Pattern to match files with channel suffix: _ch00, _ch01, _ch02, _ch03
    channel_pattern = r'^(.+)_ch(\d{2})\.tiff?$'
    grouped_files = defaultdict(list)
    
    for file in file_list:
        match = re.match(channel_pattern, file)
        if match:
            base_name = match.group(1)
            channel_num = int(match.group(2))
            grouped_files[base_name].append((channel_num, file))
    
    # Filter to only include base names that have all 4 channels
    complete_groups = {}
    for base_name, channel_files in grouped_files.items():
        if len(channel_files) == 4:
            # Sort by channel number to ensure correct order
            channel_files.sort(key=lambda x: x[0])
            complete_groups[base_name] = [f[1] for f in channel_files]
    
    return complete_groups

def main(input_dir, output_dir):
    ''' generates cell crops from images in input_dir and saves to output_dir
    Args:
        input_dir (str): directory containing tif images with 4 channels each
        output_dir (str): directory to save cropped images
    '''
    # Initialize Cellpose model
    model = models.CellposeModel(gpu=torch.cuda.is_available()) # assuming cellsam
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    file_list = os.listdir(input_dir)
    
    # Group files by base name (without channel suffix)
    grouped_files = group_files_by_base_name(file_list)
    
    print(f"Found {len(grouped_files)} complete image sets (with all 4 channels)")
    
    for base_name, channel_files in tqdm(grouped_files.items(), desc="Processing image sets"):
        try:
            # Load all 4 channels and combine into single multi-channel image
            multichannel_image = load_multichannel_image(input_dir, base_name)
            
            # Run cellpose segmentation on the multi-channel image
            # Note: Cellpose typically works best with specific channels, you may want to
            # select specific channels for segmentation (e.g., DAPI channel)
            data = model.eval(multichannel_image)
            mask = data[0]
            
            # Save cell crops with the base name as prefix
            save_cell_crops(mask, multichannel_image, base_name, output_dir)
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory containing input tif images')
    parser.add_argument('output_dir', type=str, help='Directory to save')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)