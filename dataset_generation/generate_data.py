# Authors: Samantha Mutiti & Rong Chi
import cellpose
import numpy as np
import tifffile as tf
import os, argparse
from tqdm import tqdm


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
        with open(f'{save_dir}/{save_prefix}.tif', 'w') as f:
            tf.imwrite(f, cell_crop)

def main(input_dir, output_dir):
    ''' generates cell crops from images in input_dir and saves to output_dir
    Args:
        input_dir (str): directory containing tif images
        output_dir (str): directory to save cropped images
    '''
    # Initialize Cellpose model
    model = cellpose.models.Cellpose() # assuming cellsam
    file_list = os.listdir(input_dir)
    for file in file_list:
        if file.split('.')[-1] != 'tif' and file.split('.')[-1] != 'tiff':
            continue
        image = tf.imread(file)
        data = model.eval(image)
        mask = data[0]
        save_prefix = file.split('.')[0]
        save_cell_crops(mask, image, save_prefix, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, required=True, help='Directory containing input tif images')
    parser.add_argument('output_dir', type=str, required=True, help='Directory to save')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)