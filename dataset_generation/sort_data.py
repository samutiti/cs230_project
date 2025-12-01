import os, argparse, shutil
import numpy as np
from tqdm import tqdm

SPLIT = [0.7, 0.15, 0.15]  # train, val, test split ratios

def sort(input_dir, progress=True):
    '''splits data into train, val and test
    EXPECTED DIR FORMAT:
        input_dir/
            test/
            train/
                data_1.tf
                data_2.tif
                ...
            val/
    '''
    files = os.listdir(os.path.join(input_dir, 'train'))
    np.random.shuffle(files)
    num_files = len(files)
    train, val, _ = (np.array(SPLIT) * num_files).astype(int)
    # MOVE TEST FILES
    test_files = files[train + val - 1:] # get last test files
    for file in tqdm(test_files, disable=not progress, desc='Moving test files'):
        src = os.path.join(input_dir, 'train', file)
        dst = os.path.join(input_dir, 'test', file)
        shutil.move(src, dst)
    # MOVE VAL FILES
    val_files = files[train - 1:train + val] # get val files
    for file in tqdm(val_files, disable=not progress, desc='Moving val files'):
        src = os.path.join(input_dir, 'train', file)
        dst = os.path.join(input_dir, 'val', file)
        shutil.move(src, dst)
    print (f'Moved {len(test_files)} files to test/')
    print (f'Moved {len(val_files)} files to val/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory containing tif images')
    parser.add_argument('--quiet', action='store_false', default=True, help='Show progress bar')
    args = parser.parse_args()

    sort(args.input_dir, args.quiet)
