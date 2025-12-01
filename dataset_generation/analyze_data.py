# Authors: Samantha Mutiti & Rong Chi
import os, argparse, json
import tifffile as tf

def analyze(input_dir, output_file='analysis_results.json'):
    '''Analyze the dataset in the given directory.'''
    heights = []
    widths = []
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    for file_name in file_list:
        file_path = os.path.join(input_dir, file_name)
        image = tf.imread(file_path)
        # Assuming the last channel is the mask
        height, width, _ = image.shape
        heights.append(height)
        widths.append(width)

    data_dict = {
        'average_height': sum(heights) / len(heights),
        'average_width': sum(widths) / len(widths),
        'median_height': sorted(heights)[len(heights) // 2],
        'median_width': sorted(widths)[len(widths) // 2],
        'num_images': len(file_list),
        'min_height': min(heights),
        'max_height': max(heights),  
    }

    with open(os.path.join(input_dir, output_file), 'w') as f:
        json.dump(data_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory to analyze')
    parser.add_argument('--output_file', type=str, default='analysis_results.json', help='Output JSON file name')
    args = parser.parse_args()

    analyze(args.input_dir, args.output_file)
