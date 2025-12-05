#!/usr/bin/env python3
"""
Precompute Batch Groups for Large Datasets
This script analyzes your dataset once and saves batch groupings for fast training startup.
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import tifffile as tf
import multiprocessing as mp
from functools import partial
import time

def get_image_size_fast(args):
    """Fast image size extraction using TIFF headers"""
    data_dir, filename = args
    filepath = os.path.join(data_dir, filename)
    
    try:
        # Try to read just the header for speed
        with tf.TiffFile(filepath) as tif:
            page = tif.pages[0]
            if hasattr(page, 'shape'):
                if len(page.shape) == 3:  # C, H, W
                    return filename, (page.shape[1], page.shape[2])
                else:  # H, W
                    return filename, page.shape
            else:
                # Fallback: read image dimensions
                image = tf.imread(filepath)
                if len(image.shape) == 3:
                    return filename, (image.shape[1], image.shape[2])
                else:
                    return filename, image.shape
    except Exception as e:
        print(f"Warning: Could not read {filename}: {e}")
        return filename, (64, 64)  # Default size

def analyze_dataset_sizes_parallel(data_directory, dataset_type='train', num_workers=None):
    """Analyze dataset sizes using parallel processing"""
    
    data_dir = os.path.join(data_directory, dataset_type)
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Get all TIFF files
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.tif', '.tiff'))]
    
    if not files:
        raise ValueError(f"No TIFF files found in {data_dir}")
    
    print(f"Analyzing {len(files)} files in {data_dir}...")
    
    # Use multiprocessing for speed
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Don't use too many workers
    
    # Prepare arguments for parallel processing
    args_list = [(data_dir, filename) for filename in files]
    
    # Process files in parallel
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(get_image_size_fast, args_list),
            total=len(files),
            desc="Reading image sizes"
        ))
    
    # Convert results to dictionary
    size_info = {}
    for filename, size in results:
        size_info[filename] = size
    
    return size_info

def create_batch_groups(size_info, batch_size=8, size_tolerance=0.3):
    """Create batch groups based on image sizes"""
    
    print(f"Creating batch groups with batch_size={batch_size}, tolerance={size_tolerance}")
    
    # Group files by similar sizes
    size_groups = defaultdict(list)
    
    for filename, (h, w) in size_info.items():
        # Round to nearest multiple of 16 for GPU efficiency
        bucket_h = ((h + 15) // 16) * 16
        bucket_w = ((w + 15) // 16) * 16
        bucket_size = (bucket_h, bucket_w)
        size_groups[bucket_size].append(filename)
    
    # Create batch groups
    batch_groups = []
    total_files = 0
    
    for target_size, filenames in size_groups.items():
        if len(filenames) == 0:
            continue
            
        # Create batches from this size group
        batches = []
        for i in range(0, len(filenames), batch_size):
            batch = filenames[i:i + batch_size]
            batches.append(batch)
        
        group_info = {
            'target_size': target_size,
            'num_files': len(filenames),
            'num_batches': len(batches),
            'batches': batches
        }
        
        batch_groups.append(group_info)
        total_files += len(filenames)
        
        print(f"  Size {target_size}: {len(filenames)} files -> {len(batches)} batches")
    
    # Sort by size for consistent ordering
    batch_groups.sort(key=lambda x: x['target_size'][0] * x['target_size'][1])
    
    print(f"Created {len(batch_groups)} size groups with {total_files} total files")
    
    return batch_groups

def save_batch_info(batch_groups, size_info, output_path):
    """Save batch information to file"""
    
    batch_info = {
        'version': '1.0',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_groups': len(batch_groups),
        'total_files': sum(group['num_files'] for group in batch_groups),
        'total_batches': sum(group['num_batches'] for group in batch_groups),
        'batch_groups': batch_groups,
        'size_info': size_info
    }
    
    # Save as pickle for fast loading
    pickle_path = output_path.replace('.json', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(batch_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save as JSON for human readability (without detailed batches)
    json_info = batch_info.copy()
    json_info['batch_groups'] = [
        {
            'target_size': group['target_size'],
            'num_files': group['num_files'],
            'num_batches': group['num_batches']
        }
        for group in batch_groups
    ]
    del json_info['size_info']  # Too large for JSON
    
    with open(output_path, 'w') as f:
        json.dump(json_info, f, indent=2)
    
    print(f"Batch info saved to:")
    print(f"  Pickle (for training): {pickle_path}")
    print(f"  JSON (for inspection): {output_path}")
    
    return pickle_path

def load_batch_info(pickle_path):
    """Load precomputed batch information"""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def analyze_batch_statistics(batch_groups):
    """Analyze and display batch statistics"""
    
    print("\n" + "="*60)
    print("BATCH ANALYSIS SUMMARY")
    print("="*60)
    
    total_files = sum(group['num_files'] for group in batch_groups)
    total_batches = sum(group['num_batches'] for group in batch_groups)
    
    print(f"Total files: {total_files:,}")
    print(f"Total batches: {total_batches:,}")
    print(f"Size groups: {len(batch_groups)}")
    
    # Size distribution
    sizes = [group['target_size'] for group in batch_groups]
    areas = [h * w for h, w in sizes]
    
    print(f"\nSize Statistics:")
    print(f"  Smallest: {min(sizes)}")
    print(f"  Largest: {max(sizes)}")
    print(f"  Area range: {min(areas):,} - {max(areas):,} pixels")
    
    # Batch distribution
    batch_counts = [group['num_batches'] for group in batch_groups]
    file_counts = [group['num_files'] for group in batch_groups]
    
    print(f"\nBatch Distribution:")
    print(f"  Avg files per group: {np.mean(file_counts):.1f}")
    print(f"  Avg batches per group: {np.mean(batch_counts):.1f}")
    print(f"  Largest group: {max(file_counts)} files")
    print(f"  Smallest group: {min(file_counts)} files")
    
    # Efficiency analysis
    total_slots = sum(group['num_batches'] * 8 for group in batch_groups)  # Assuming batch_size=8
    efficiency = total_files / total_slots * 100
    
    print(f"\nBatching Efficiency:")
    print(f"  Batch utilization: {efficiency:.1f}%")
    print(f"  Wasted slots: {total_slots - total_files}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Precompute batch groups for efficient training')
    parser.add_argument('data_directory', help='Path to dataset directory')
    parser.add_argument('--output', '-o', default='batch_info.json', help='Output file path')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Target batch size')
    parser.add_argument('--tolerance', '-t', type=float, default=0.3, help='Size tolerance')
    parser.add_argument('--workers', '-w', type=int, help='Number of worker processes')
    parser.add_argument('--dataset-type', choices=['train', 'val', 'test'], default='train',
                       help='Dataset type to analyze')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_directory):
        print(f"Error: Directory not found: {args.data_directory}")
        sys.exit(1)
    
    print(f"Precomputing batch groups for: {args.data_directory}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Size tolerance: {args.tolerance}")
    print(f"Workers: {args.workers or 'auto'}")
    
    try:
        # Step 1: Analyze dataset sizes
        start_time = time.time()
        size_info = analyze_dataset_sizes_parallel(
            args.data_directory, 
            args.dataset_type, 
            args.workers
        )
        analysis_time = time.time() - start_time
        print(f"Size analysis completed in {analysis_time:.1f} seconds")
        
        # Step 2: Create batch groups
        batch_groups = create_batch_groups(
            size_info, 
            args.batch_size, 
            args.tolerance
        )
        
        # Step 3: Save batch information
        pickle_path = save_batch_info(batch_groups, size_info, args.output)
        
        # Step 4: Display statistics
        analyze_batch_statistics(batch_groups)
        
        print(f"\n✅ Batch preprocessing completed successfully!")
        print(f"Use this file in training: {pickle_path}")
        print(f"\nTo use in training, add to your config:")
        print(f'  "precomputed_batches": "{pickle_path}"')
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()