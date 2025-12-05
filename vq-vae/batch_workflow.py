#!/usr/bin/env python3
"""
Complete Batch Preprocessing and Training Workflow
This script provides a complete workflow for preprocessing batches and training.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.1f} seconds")
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: {description} completed!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def preprocess_batches(data_directory, batch_size=8, tolerance=0.3, workers=None):
    """Preprocess batches for training and validation"""
    
    print("Starting batch preprocessing...")
    
    # Preprocess training batches
    train_cmd = f"python precompute_batches.py '{data_directory}' " \
                f"--output train_batches.json " \
                f"--batch-size {batch_size} " \
                f"--tolerance {tolerance} " \
                f"--dataset-type train"
    
    if workers:
        train_cmd += f" --workers {workers}"
    
    if not run_command(train_cmd, "Preprocessing training batches"):
        return False
    
    # Preprocess validation batches if validation directory exists
    val_dir = os.path.join(data_directory, 'val')
    if os.path.exists(val_dir):
        val_cmd = f"python precompute_batches.py '{data_directory}' " \
                  f"--output val_batches.json " \
                  f"--batch-size {batch_size} " \
                  f"--tolerance {tolerance} " \
                  f"--dataset-type val"
        
        if workers:
            val_cmd += f" --workers {workers}"
        
        if not run_command(val_cmd, "Preprocessing validation batches"):
            return False
    else:
        print("No validation directory found, skipping validation batch preprocessing")
    
    return True

def create_training_config(data_directory, config_template="precomputed_config.json"):
    """Create training configuration with correct paths"""
    
    # Load template config
    with open(config_template, 'r') as f:
        config = json.load(f)
    
    # Update paths
    config['data_directory'] = data_directory
    config['precomputed_batches'] = os.path.abspath('train_batches.pkl')
    
    # Add validation batches if they exist
    if os.path.exists('val_batches.pkl'):
        config['precomputed_val_batches'] = os.path.abspath('val_batches.pkl')
    
    # Save updated config
    output_config = 'training_config.json'
    with open(output_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to: {output_config}")
    return output_config

def run_training(config_file):
    """Run training with the specified configuration"""
    
    train_cmd = f"python improved_train_dynamic.py --config {config_file}"
    
    return run_command(train_cmd, "Training VQ-VAE model")

def verify_setup(data_directory):
    """Verify that all required files and directories exist"""
    
    print("Verifying setup...")
    
    # Check data directory
    if not os.path.exists(data_directory):
        print(f"ERROR: Data directory not found: {data_directory}")
        return False
    
    train_dir = os.path.join(data_directory, 'train')
    if not os.path.exists(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        return False
    
    # Check required Python files
    required_files = [
        'precompute_batches.py',
        'fast_batch_sampler.py',
        'improved_train_dynamic.py',
        'improved_dynamic_model.py',
        'precomputed_config.json'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Required file not found: {file}")
            return False
    
    print("✅ Setup verification passed!")
    return True

def estimate_preprocessing_time(data_directory, workers=None):
    """Estimate preprocessing time based on dataset size"""
    
    train_dir = os.path.join(data_directory, 'train')
    if not os.path.exists(train_dir):
        return
    
    # Count TIFF files
    tiff_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.tif', '.tiff'))]
    num_files = len(tiff_files)
    
    if workers is None:
        workers = min(8, os.cpu_count())
    
    # Estimate time (roughly 0.01 seconds per file with multiprocessing)
    estimated_seconds = (num_files * 0.01) / workers
    estimated_minutes = estimated_seconds / 60
    
    print(f"Dataset analysis:")
    print(f"  Files to process: {num_files:,}")
    print(f"  Workers: {workers}")
    print(f"  Estimated preprocessing time: {estimated_minutes:.1f} minutes")

def main():
    parser = argparse.ArgumentParser(description='Complete batch preprocessing and training workflow')
    parser.add_argument('data_directory', help='Path to dataset directory')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Target batch size')
    parser.add_argument('--tolerance', '-t', type=float, default=0.3, help='Size tolerance')
    parser.add_argument('--workers', '-w', type=int, help='Number of worker processes')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip preprocessing (use existing batch files)')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip training (only preprocess batches)')
    parser.add_argument('--config-template', default='precomputed_config.json',
                       help='Configuration template file')
    
    args = parser.parse_args()
    
    print("VQ-VAE Batch Preprocessing and Training Workflow")
    print("="*60)
    print(f"Data directory: {args.data_directory}")
    print(f"Batch size: {args.batch_size}")
    print(f"Size tolerance: {args.tolerance}")
    print(f"Workers: {args.workers or 'auto'}")
    
    # Step 1: Verify setup
    if not verify_setup(args.data_directory):
        sys.exit(1)
    
    # Step 2: Estimate preprocessing time
    if not args.skip_preprocessing:
        estimate_preprocessing_time(args.data_directory, args.workers)
        
        # Ask for confirmation for large datasets
        train_dir = os.path.join(args.data_directory, 'train')
        num_files = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.tif', '.tiff'))])
        
        if num_files > 10000:
            response = input(f"\nDataset has {num_files:,} files. Continue with preprocessing? (y/N): ")
            if response.lower() != 'y':
                print("Preprocessing cancelled.")
                sys.exit(0)
    
    # Step 3: Preprocess batches
    if not args.skip_preprocessing:
        if not preprocess_batches(args.data_directory, args.batch_size, args.tolerance, args.workers):
            print("Batch preprocessing failed!")
            sys.exit(1)
    else:
        print("Skipping batch preprocessing (using existing files)")
    
    # Step 4: Create training configuration
    config_file = create_training_config(args.data_directory, args.config_template)
    
    # Step 5: Run training
    if not args.skip_training:
        print(f"\nStarting training with configuration: {config_file}")
        if not run_training(config_file):
            print("Training failed!")
            sys.exit(1)
    else:
        print("Skipping training")
        print(f"To run training manually: python improved_train_dynamic.py --config {config_file}")
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    if not args.skip_training:
        print("Training has finished. Check the results directory for outputs.")
    else:
        print("Batch preprocessing completed. You can now run training with:")
        print(f"  python improved_train_dynamic.py --config {config_file}")

if __name__ == "__main__":
    main()