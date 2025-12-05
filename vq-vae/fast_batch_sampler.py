# Fast Batch Sampler Using Precomputed Groups
# Authors: Samantha Mutiti & Rong Chi (Enhanced version)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import numpy as np
import pickle
import random
from collections import defaultdict

class PrecomputedBatchSampler(Sampler):
    """Fast batch sampler using precomputed batch groups"""
    
    def __init__(self, dataset, precomputed_batches_path, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        # Load precomputed batch information
        print(f"Loading precomputed batches from: {precomputed_batches_path}")
        with open(precomputed_batches_path, 'rb') as f:
            self.batch_info = pickle.load(f)
        
        # Create filename to index mapping for fast lookup
        self.filename_to_idx = {
            self.dataset.file_list[i]: i 
            for i in range(len(self.dataset.file_list))
        }
        
        # Convert filename batches to index batches
        self.index_batches = []
        self.batch_sizes = []  # Track actual batch sizes
        
        for group in self.batch_info['batch_groups']:
            for filename_batch in group['batches']:
                # Convert filenames to indices
                index_batch = []
                for filename in filename_batch:
                    if filename in self.filename_to_idx:
                        index_batch.append(self.filename_to_idx[filename])
                
                if index_batch:  # Only add non-empty batches
                    self.index_batches.append(index_batch)
                    self.batch_sizes.append(len(index_batch))
        
        print(f"Loaded {len(self.index_batches)} precomputed batches")
        print(f"Total samples: {sum(self.batch_sizes)}")
        print(f"Avg batch size: {np.mean(self.batch_sizes):.1f}")
        
    def __iter__(self):
        """Generate batches of indices"""
        batch_indices = list(range(len(self.index_batches)))
        
        if self.shuffle:
            # Shuffle the order of batches
            random.shuffle(batch_indices)
            
            # Also shuffle within each batch
            for batch_idx in batch_indices:
                batch = self.index_batches[batch_idx].copy()
                random.shuffle(batch)
                yield batch
        else:
            # Return batches in order
            for batch_idx in batch_indices:
                yield self.index_batches[batch_idx]
    
    def __len__(self):
        """Return the number of batches"""
        return len(self.index_batches)

class FastSmartCollate:
    """Fast collate function using precomputed size information"""
    
    def __init__(self, precomputed_batches_path):
        # Load size information
        with open(precomputed_batches_path, 'rb') as f:
            batch_info = pickle.load(f)
        self.size_info = batch_info['size_info']
    
    def __call__(self, batch):
        """Custom collate function with precomputed sizes"""
        images, masks, filenames = zip(*batch)
        
        # Get target size for this batch (all images in batch should be similar)
        # Use the size of the first image as reference
        first_filename = filenames[0]
        if first_filename in self.size_info:
            target_h, target_w = self.size_info[first_filename]
            # Round up to nearest multiple of 16
            target_h = ((target_h + 15) // 16) * 16
            target_w = ((target_w + 15) // 16) * 16
        else:
            # Fallback: find max dimensions in batch
            target_h = max(img.shape[1] for img in images)
            target_w = max(img.shape[2] for img in images)
        
        # Pad all images to target size
        padded_images = []
        padded_masks = []
        
        for img, mask in zip(images, masks):
            # Calculate padding needed
            pad_h = target_h - img.shape[1]
            pad_w = target_w - img.shape[2]
            
            if pad_h > 0 or pad_w > 0:
                # Apply padding (left, right, top, bottom)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                
                # Pad image with reflection
                padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
                
                # Pad mask with zeros
                if mask.dim() == 2:
                    padded_mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                else:
                    padded_mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            else:
                padded_img = img
                padded_mask = mask
            
            padded_images.append(padded_img)
            padded_masks.append(padded_mask)
        
        # Stack into batches
        batch_images = torch.stack(padded_images, dim=0)
        batch_masks = torch.stack(padded_masks, dim=0)
        
        return batch_images, batch_masks, filenames

def create_fast_dataloader(dataset, precomputed_batches_path, shuffle=True, 
                          num_workers=4, pin_memory=True):
    """Create a fast dataloader using precomputed batch groups"""
    
    # Create batch sampler
    batch_sampler = PrecomputedBatchSampler(
        dataset, 
        precomputed_batches_path, 
        shuffle=shuffle
    )
    
    # Create collate function
    collate_fn = FastSmartCollate(precomputed_batches_path)
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

class BatchGroupAnalyzer:
    """Analyze precomputed batch groups for optimization"""
    
    def __init__(self, precomputed_batches_path):
        with open(precomputed_batches_path, 'rb') as f:
            self.batch_info = pickle.load(f)
    
    def analyze_efficiency(self):
        """Analyze batching efficiency"""
        total_files = 0
        total_slots = 0
        group_stats = []
        
        for group in self.batch_info['batch_groups']:
            num_files = group['num_files']
            num_batches = group['num_batches']
            
            # Estimate batch size from first batch
            if group['batches']:
                estimated_batch_size = len(group['batches'][0])
                slots = num_batches * estimated_batch_size
            else:
                slots = 0
            
            efficiency = (num_files / slots * 100) if slots > 0 else 0
            
            group_stats.append({
                'target_size': group['target_size'],
                'num_files': num_files,
                'num_batches': num_batches,
                'efficiency': efficiency
            })
            
            total_files += num_files
            total_slots += slots
        
        overall_efficiency = (total_files / total_slots * 100) if total_slots > 0 else 0
        
        return {
            'overall_efficiency': overall_efficiency,
            'total_files': total_files,
            'total_slots': total_slots,
            'wasted_slots': total_slots - total_files,
            'group_stats': group_stats
        }
    
    def print_analysis(self):
        """Print detailed analysis"""
        stats = self.analyze_efficiency()
        
        print("\n" + "="*60)
        print("PRECOMPUTED BATCH ANALYSIS")
        print("="*60)
        
        print(f"Overall efficiency: {stats['overall_efficiency']:.1f}%")
        print(f"Total files: {stats['total_files']:,}")
        print(f"Total slots: {stats['total_slots']:,}")
        print(f"Wasted slots: {stats['wasted_slots']:,}")
        
        print(f"\nGroup Details:")
        for i, group in enumerate(stats['group_stats'][:10]):  # Show first 10 groups
            print(f"  Group {i+1}: {group['target_size']} - "
                  f"{group['num_files']} files, {group['num_batches']} batches, "
                  f"{group['efficiency']:.1f}% efficient")
        
        if len(stats['group_stats']) > 10:
            print(f"  ... and {len(stats['group_stats']) - 10} more groups")
        
        print("="*60)

def verify_precomputed_batches(dataset, precomputed_batches_path):
    """Verify that precomputed batches match the current dataset"""
    
    print("Verifying precomputed batches...")
    
    with open(precomputed_batches_path, 'rb') as f:
        batch_info = pickle.load(f)
    
    # Check if all files in batches exist in dataset
    dataset_files = set(dataset.file_list)
    batch_files = set()
    
    for group in batch_info['batch_groups']:
        for batch in group['batches']:
            batch_files.update(batch)
    
    missing_files = batch_files - dataset_files
    extra_files = dataset_files - batch_files
    
    print(f"Dataset files: {len(dataset_files)}")
    print(f"Batch files: {len(batch_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Extra files: {len(extra_files)}")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} files in batches not found in dataset")
        if len(missing_files) <= 10:
            print("Missing files:", list(missing_files))
    
    if extra_files:
        print(f"Warning: {len(extra_files)} files in dataset not in batches")
        if len(extra_files) <= 10:
            print("Extra files:", list(extra_files))
    
    if not missing_files and not extra_files:
        print("✅ Precomputed batches match dataset perfectly!")
        return True
    else:
        print("⚠️  Precomputed batches don't match current dataset")
        return False

# Utility functions for integration
def update_config_for_precomputed_batches(config, precomputed_batches_path):
    """Update training config to use precomputed batches"""
    config['precomputed_batches'] = precomputed_batches_path
    config['use_smart_batching'] = False  # Disable runtime analysis
    config['use_precomputed_batches'] = True
    return config

def estimate_training_time(precomputed_batches_path, epochs, seconds_per_batch=1.0):
    """Estimate training time using precomputed batches"""
    
    with open(precomputed_batches_path, 'rb') as f:
        batch_info = pickle.load(f)
    
    total_batches = batch_info['total_batches']
    total_time_seconds = total_batches * epochs * seconds_per_batch
    
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60
    
    print(f"Training time estimate:")
    print(f"  Total batches per epoch: {total_batches:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Estimated time: {hours:.0f}h {minutes:.0f}m")
    print(f"  (assuming {seconds_per_batch}s per batch)")