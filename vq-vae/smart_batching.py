# Smart Batching System for Variable-Size Images
# Authors: Samantha Mutiti & Rong Chi (Enhanced version)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import numpy as np
import math
from collections import defaultdict
import random

class SizeBucket:
    """Represents a bucket of images with similar sizes"""
    def __init__(self, target_size, tolerance=0.2):
        self.target_size = target_size
        self.tolerance = tolerance
        self.indices = []
        
    def fits(self, size):
        """Check if a size fits in this bucket"""
        h, w = size
        target_h, target_w = self.target_size
        
        # Check if size is within tolerance
        h_ratio = abs(h - target_h) / target_h
        w_ratio = abs(w - target_w) / target_w
        
        return h_ratio <= self.tolerance and w_ratio <= self.tolerance
    
    def add_index(self, idx):
        """Add an index to this bucket"""
        self.indices.append(idx)
    
    def __len__(self):
        return len(self.indices)

class SmartBatchSampler(Sampler):
    """Smart batch sampler that groups similar-sized images together"""
    
    def __init__(self, dataset, batch_size, drop_last=False, size_tolerance=0.3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.size_tolerance = size_tolerance
        
        # Analyze dataset sizes and create buckets
        self.buckets = self._create_size_buckets()
        
    def _get_image_size(self, idx):
        """Get the size of an image without loading it fully"""
        try:
            # Get image without full processing to determine size
            image_filename = self.dataset.file_list[idx]
            import tifffile as tf
            import os
            
            # Read image header to get dimensions
            image_path = os.path.join(self.dataset.data_directory, image_filename)
            with tf.TiffFile(image_path) as tif:
                page = tif.pages[0]
                if hasattr(page, 'shape'):
                    if len(page.shape) == 3:  # C, H, W
                        return page.shape[1], page.shape[2]
                    else:  # H, W
                        return page.shape[0], page.shape[1]
                else:
                    # Fallback: load the image
                    image = tf.imread(image_path)
                    if len(image.shape) == 3:
                        return image.shape[1], image.shape[2]
                    else:
                        return image.shape[0], image.shape[1]
        except:
            # Fallback: assume a default size
            return 64, 64
    
    def _create_size_buckets(self):
        """Create buckets based on image sizes in the dataset"""
        print("Analyzing dataset for smart batching...")
        
        # Collect all sizes
        sizes = []
        for idx in range(len(self.dataset)):
            size = self._get_image_size(idx)
            sizes.append((idx, size))
        
        # Group sizes and create buckets
        size_groups = defaultdict(list)
        for idx, size in sizes:
            # Round to nearest multiple of 16 for efficiency
            h, w = size
            bucket_h = ((h + 15) // 16) * 16
            bucket_w = ((w + 15) // 16) * 16
            bucket_size = (bucket_h, bucket_w)
            size_groups[bucket_size].append(idx)
        
        # Create buckets
        buckets = []
        for target_size, indices in size_groups.items():
            if len(indices) > 0:
                bucket = SizeBucket(target_size, self.size_tolerance)
                bucket.indices = indices
                buckets.append(bucket)
        
        # Sort buckets by size for consistent ordering
        buckets.sort(key=lambda b: b.target_size[0] * b.target_size[1])
        
        print(f"Created {len(buckets)} size buckets:")
        for i, bucket in enumerate(buckets):
            print(f"  Bucket {i}: {bucket.target_size} with {len(bucket)} images")
        
        return buckets
    
    def __iter__(self):
        """Generate batches of indices"""
        # Shuffle indices within each bucket
        for bucket in self.buckets:
            random.shuffle(bucket.indices)
        
        # Generate batches from each bucket
        for bucket in self.buckets:
            indices = bucket.indices.copy()
            
            while len(indices) >= self.batch_size:
                batch = indices[:self.batch_size]
                indices = indices[self.batch_size:]
                yield batch
            
            # Handle remaining indices
            if len(indices) > 0 and not self.drop_last:
                yield indices
    
    def __len__(self):
        """Return the number of batches"""
        total_batches = 0
        for bucket in self.buckets:
            bucket_batches = len(bucket) // self.batch_size
            if len(bucket) % self.batch_size != 0 and not self.drop_last:
                bucket_batches += 1
            total_batches += bucket_batches
        return total_batches

def smart_collate_fn(batch):
    """Custom collate function that pads images to the same size within a batch"""
    images, masks, filenames = zip(*batch)
    
    # Find the maximum dimensions in this batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Pad all images to the maximum size
    padded_images = []
    padded_masks = []
    
    for img, mask in zip(images, masks):
        # Calculate padding needed
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        # Apply padding (left, right, top, bottom)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Pad image
        padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        padded_images.append(padded_img)
        
        # Pad mask
        if mask.dim() == 2:
            padded_mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        else:
            padded_mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        padded_masks.append(padded_mask)
    
    # Stack into batches
    batch_images = torch.stack(padded_images, dim=0)
    batch_masks = torch.stack(padded_masks, dim=0)
    
    return batch_images, batch_masks, filenames

class AccumulatedGradientTrainer:
    """Trainer that accumulates gradients to simulate larger batch sizes"""
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def train_step(self, x, masks=None, gradient_clip_norm=None):
        """Training step with gradient accumulation"""
        # Forward pass
        x_reconstructed, vq_loss, _, _ = self.model(x)
        
        # Compute loss
        total_loss, reconstruction_loss, perceptual_loss = self.model.compute_loss(
            x_reconstructed, x, vq_loss, masks
        )
        
        # Scale loss by accumulation steps
        scaled_loss = total_loss / self.accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        self.step_count += 1
        
        # Update weights every accumulation_steps
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss, reconstruction_loss, vq_loss, perceptual_loss
    
    def finalize_step(self, gradient_clip_norm=None):
        """Finalize any remaining gradients"""
        if self.step_count % self.accumulation_steps != 0:
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

def create_smart_dataloader(dataset, batch_size, shuffle=True, num_workers=4, 
                          size_tolerance=0.3, pin_memory=True):
    """Create a smart dataloader with size-based batching"""
    
    if shuffle:
        # Use smart batch sampler
        batch_sampler = SmartBatchSampler(
            dataset, 
            batch_size=batch_size, 
            drop_last=False,
            size_tolerance=size_tolerance
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=smart_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        # For validation, use regular batching with smart collate
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=smart_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

# Optimizations for batch_size=1 training
class SingleImageOptimizer:
    """Optimizations for efficient single-image training"""
    
    @staticmethod
    def optimize_model_for_single_batch(model):
        """Apply optimizations for single-batch training"""
        # Disable batch normalization momentum for single samples
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # Use running statistics instead of batch statistics
                module.momentum = 0.1
                module.track_running_stats = True
        
        return model
    
    @staticmethod
    def create_efficient_single_dataloader(dataset, num_workers=2):
        """Create an efficient dataloader for single-image batches"""
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )

def analyze_dataset_sizes(dataset, max_samples=100):
    """Analyze dataset to understand size distribution"""
    print("Analyzing dataset size distribution...")
    
    sizes = []
    sample_indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
    
    for idx in sample_indices:
        try:
            image, _, filename = dataset[idx]
            size = (image.shape[1], image.shape[2])  # H, W
            sizes.append(size)
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
    
    if not sizes:
        return
    
    # Analyze size distribution
    heights = [s[0] for s in sizes]
    widths = [s[1] for s in sizes]
    areas = [h * w for h, w in sizes]
    
    print(f"Size analysis from {len(sizes)} samples:")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
    print(f"  Area:   min={min(areas)}, max={max(areas)}, mean={np.mean(areas):.0f}")
    
    # Suggest optimal batch sizes
    unique_sizes = list(set(sizes))
    print(f"  Unique sizes: {len(unique_sizes)}")
    
    if len(unique_sizes) < 10:
        print("  Recommendation: Use smart batching with batch_size=8-16")
    elif len(unique_sizes) < 50:
        print("  Recommendation: Use smart batching with batch_size=4-8")
    else:
        print("  Recommendation: Use batch_size=1 with gradient accumulation")
    
    return {
        'sizes': sizes,
        'unique_sizes': len(unique_sizes),
        'height_range': (min(heights), max(heights)),
        'width_range': (min(widths), max(widths))
    }