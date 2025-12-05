# Improved Evaluation Code with Enhanced Visualization and Analysis
# Authors: Samantha Mutiti & Rong Chi (Enhanced version)
import json, os, argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import mean_squared_error, structural_similarity as ssim
from scipy.stats import pearsonr
import tifffile as tf

from improved_dynamic_model import ImprovedDynamicCellVQVAE
from improved_train_dynamic import ImprovedDynamicCropDataset

def plot_training_history(filepath, show=True, save=True, save_dir=os.getcwd()):
    """Enhanced training history plotting"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    train_history = data.get('train', {})
    val_history = data.get('val', {})
    
    if not train_history:
        print("No training history found in the file")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot each metric
    metrics = ['total_loss', 'reconstruction_loss', 'vq_loss', 'perceptual_loss']
    titles = ['Total Loss', 'Reconstruction Loss', 'VQ Loss', 'Perceptual Loss']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        if metric in train_history:
            epochs = range(1, len(train_history[metric]) + 1)
            ax.plot(epochs, train_history[metric], label='Train', linewidth=2)
            
            if metric in val_history:
                ax.plot(epochs, val_history[metric], label='Validation', linewidth=2)
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set log scale for better visualization if needed
            if metric == 'total_loss' and max(train_history[metric]) > 1000:
                ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        save_path = os.path.join(save_dir, 'enhanced_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def compute_reconstruction_metrics(original, reconstructed):
    """Compute comprehensive reconstruction metrics"""
    metrics = {}
    
    # Convert to numpy if needed
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.cpu().numpy()
    
    # Ensure same shape
    if original.shape != reconstructed.shape:
        print(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
        return metrics
    
    # MSE
    metrics['mse'] = mean_squared_error(original.flatten(), reconstructed.flatten())
    
    # PSNR
    mse = metrics['mse']
    if mse > 0:
        max_pixel = max(original.max(), reconstructed.max())
        metrics['psnr'] = 20 * np.log10(max_pixel / np.sqrt(mse))
    else:
        metrics['psnr'] = float('inf')
    
    # Pearson correlation
    corr, p_value = pearsonr(original.flatten(), reconstructed.flatten())
    metrics['pearson_corr'] = corr
    metrics['pearson_p_value'] = p_value
    
    # SSIM (for 2D images)
    if len(original.shape) >= 2:
        if len(original.shape) == 4:  # Batch of multi-channel images
            ssim_values = []
            for b in range(original.shape[0]):
                for c in range(original.shape[1]):
                    ssim_val = ssim(original[b, c], reconstructed[b, c], data_range=original[b, c].max() - original[b, c].min())
                    ssim_values.append(ssim_val)
            metrics['ssim'] = np.mean(ssim_values)
        elif len(original.shape) == 3:  # Multi-channel image
            ssim_values = []
            for c in range(original.shape[0]):
                ssim_val = ssim(original[c], reconstructed[c], data_range=original[c].max() - original[c].min())
                ssim_values.append(ssim_val)
            metrics['ssim'] = np.mean(ssim_values)
        else:  # Single 2D image
            metrics['ssim'] = ssim(original, reconstructed, data_range=original.max() - original.min())
    
    return metrics

def model_inference(config_file, image_directory, model_path=None, device=None):
    """Enhanced model inference with comprehensive analysis"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = ImprovedDynamicCellVQVAE(
        activation=config['activation'], 
        embedding_dim=int(config['embedding_dim']),
        commitment_cost=config.get('commitment_cost', 0.25),
        reconstruction_loss_weight=config.get('reconstruction_loss_weight', 1.0),
        num_embeddings=config.get('num_embeddings', 512),
        use_perceptual_loss=config.get('use_perceptual_loss', True),
        perceptual_loss_weight=config.get('perceptual_loss_weight', 0.1)
    ).to(device)
    
    # Load model weights
    if model_path is None:
        model_path = os.path.join(config['save_directory'], 'improved_vq_vae_dynamic_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from {model_path}")

    # Load test dataset
    dataset = ImprovedDynamicCropDataset(
        file_dir=image_directory, 
        type='test',
        normalize_method=config.get('normalize_method', 'robust')
    )
    
    if len(dataset) == 0:
        print("No test images found. Trying 'val' directory...")
        dataset = ImprovedDynamicCropDataset(
            file_dir=image_directory, 
            type='val',
            normalize_method=config.get('normalize_method', 'robust')
        )
    
    if len(dataset) == 0:
        print("No validation images found. Using training images...")
        dataset = ImprovedDynamicCropDataset(
            file_dir=image_directory, 
            type='train',
            normalize_method=config.get('normalize_method', 'robust')
        )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = {
        'reconstructions': [],
        'originals': [],
        'embeddings': [],
        'embed_indices': [],
        'filenames': [],
        'metrics': []
    }
    
    print(f"Running inference on {len(dataset)} images...")
    
    with torch.no_grad():
        for i, (images, masks, filenames) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            reconstructed, vq_loss, embeddings, embed_indices = model(images)
            
            # Move to CPU
            original = images.cpu()
            reconstructed = reconstructed.cpu()
            embeddings = embeddings.cpu()
            embed_indices = embed_indices.cpu()
            
            # Compute metrics
            metrics = compute_reconstruction_metrics(original, reconstructed)
            metrics['vq_loss'] = vq_loss.item()
            
            # Store results
            results['reconstructions'].append(reconstructed)
            results['originals'].append(original)
            results['embeddings'].append(embeddings)
            results['embed_indices'].append(embed_indices)
            results['filenames'].extend(filenames)
            results['metrics'].append(metrics)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images")
    
    return results, model

def visualize_reconstructions(results, save_dir, num_samples=8):
    """Create comprehensive visualization of reconstructions"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(results['reconstructions']))
    
    for i in range(num_samples):
        original = results['originals'][i][0].numpy()  # Remove batch dimension
        reconstructed = results['reconstructions'][i][0].numpy()
        filename = results['filenames'][i]
        metrics = results['metrics'][i]
        
        # Create figure
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Reconstruction Analysis: {filename}', fontsize=14)
        
        # Plot original channels
        for c in range(4):
            axes[0, c].imshow(original[c], cmap='gray')
            axes[0, c].set_title(f'Original Ch{c+1}')
            axes[0, c].axis('off')
        
        # Plot reconstructed channels
        for c in range(4):
            axes[1, c].imshow(reconstructed[c], cmap='gray')
            axes[1, c].set_title(f'Reconstructed Ch{c+1}')
            axes[1, c].axis('off')
        
        # Plot difference maps
        for c in range(4):
            diff = np.abs(original[c] - reconstructed[c])
            im = axes[2, c].imshow(diff, cmap='hot')
            axes[2, c].set_title(f'Difference Ch{c+1}')
            axes[2, c].axis('off')
            plt.colorbar(im, ax=axes[2, c], fraction=0.046, pad=0.04)
        
        # Add metrics text
        metrics_text = f"MSE: {metrics['mse']:.6f}\n"
        metrics_text += f"PSNR: {metrics['psnr']:.2f} dB\n"
        metrics_text += f"SSIM: {metrics.get('ssim', 'N/A'):.4f}\n"
        metrics_text += f"Correlation: {metrics['pearson_corr']:.4f}\n"
        metrics_text += f"VQ Loss: {metrics['vq_loss']:.6f}"
        
        fig.text(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'reconstruction_{i+1:03d}_{filename}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def analyze_codebook_usage(results, save_dir):
    """Analyze vector quantization codebook usage"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect all embedding indices
    all_indices = []
    for embed_indices in results['embed_indices']:
        all_indices.extend(embed_indices.flatten().tolist())
    
    all_indices = np.array(all_indices)
    unique_indices, counts = np.unique(all_indices, return_counts=True)
    
    # Plot codebook usage
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(all_indices, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Codebook Usage Distribution')
    plt.xlabel('Embedding Index')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot usage frequency
    plt.subplot(2, 2, 2)
    plt.plot(unique_indices, counts, 'o-', markersize=3)
    plt.title('Embedding Usage Frequency')
    plt.xlabel('Embedding Index')
    plt.ylabel('Usage Count')
    plt.grid(True, alpha=0.3)
    
    # Usage statistics
    total_embeddings = len(results['embed_indices'][0].flatten()) if results['embed_indices'] else 0
    num_embeddings = max(all_indices) + 1 if len(all_indices) > 0 else 0
    used_embeddings = len(unique_indices)
    usage_ratio = used_embeddings / num_embeddings if num_embeddings > 0 else 0
    
    plt.subplot(2, 2, 3)
    plt.bar(['Total', 'Used', 'Unused'], 
            [num_embeddings, used_embeddings, num_embeddings - used_embeddings],
            color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('Codebook Utilization')
    plt.ylabel('Number of Embeddings')
    
    # Add statistics text
    stats_text = f"Total Embeddings: {num_embeddings}\n"
    stats_text += f"Used Embeddings: {used_embeddings}\n"
    stats_text += f"Usage Ratio: {usage_ratio:.2%}\n"
    stats_text += f"Avg Usage per Embedding: {counts.mean():.1f}\n"
    stats_text += f"Max Usage: {counts.max()}\n"
    stats_text += f"Min Usage: {counts.min()}"
    
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'codebook_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'total_embeddings': num_embeddings,
        'used_embeddings': used_embeddings,
        'usage_ratio': usage_ratio,
        'usage_stats': {
            'mean': float(counts.mean()),
            'std': float(counts.std()),
            'min': int(counts.min()),
            'max': int(counts.max())
        }
    }

def generate_comprehensive_report(results, codebook_stats, save_dir):
    """Generate a comprehensive evaluation report"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Aggregate metrics
    all_metrics = results['metrics']
    
    report = {
        'summary': {
            'num_samples': len(all_metrics),
            'avg_mse': np.mean([m['mse'] for m in all_metrics]),
            'avg_psnr': np.mean([m['psnr'] for m in all_metrics if m['psnr'] != float('inf')]),
            'avg_ssim': np.mean([m['ssim'] for m in all_metrics if 'ssim' in m]),
            'avg_correlation': np.mean([m['pearson_corr'] for m in all_metrics]),
            'avg_vq_loss': np.mean([m['vq_loss'] for m in all_metrics])
        },
        'codebook_analysis': codebook_stats,
        'detailed_metrics': all_metrics
    }
    
    # Save report
    with open(os.path.join(save_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create summary plot
    plt.figure(figsize=(15, 10))
    
    # MSE distribution
    plt.subplot(2, 3, 1)
    mse_values = [m['mse'] for m in all_metrics]
    plt.hist(mse_values, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'MSE Distribution\nMean: {np.mean(mse_values):.6f}')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # PSNR distribution
    plt.subplot(2, 3, 2)
    psnr_values = [m['psnr'] for m in all_metrics if m['psnr'] != float('inf')]
    if psnr_values:
        plt.hist(psnr_values, bins=20, alpha=0.7, edgecolor='black')
        plt.title(f'PSNR Distribution\nMean: {np.mean(psnr_values):.2f} dB')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # SSIM distribution
    plt.subplot(2, 3, 3)
    ssim_values = [m['ssim'] for m in all_metrics if 'ssim' in m]
    if ssim_values:
        plt.hist(ssim_values, bins=20, alpha=0.7, edgecolor='black')
        plt.title(f'SSIM Distribution\nMean: {np.mean(ssim_values):.4f}')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # Correlation distribution
    plt.subplot(2, 3, 4)
    corr_values = [m['pearson_corr'] for m in all_metrics]
    plt.hist(corr_values, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'Correlation Distribution\nMean: {np.mean(corr_values):.4f}')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # VQ Loss distribution
    plt.subplot(2, 3, 5)
    vq_loss_values = [m['vq_loss'] for m in all_metrics]
    plt.hist(vq_loss_values, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f'VQ Loss Distribution\nMean: {np.mean(vq_loss_values):.6f}')
    plt.xlabel('VQ Loss')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    summary_text = f"Evaluation Summary\n\n"
    summary_text += f"Samples: {report['summary']['num_samples']}\n"
    summary_text += f"Avg MSE: {report['summary']['avg_mse']:.6f}\n"
    summary_text += f"Avg PSNR: {report['summary']['avg_psnr']:.2f} dB\n"
    summary_text += f"Avg SSIM: {report['summary']['avg_ssim']:.4f}\n"
    summary_text += f"Avg Correlation: {report['summary']['avg_correlation']:.4f}\n"
    summary_text += f"Avg VQ Loss: {report['summary']['avg_vq_loss']:.6f}\n\n"
    summary_text += f"Codebook Usage: {codebook_stats['usage_ratio']:.1%}\n"
    summary_text += f"Used Embeddings: {codebook_stats['used_embeddings']}/{codebook_stats['total_embeddings']}"
    
    plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive evaluation report saved to {save_dir}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced VQ-VAE Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--model_path', type=str, help='Path to model weights (optional)')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--plot_history', action='store_true', help='Plot training history')
    parser.add_argument('--history_file', type=str, help='Path to training history JSON file')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Plot training history if requested
    if args.plot_history and args.history_file:
        plot_training_history(args.history_file, show=False, save=True, save_dir=args.save_dir)
    
    # Run inference and evaluation
    print("Running model inference...")
    results, model = model_inference(args.config, args.data_dir, args.model_path)
    
    print("Creating visualizations...")
    visualize_reconstructions(results, os.path.join(args.save_dir, 'reconstructions'), args.num_samples)
    
    print("Analyzing codebook usage...")
    codebook_stats = analyze_codebook_usage(results, args.save_dir)
    
    print("Generating comprehensive report...")
    report = generate_comprehensive_report(results, codebook_stats, args.save_dir)
    
    print(f"Evaluation complete! Results saved to {args.save_dir}")
    print(f"Average MSE: {report['summary']['avg_mse']:.6f}")
    print(f"Average PSNR: {report['summary']['avg_psnr']:.2f} dB")
    print(f"Average SSIM: {report['summary']['avg_ssim']:.4f}")
    print(f"Codebook usage: {codebook_stats['usage_ratio']:.1%}")