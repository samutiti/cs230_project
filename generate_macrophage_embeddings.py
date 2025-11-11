#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
from tqdm import tqdm

import inference
import image_utils
from vit_model import ViTPoolClassifier
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Configuration - Using bg model with proper channel mapping
    # Channel mapping: Ch0: H2B-mTurq (nuclear) -> b, Ch2: pHrodoRed (ramos, pH sensitive) -> g
    model_channels = "bg"  # Using bg (2 channels: blue/nuclei + green/protein) with correct channel assignment
    model_type = "mae_contrast_supcon_model"
    gpu = 0 if torch.cuda.is_available() else -1
    
    # Paths
    images_dir = "/scratch/users/samutiti/livecell/images/macrophage_crops/crops"
    labels_path = "/scratch/users/samutiti/livecell/images/macrophage_crops/phage_labels.json"
    output_dir = "/scratch/users/samutiti/SubCellPortable/macrophage_embeddings"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load labels
    logger.info("Loading labels...")
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    # Map label files to actual image files (remove _V suffix)
    image_to_labels = {}
    for label_file, labels in labels_data.items():
        # Remove _V.tiff suffix and replace with .tiff
        actual_image = label_file.replace('_V.tiff', '.tiff')
        image_to_labels[actual_image] = labels
    
    logger.info(f"Found {len(image_to_labels)} labeled images")
    
    # Get list of actual image files that have labels
    image_files = []
    for img_name in image_to_labels.keys():
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            image_files.append(img_name)
        else:
            logger.warning(f"Image file not found: {img_path}")
    
    logger.info(f"Found {len(image_files)} existing image files with labels")
    
    # Load SubCell model
    logger.info("Loading SubCell model...")
    model_config_path = os.path.join("models", model_channels, model_type, "model_config.yaml")
    
    with open(model_config_path, 'r') as f:
        model_config_file = yaml.safe_load(f)
    
    classifier_paths = model_config_file.get("classifier_paths")
    encoder_path = model_config_file["encoder_path"]
    model_config = model_config_file.get("model_config")
    
    # Check if model files exist
    missing_files = []
    if classifier_paths:
        for cp in classifier_paths:
            if not os.path.exists(cp):
                missing_files.append(cp)
    if not os.path.exists(encoder_path):
        missing_files.append(encoder_path)
    
    if missing_files:
        logger.error(f"Missing model files: {missing_files}")
        logger.error("Please run the model download process first or check model paths")
        return
    
    # Initialize model
    model = ViTPoolClassifier(model_config)
    model.load_model_dict(encoder_path, classifier_paths)
    model.eval()
    
    # Set device
    if torch.cuda.is_available() and gpu != -1:
        device = torch.device(f"cuda:{gpu}")
        logger.info(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    model.to(device)
    
    # Process images and generate embeddings
    logger.info("Generating embeddings...")
    embeddings_list = []
    labels_list = []
    filenames_list = []
    
    for img_name in tqdm(image_files, desc="Processing images"):
        try:
            img_path = os.path.join(images_dir, img_name)
            
            # Load 4-channel image and extract relevant channels for bg model
            cell_data = []
            # Load the full multi-channel image
            import cv2
            multi_channel_img = cv2.imread(img_path, -1)
            
            if multi_channel_img.ndim != 3 or multi_channel_img.shape[2] != 4:
                logger.warning(f"Expected 4-channel image, got shape {multi_channel_img.shape} for {img_name}")
                continue
            
            # Channel mapping for macrophage data:
            # Ch0: H2B-mTurq (nuclear) -> b channel (blue/nuclei)
            # Ch1: mVenus-CaaX (cyto) -> could be used but less relevant for phagocytosis
            # Ch2: pHrodoRed (ramos, pH sensitive) -> g channel (green/protein) - MOST IMPORTANT
            # Ch3: H2B-iRFP (ramos nuclear) -> less relevant
            
            b_channel = multi_channel_img[:, :, 0]  # Ch0: H2B-mTurq (nuclear)
            g_channel = multi_channel_img[:, :, 2]  # Ch2: pHrodoRed (pH sensitive, lysosomal)
            
            # For bg model: b channel (nuclei) and g channel (protein/phagocytosis signal)
            cell_data.append([b_channel])  # b channel - nuclear
            cell_data.append([g_channel])  # g channel - pHrodoRed (phagocytosis signal)
            
            # Generate embedding
            output_path = os.path.join(output_dir, img_name.replace('.tiff', ''))
            embedding, probabilities = inference.run_model(
                model, cell_data, device, output_path
            )
            
            # Store results
            embeddings_list.append(embedding)
            labels_list.append(image_to_labels[img_name])
            filenames_list.append(img_name)
            
        except Exception as e:
            logger.error(f"Error processing {img_name}: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {len(embeddings_list)} images")
    
    # Save combined results
    embeddings_array = np.array(embeddings_list)
    
    # Create DataFrame with embeddings and metadata
    embedding_df = pd.DataFrame(embeddings_array, columns=[f'emb_{i}' for i in range(embeddings_array.shape[1])])
    embedding_df['filename'] = filenames_list
    embedding_df['labels'] = labels_list
    
    # Save to files
    embeddings_file = os.path.join(output_dir, 'macrophage_embeddings.npy')
    metadata_file = os.path.join(output_dir, 'macrophage_metadata.json')
    combined_file = os.path.join(output_dir, 'macrophage_embeddings_with_metadata.csv')
    
    np.save(embeddings_file, embeddings_array)
    embedding_df.to_csv(combined_file, index=False)
    
    # Save metadata separately for easier loading
    metadata = {
        'filenames': filenames_list,
        'labels': labels_list
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved embeddings to: {embeddings_file}")
    logger.info(f"Saved metadata to: {metadata_file}")
    logger.info(f"Saved combined data to: {combined_file}")
    
    # Print label distribution
    all_labels = []
    for label_list in labels_list:
        all_labels.extend(label_list)
    
    from collections import Counter
    label_counts = Counter(all_labels)
    logger.info("Label distribution:")
    for label, count in label_counts.most_common():
        logger.info(f"  {label}: {count}")

if __name__ == "__main__":
    main()