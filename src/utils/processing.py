"""
Data processing utilities for YOLO project
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import yaml


def create_dataset_structure(root_dir='data/processed'):
    """
    Create the standard YOLO dataset directory structure
    
    Args:
        root_dir: Root directory for the processed dataset
    """
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(root_dir, split, subdir), exist_ok=True)
    print(f"Created dataset structure at {root_dir}")


def convert_to_yolo_format(annotation_dict, img_width, img_height):
    """
    Convert bounding box annotations to YOLO format
    
    Args:
        annotation_dict: Dictionary with keys 'class_id', 'x_min', 'y_min', 'x_max', 'y_max'
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        List of strings in YOLO format: [class_id center_x center_y width height]
    """
    yolo_annotations = []
    
    for ann in annotation_dict:
        class_id = ann['class_id']
        x_min, y_min = ann['x_min'], ann['y_min']
        x_max, y_max = ann['x_max'], ann['y_max']
        
        # Convert to YOLO format (normalized)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Ensure values are between 0 and 1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def split_dataset(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True):
    """
    Split a dataset into train, validation, and test sets
    
    Args:
        source_dir: Directory containing images and annotations
        target_dir: Directory where the split dataset will be stored
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        shuffle: Whether to shuffle the data before splitting
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Create destination structure
    create_dataset_structure(target_dir)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(source_dir, 'images')) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if shuffle:
        random.shuffle(image_files)
    
    # Calculate split indices
    num_samples = len(image_files)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train+num_val]
    test_files = image_files[num_train+num_val:]
    
    # Copy files to respective directories
    for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for file in files:
            # Copy image
            shutil.copy2(
                os.path.join(source_dir, 'images', file),
                os.path.join(target_dir, split, 'images', file)
            )
            
            # Copy corresponding label if it exists
            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(source_dir, 'labels', label_file)
            if os.path.exists(label_path):
                shutil.copy2(
                    label_path,
                    os.path.join(target_dir, split, 'labels', label_file)
                )
    
    # Create counts
    counts = {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files),
        'total': num_samples
    }
    
    print(f"Dataset split complete: {counts}")
    return counts 