#!/usr/bin/env python
"""
Demo script for Albumentations augmentations with YOLO
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import albumentations as A

# Import project modules
from config import *
from utils.augmentations import (
    get_train_transforms,
    apply_augmentations,
    create_custom_dataset_with_augmentations
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Demo Albumentations augmentations for YOLO")
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to YOLO format labels')
    parser.add_argument('--output-dir', type=str, default='augmentation_demo',
                        help='Directory to save augmented images')
    parser.add_argument('--num-augmentations', type=int, default=10,
                        help='Number of augmentations to generate')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='Augmentation intensity preset')
    parser.add_argument('--imgsz', type=int, default=IMAGE_SIZE,
                        help='Image size')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to YOLO dataset for batch augmentation (optional)')
    
    return parser.parse_args()


def get_augmentation_preset(preset, height=640, width=640):
    """
    Get predefined augmentation presets
    
    Args:
        preset: Preset name ('light', 'medium', 'heavy')
        height: Target image height
        width: Target image width
        
    Returns:
        Albumentations Compose object
    """
    if preset == 'light':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
            A.Resize(height=height, width=width, p=1.0),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
        ))
    
    elif preset == 'medium':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Resize(height=height, width=width, p=1.0),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
        ))
        
    elif preset == 'heavy':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
            ], p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=(3, 7), p=0.5),
                A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=15, p=0.5),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
                A.CLAHE(clip_limit=4.0, p=0.5),
            ], p=0.4),
            A.Resize(height=height, width=width, p=1.0),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
        ))
    
    # Default to medium if preset not recognized
    return get_augmentation_preset('medium', height, width)


def demonstrate_single_image(image_path, labels_path, output_dir, args):
    """
    Demonstrate augmentations on a single image
    
    Args:
        image_path: Path to input image
        labels_path: Path to YOLO format labels file
        output_dir: Directory to save augmented images
        args: Command line arguments
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    # Read labels
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    
    # Parse YOLO format labels
    bboxes = []
    class_labels = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)
    
    # Create augmentation transforms based on preset
    transforms = get_augmentation_preset(args.preset, args.imgsz, args.imgsz)
    
    # Create a grid of augmented images
    rows = int(np.ceil(args.num_augmentations / 4))
    cols = min(4, args.num_augmentations)
    
    # Create the original image + augmented images
    fig, axes = plt.subplots(rows + 1, cols, figsize=(cols * 5, (rows + 1) * 5))
    
    # If there's only one row, wrap axes in a list
    if rows == 1:
        axes = np.array([axes])
    
    # Display original image with annotations in the first row
    h, w = image.shape[:2]
    class_names = {i: name for i, name in enumerate(config.CLASS_COLORS.keys())}
    
    # Convert to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization of original image
    from utils.visualization import visualize_detection
    detections = []
    for bbox, class_id in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        detections.append([x_center, y_center, width, height, 1.0, class_id])
    
    original_with_boxes = visualize_detection(image_rgb, detections, class_names)
    
    # Save the original with boxes
    cv2.imwrite(str(output_dir / 'original.jpg'), cv2.cvtColor(original_with_boxes, cv2.COLOR_RGB2BGR))
    
    # Display original in first spot
    for i in range(cols):
        if i == 0:
            axes[0, i].imshow(original_with_boxes)
            axes[0, i].set_title('Original')
        axes[0, i].axis('off')
    
    # Generate and display augmentations
    for i in range(args.num_augmentations):
        row = (i // cols) + 1
        col = i % cols
        
        # Apply augmentation
        vis_path = output_dir / f"aug_{i+1}.jpg"
        augmented_img, augmented_bboxes, augmented_class_labels = apply_augmentations(
            image, bboxes, class_labels, transforms, str(vis_path)
        )
        
        # Read the saved visualization
        vis_img = cv2.imread(str(vis_path))
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Display the augmentation
        axes[row, col].imshow(vis_img)
        axes[row, col].set_title(f'Augmentation {i+1}')
        axes[row, col].axis('off')
    
    # Hide any unused axes
    for i in range(args.num_augmentations, rows * cols):
        row = (i // cols) + 1
        col = i % cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis('off')
    
    # Save the grid visualization
    plt.tight_layout()
    plt.savefig(str(output_dir / 'augmentations_grid.png'), dpi=150)
    print(f"Saved augmentation grid to {output_dir / 'augmentations_grid.png'}")
    plt.close()


def main():
    """Main function"""
    args = parse_args()
    
    # Print augmentation configuration
    print("=" * 50)
    print("Albumentations Demo Configuration:")
    print(f"Image: {args.image}")
    print(f"Labels: {args.labels}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Augmentation Preset: {args.preset}")
    print(f"Number of Augmentations: {args.num_augmentations}")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we're doing batch augmentation on a dataset
    if args.dataset:
        print(f"Performing batch augmentation on dataset: {args.dataset}")
        dataset_path = Path(args.dataset)
        
        # Create augmented dataset
        augmented_dataset_path = create_custom_dataset_with_augmentations(
            dataset_path,
            num_augmentations=args.num_augmentations,
            visualize=True
        )
        print(f"Augmented dataset created at: {augmented_dataset_path}")
    else:
        # Single image demonstration
        demonstrate_single_image(args.image, args.labels, args.output_dir, args)
    
    print("Augmentation demonstration complete!")
    

if __name__ == "__main__":
    main() 