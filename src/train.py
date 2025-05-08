#!/usr/bin/env python
"""
YOLO Training Script using Ultralytics YOLO
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

# Import project config
from config import *
from utils.augmentations import (
    create_custom_dataset_with_augmentations,
    create_efficient_dataloaders,
    patch_ultralytics_for_albumentation
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument('--data', type=str, default=DATASET_CONFIG,
                        help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default=PRETRAINED_MODEL,
                        help='Path to model.pt')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs to train for')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=IMAGE_SIZE,
                        help='Image size')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to train on (cuda device, i.e. 0 or cpu)')
    parser.add_argument('--workers', type=int, default=WORKERS,
                        help='Number of worker threads for data loading')
    parser.add_argument('--project', type=str, default=RESULTS_PATH,
                        help='Project name')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--use-albumentation', action='store_true', default=USE_ALBUMENTATION,
                        help='Use Albumentations for data augmentation')
    parser.add_argument('--albumentation-preset', type=str, default=ALBUMENTATION_PRESET,
                        help='Albumentations preset (light, medium, heavy)')
    parser.add_argument('--albumentation-per-image', type=int, default=ALBUMENTATION_AUGMENTATIONS_PER_IMAGE,
                        help='Number of augmentations per image')
    parser.add_argument('--efficient', action='store_true', default=EFFICIENT_AUGMENTATION,
                        help='Use efficient on-the-fly augmentation')
    parser.add_argument('--cache-images', action='store_true', default=CACHE_IMAGES,
                        help='Cache images in memory (for small datasets)')
    
    return parser.parse_args()


def prepare_dataset_with_augmentations(data_yaml_path, args):
    """
    Prepare dataset with Albumentations augmentations
    
    Args:
        data_yaml_path: Path to dataset.yaml
        args: Command line arguments
        
    Returns:
        Path to augmented dataset configuration
    """
    # Load the dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get the base directory
    base_dir = Path(data_yaml_path).parent
    
    # If path is relative in the yaml, make it absolute by joining with base_dir
    dataset_path = Path(data_config['path'])
    if not dataset_path.is_absolute():
        dataset_path = (base_dir / dataset_path).resolve()
    
    # Prepare paths for train data augmentation
    train_images_path = dataset_path / data_config['train']
    train_dir = train_images_path.parent.parent  # Go up two levels to get the train directory
    
    # Apply augmentations to the training set
    print(f"Applying Albumentations augmentations to training data ({args.albumentation_preset} preset)...")
    print(f"Creating {args.albumentation_per_image} augmented versions per image...")
    
    # Create augmented dataset
    augmented_train_dir = create_custom_dataset_with_augmentations(
        train_dir,
        num_augmentations=args.albumentation_per_image,
        visualize=ALBUMENTATION_VISUALIZE
    )
    
    # Create a new dataset configuration for the augmented data
    augmented_data_config = data_config.copy()
    
    # Update paths for augmented data
    augmented_data_config['path'] = str(augmented_train_dir)
    
    # Save the new dataset configuration
    augmented_yaml_path = base_dir / f"{Path(data_yaml_path).stem}_augmented.yaml"
    with open(augmented_yaml_path, 'w') as f:
        yaml.dump(augmented_data_config, f, default_flow_style=False)
    
    print(f"Augmented dataset configuration saved to: {augmented_yaml_path}")
    return augmented_yaml_path


def main():
    """Main function for training YOLO model"""
    args = parse_args()
    
    # Print training configuration
    print("=" * 50)
    print("YOLO Training Configuration:")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {args.device}")
    
    # Print augmentation configuration if enabled
    if args.use_albumentation:
        print(f"Using Albumentations: {args.use_albumentation}")
        print(f"Augmentation Preset: {args.albumentation_preset}")
        
        if args.efficient:
            print(f"Using efficient on-the-fly augmentation")
            print(f"Cache images: {args.cache_images}")
        else:
            print(f"Pre-generating augmented dataset")
            print(f"Augmentations Per Image: {args.albumentation_per_image}")
    
    print("=" * 50)
    
    # Create output directory
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up augmentations
    data_path = args.data
    
    if args.use_albumentation:
        if args.efficient:
            # Use on-the-fly augmentation - we don't need to pre-generate an augmented dataset
            print("Using efficient on-the-fly Albumentations augmentation during training...")
            
            try:
                # Prepare for monkey patching Ultralytics
                print("Preparing to integrate Albumentations with Ultralytics YOLO...")
                patched_classes = patch_ultralytics_for_albumentation(
                    data_path,
                    img_size=args.imgsz,
                    augment=True,
                    augment_preset=args.albumentation_preset,
                    cache=args.cache_images
                )
                
                # We'll use the AlbumentationsYOLODataset in Ultralytics
                # This will be injected into the model training process
                yolo_albumentation_dataset = patched_classes.get("AlbumentationsYOLODataset")
                
                # Store it for YOLO to use it
                import ultralytics.data.dataset
                ultralytics.data.dataset.YOLODataset = yolo_albumentation_dataset
                print("Successfully integrated Albumentations with Ultralytics YOLO.")
                
            except Exception as e:
                print(f"Warning: Could not set up efficient augmentation: {e}")
                print("Falling back to standard training...")
        else:
            # Pre-generate augmented dataset
            data_path = prepare_dataset_with_augmentations(args.data, args)
    
    # Load model
    model = YOLO(args.model)
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,      # Overwrite existing experiment
        pretrained=True,    # Use pretrained weights
        verbose=True        # Print verbose output
    )
    
    # Save the model
    model_path = save_dir / 'weights' / 'best.pt'
    print(f"Best model saved to {model_path}")
    
    # Evaluate the model
    metrics = model.val()
    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Visualize results if available
    from utils.visualization import plot_results
    results_csv = save_dir / 'results.csv'
    if results_csv.exists():
        plot_path = plot_results(results_csv, save_dir)
        print(f"Training plots saved to {plot_path}")
    
    return save_dir


if __name__ == "__main__":
    main() 