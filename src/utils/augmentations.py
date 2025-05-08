"""
Augmentation utilities for YOLO training using Albumentations
"""

import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import os
import sys

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_train_transforms(height=640, width=640, p=1.0, bbox_format='yolo'):
    """
    Get training augmentations pipeline using Albumentations
    
    Args:
        height: Target height
        width: Target width
        p: Probability of applying augmentations
        bbox_format: Format of bounding boxes ('yolo', 'pascal_voc', 'albumentations')
        
    Returns:
        Albumentations Compose object with transformations
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        
        # Color transformations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.8),
        ], p=0.8),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.5),
        
        # Weather and effects
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=15, p=0.2),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2),
        ], p=0.3),
        
        # Image quality
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
            A.ToGray(p=0.5),
        ], p=0.3),
        
        # Normalize at the end (optional, Ultralytics YOLO handles this internally)
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ], bbox_params=A.BboxParams(
        format=bbox_format,
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))


def get_val_transforms(height=640, width=640, p=1.0, bbox_format='yolo'):
    """
    Get validation augmentations pipeline (usually just resize)
    
    Args:
        height: Target height
        width: Target width
        p: Probability of applying augmentations
        bbox_format: Format of bounding boxes ('yolo', 'pascal_voc', 'albumentations')
        
    Returns:
        Albumentations Compose object with transformations
    """
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        # Normalize at the end (optional, Ultralytics YOLO handles this internally)
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ], bbox_params=A.BboxParams(
        format=bbox_format,
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))


def get_test_transforms(height=640, width=640, p=1.0):
    """
    Get test-time augmentations pipeline (usually just resize, no bbox params)
    
    Args:
        height: Target height
        width: Target width
        p: Probability of applying augmentations
        
    Returns:
        Albumentations Compose object with transformations
    """
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        # Normalize at the end (optional, Ultralytics YOLO handles this internally)
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ])


def apply_augmentations(image, bboxes, class_labels, transforms, visualization_path=None):
    """
    Apply augmentations to an image and bounding boxes
    
    Args:
        image: Input image (numpy array in BGR format)
        bboxes: List of bounding boxes [[x_center, y_center, width, height], ...] in YOLO format (normalized)
        class_labels: List of class labels corresponding to bboxes
        transforms: Albumentations transforms to apply
        visualization_path: Optional path to save visualization of augmented image
        
    Returns:
        Tuple of (augmented_image, augmented_bboxes, augmented_class_labels)
    """
    # Apply transformations
    transformed = transforms(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    augmented_class_labels = transformed['class_labels']
    
    # Optionally visualize and save the augmented image
    if visualization_path:
        # Convert to format for visualization
        import matplotlib.pyplot as plt
        from visualization import visualize_detection
        
        h, w = augmented_image.shape[:2]
        detections = []
        
        for i, (bbox, class_id) in enumerate(zip(augmented_bboxes, augmented_class_labels)):
            x_center, y_center, width, height = bbox
            detections.append([x_center, y_center, width, height, 1.0, class_id])  # Conf=1.0
            
        # Use the visualization function
        class_names = {i: name for i, name in enumerate(config.CLASS_COLORS.keys())}
        
        # Convert BGR to RGB for visualization
        rgb_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
        result_img = visualize_detection(rgb_image, detections, class_names)
        
        # Save the visualization
        cv2.imwrite(visualization_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return augmented_image, augmented_bboxes, augmented_class_labels


def create_custom_dataset_with_augmentations(yolo_dataset_path, num_augmentations=5, visualize=False):
    """
    Create a custom dataset with augmentations from an existing YOLO dataset
    
    Args:
        yolo_dataset_path: Path to YOLO dataset with images/ and labels/ subdirectories
        num_augmentations: Number of augmentations to generate per image
        visualize: Whether to save visualizations of augmented images
        
    Returns:
        Path to augmented dataset
    """
    from pathlib import Path
    import os
    import cv2
    import shutil
    import random
    
    # Setup directories
    dataset_path = Path(yolo_dataset_path)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    # Create augmented dataset directories
    augmented_dir = dataset_path.parent / f"{dataset_path.name}_augmented"
    augmented_images_dir = augmented_dir / 'images'
    augmented_labels_dir = augmented_dir / 'labels'
    augmented_vis_dir = augmented_dir / 'visualizations' if visualize else None
    
    # Create directories
    augmented_images_dir.mkdir(parents=True, exist_ok=True)
    augmented_labels_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        augmented_vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
    
    # Initialize transforms
    height, width = config.IMAGE_SIZE, config.IMAGE_SIZE
    transforms = get_train_transforms(height=height, width=width)
    
    # Process each image
    for img_path in image_files:
        # Get corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        # Read labels
        with open(label_path, 'r') as f:
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
        
        # Copy the original image and label
        shutil.copy(img_path, augmented_images_dir / img_path.name)
        shutil.copy(label_path, augmented_labels_dir / label_path.name)
        
        # Generate augmentations
        for i in range(num_augmentations):
            # Apply augmentations
            vis_path = augmented_vis_dir / f"{img_path.stem}_aug{i}.jpg" if visualize else None
            aug_image, aug_bboxes, aug_class_labels = apply_augmentations(
                image, bboxes, class_labels, transforms, vis_path
            )
            
            # Skip if no bounding boxes are left after augmentation
            if len(aug_bboxes) == 0:
                continue
                
            # Save augmented image
            aug_img_path = augmented_images_dir / f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(aug_img_path), aug_image)
            
            # Save augmented labels
            aug_label_path = augmented_labels_dir / f"{img_path.stem}_aug{i}.txt"
            with open(aug_label_path, 'w') as f:
                for class_id, bbox in zip(aug_class_labels, aug_bboxes):
                    x_center, y_center, width, height = bbox
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Augmented dataset created at: {augmented_dir}")
    return augmented_dir 