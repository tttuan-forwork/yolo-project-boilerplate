"""
Augmentation utilities for YOLO training using Albumentations
"""

import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import os
import sys
import torch
from torch.utils.data import Dataset
import yaml

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


class YOLOAlbumentationsDataset(Dataset):
    """
    A PyTorch Dataset that applies Albumentations augmentations on-the-fly to YOLO data
    
    This is more efficient for large datasets as it doesn't create separate augmented files.
    """
    
    def __init__(self, data_yaml, img_size=640, augment=True, augment_preset='medium', cache=False):
        """
        Initialize the dataset
        
        Args:
            data_yaml: Path to YOLO dataset.yaml file
            img_size: Target image size
            augment: Whether to apply augmentations
            augment_preset: Augmentation preset ('light', 'medium', 'heavy')
            cache: Whether to cache images in memory
        """
        self.img_size = img_size
        self.augment = augment
        self.cache = cache
        self.cached_images = {}
        
        # Load dataset config
        with open(data_yaml, 'r') as f:
            self.data_dict = yaml.safe_load(f)
        
        # Setup paths
        self.data_dir = Path(data_yaml).parent
        dataset_path = Path(self.data_dict['path'])
        if not dataset_path.is_absolute():
            dataset_path = (self.data_dir / dataset_path).resolve()
            
        self.train_img_dir = dataset_path / self.data_dict['train']
        self.train_label_dir = self.train_img_dir.parent.parent / 'labels' / self.train_img_dir.name
        
        # Get list of images and labels
        self.img_files = sorted([f for f in self.train_img_dir.glob('*.*') 
                          if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
        
        self.label_files = [self.train_label_dir / f"{img_file.stem}.txt" 
                           for img_file in self.img_files]
        
        # Setup transforms
        if self.augment:
            from .augmentations import get_train_transforms
            self.transform = self._get_preset_transforms(augment_preset, img_size, img_size)
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.1
            ))
            
        print(f"YOLOAlbumentationsDataset: {len(self.img_files)} images, augment={augment}, preset={augment_preset}")
    
    def _get_preset_transforms(self, preset, height, width):
        """Get augmentation preset transforms"""
        if preset == 'light':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.Resize(height=height, width=width, p=1.0),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.1,
            ))
        
        elif preset == 'medium':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ], p=0.3),
                A.Resize(height=height, width=width, p=1.0),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.1,
            ))
        
        elif preset == 'heavy':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                    A.RandomRain(p=1.0),
                    A.RandomShadow(p=1.0),
                    A.CLAHE(p=1.0),
                ], p=0.3),
                A.Resize(height=height, width=width, p=1.0),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.1,
            ))
        
        # Default
        return self._get_preset_transforms('medium', height, width)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """Get item with on-the-fly augmentation"""
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # Load image
        if img_path in self.cached_images and self.cache:
            img = self.cached_images[img_path]
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            if self.cache:
                self.cached_images[img_path] = img
        
        # Load labels (if exists)
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # Apply transforms
        transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        img_transformed = transformed['image']
        bboxes_transformed = transformed['bboxes']
        class_labels_transformed = transformed['class_labels']
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img_transformed.transpose(2, 0, 1)).float() / 255.0
        
        # Format labels as tensor
        labels = []
        if len(bboxes_transformed) > 0:
            for cls, bbox in zip(class_labels_transformed, bboxes_transformed):
                # YOLO format: class, x_center, y_center, width, height
                x_center, y_center, width, height = bbox
                labels.append([cls, x_center, y_center, width, height])
        
        if len(labels) > 0:
            labels_tensor = torch.tensor(labels)
        else:
            # Empty labels tensor with correct shape
            labels_tensor = torch.zeros((0, 5))
        
        return img_tensor, labels_tensor, img_path


def create_efficient_dataloaders(data_yaml, batch_size=16, img_size=640, augment=True, augment_preset='medium', 
                                workers=8, shuffle=True, cache_images=False):
    """
    Create efficient data loaders with on-the-fly augmentation for YOLO training
    
    Args:
        data_yaml: Path to YOLO dataset.yaml file
        batch_size: Batch size
        img_size: Target image size
        augment: Whether to apply augmentations
        augment_preset: Augmentation preset ('light', 'medium', 'heavy')
        workers: Number of worker threads
        shuffle: Whether to shuffle the data
        cache_images: Whether to cache images in memory (only for small datasets)
        
    Returns:
        Dictionary with train and val dataloaders
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = YOLOAlbumentationsDataset(
        data_yaml=data_yaml,
        img_size=img_size,
        augment=augment,
        augment_preset=augment_preset,
        cache=cache_images
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,  # Custom collate function to handle variable size labels
        pin_memory=True
    )
    
    return {
        'train': train_loader
    }


def collate_fn(batch):
    """
    Custom collate function for YOLO dataloaders to handle variable size labels
    
    Args:
        batch: Batch of (img, labels, path) tuples
        
    Returns:
        Tuple of (imgs, labels, paths)
    """
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, labels, paths


def patch_ultralytics_for_albumentation(data_yaml, img_size=640, augment=True, augment_preset='medium', cache=False):
    """
    Create a function to patch Ultralytics DataLoader with Albumentations
    This is an experimental method to integrate directly with Ultralytics
    
    Args:
        data_yaml: Path to YOLO dataset.yaml 
        img_size: Target image size
        augment: Whether to apply augmentations
        augment_preset: Augmentation preset
        cache: Whether to cache images
        
    Returns:
        Dict containing patched dataset
    """
    from ultralytics.data.dataset import YOLODataset
    
    class AlbumentationsYOLODataset(YOLODataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.augment_with_albumentation = augment
            self.albumentation_preset = augment_preset
            self.albumentation_transform = self._get_preset_transforms(
                augment_preset, img_size, img_size
            )
            print(f"AlbumentationsYOLODataset initialized with preset: {augment_preset}")
        
        def _get_preset_transforms(self, preset, height, width):
            """Get augmentation preset transforms"""
            if preset == 'light':
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                ], bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels'],
                    min_visibility=0.1,
                ))
            
            elif preset == 'medium':
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.3),
                    A.OneOf([
                        A.GaussNoise(p=1.0),
                        A.GaussianBlur(p=1.0),
                        A.MotionBlur(p=1.0),
                    ], p=0.3),
                ], bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels'],
                    min_visibility=0.1,
                ))
            
            elif preset == 'heavy':
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.5),
                    A.OneOf([
                        A.GaussNoise(var_limit=(10, 50), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    ], p=0.5),
                    A.OneOf([
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                        A.RandomRain(p=1.0),
                        A.RandomShadow(p=1.0),
                        A.CLAHE(p=1.0),
                    ], p=0.3),
                ], bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels'],
                    min_visibility=0.1,
                ))
            
            # Default
            return self._get_preset_transforms('medium', height, width)
        
        def get_image_and_label(self, index):
            """Override to apply Albumentations augmentation"""
            img, labels = super().get_image_and_label(index)
            
            # If training and augmentation is enabled, apply Albumentations
            if self.augment and self.augment_with_albumentation:
                # Convert labels to Albumentations format
                bboxes = []
                class_labels = []
                
                if len(labels) > 0:
                    for label in labels:
                        class_id = int(label[0])
                        x_center, y_center, width, height = label[1:5]
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
                
                # Apply Albumentations
                transformed = self.albumentation_transform(
                    image=img, 
                    bboxes=bboxes, 
                    class_labels=class_labels
                )
                
                img_transformed = transformed['image']
                bboxes_transformed = transformed['bboxes']
                class_labels_transformed = transformed['class_labels']
                
                # Convert back to YOLO format
                new_labels = []
                for cls, bbox in zip(class_labels_transformed, bboxes_transformed):
                    x_center, y_center, width, height = bbox
                    new_labels.append([cls, x_center, y_center, width, height])
                
                if len(new_labels) > 0:
                    labels = np.array(new_labels)
                else:
                    labels = np.zeros((0, 5))
                
                return img_transformed, labels
            
            return img, labels
    
    return {"AlbumentationsYOLODataset": AlbumentationsYOLODataset} 