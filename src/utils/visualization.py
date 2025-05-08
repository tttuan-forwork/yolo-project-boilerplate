"""
Visualization utilities for YOLO project
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os
import sys

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    Plot one bounding box on image
    
    Args:
        x: Bounding box coordinates [x1, y1, x2, y2]
        img: Image to plot on
        color: Box color
        label: Box label
        line_thickness: Line thickness
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def visualize_detection(image, detections, class_names, colors=None, conf_threshold=0.25):
    """
    Visualize detections on an image
    
    Args:
        image: Input image (numpy array)
        detections: YOLO detections
        class_names: Dictionary mapping class IDs to names
        colors: Dictionary mapping class names to colors
        conf_threshold: Confidence threshold for displaying detections
        
    Returns:
        Image with detections
    """
    colors = colors or config.CLASS_COLORS
    img_copy = image.copy()
    
    if len(detections) == 0:
        return img_copy
    
    for detection in detections:
        # YOLO format: [x_center, y_center, width, height, confidence, class_id]
        if isinstance(detection, list) and len(detection) >= 6:
            # Unpack detection
            x_center, y_center, width, height, confidence, class_id = detection[:6]
            
            if confidence < conf_threshold:
                continue
                
            # Convert to top-left, bottom-right format
            h, w = img_copy.shape[:2]
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            # Get class name and color
            class_id = int(class_id)
            class_name = class_names.get(class_id, f"class_{class_id}")
            color = colors.get(class_name, (0, 255, 0))
            
            # Create label
            label = f"{class_name} {confidence:.2f}"
            
            # Draw box and label
            plot_one_box([x1, y1, x2, y2], img_copy, color, label)
    
    return img_copy


def plot_results(results_file, save_dir=None):
    """
    Plot training results from Ultralytics results CSV
    
    Args:
        results_file: Path to results.csv
        save_dir: Directory to save plots
    """
    import pandas as pd
    
    save_dir = Path(save_dir or '')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    # Plot metrics
    metrics = ['box_loss', 'cls_loss', 'dfl_loss', 'precision', 'recall', 'mAP50', 'mAP50-95']
    
    for i, metric in enumerate(metrics[:6]):
        if metric in results.columns:
            axs[i].plot(results['epoch'], results[metric], label=f'train_{metric}', linewidth=2)
            if f'val_{metric}' in results.columns:
                axs[i].plot(results['epoch'], results[f'val_{metric}'], label=f'val_{metric}', linewidth=2)
            axs[i].set_title(metric)
            axs[i].set_xlabel('Epoch')
            axs[i].grid()
            axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_results.png', dpi=200)
    plt.close()
    
    return save_dir / 'training_results.png' 