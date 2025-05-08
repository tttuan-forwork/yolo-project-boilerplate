# YOLO Project

A computer vision project using Ultralytics YOLO for object detection, training, and inference.

## Project Structure

```
yolo-project/
├── data/                  # Dataset storage
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed datasets in YOLO format
│   └── dataset.yaml       # Dataset configuration
├── models/                # Saved model weights
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── train.py           # Training script
│   ├── detect.py          # Inference script
│   ├── augment_demo.py    # Augmentation demonstration script
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── processing.py  # Data processing utilities
│   │   ├── visualization.py # Visualization utilities
│   │   └── augmentations.py # Data augmentation utilities
│   └── config.py          # Configuration parameters
├── results/               # Model outputs and visualizations
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

```bash
python src/train.py --data data/dataset.yaml --model yolov8n.pt --epochs 100
```

With Albumentations augmentation:
```bash
python src/train.py --data data/dataset.yaml --model yolov8n.pt --epochs 100 --use-albumentation --albumentation-preset medium --albumentation-per-image 5
```

### Inference

```bash
python src/detect.py --model models/best.pt --source path/to/images
```

### Data Augmentation

The project includes support for powerful augmentations using the Albumentations library:

```bash
# Demonstrate augmentations on a single image
python src/augment_demo.py --image path/to/image.jpg --labels path/to/labels.txt --output-dir augmentation_demo --preset medium

# Augment an entire dataset
python src/augment_demo.py --dataset path/to/dataset --num-augmentations 5 --preset heavy
```

Available augmentation presets:
- `light`: Basic augmentations (flips, brightness/contrast)
- `medium`: Moderate augmentations (flips, shifts, rotations, color adjustments, blur)
- `heavy`: Intensive augmentations (all medium plus weather effects, noise, advanced transformations)

## Data Format

This project uses the standard YOLO dataset format:
- Each image has a corresponding .txt annotation file
- Each line in the .txt file represents: `class x_center y_center width height`
- All values are normalized to [0, 1]

See `data/dataset.yaml` for configuration details. 