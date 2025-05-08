"""
Configuration parameters for YOLO project
"""

# Paths
DATA_PATH = 'data/processed'
DATASET_CONFIG = 'data/dataset.yaml'
MODELS_PATH = 'models'
RESULTS_PATH = 'results'

# Training parameters
PRETRAINED_MODEL = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 'auto'  # 'cuda:0', 'cpu', or 'auto'
WORKERS = 8

# Augmentation parameters
USE_ALBUMENTATION = True
ALBUMENTATION_PRESET = 'medium'  # 'light', 'medium', 'heavy'
ALBUMENTATION_AUGMENTATIONS_PER_IMAGE = 5  # Number of augmented versions to create per image
ALBUMENTATION_VISUALIZE = False  # Whether to save visualizations of augmented images

# Inference parameters
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300

# Visualization
CLASS_COLORS = {
    'person': (0, 255, 0),     # Green
    'car': (255, 0, 0),        # Blue
    'motorcycle': (0, 0, 255), # Red
    'bicycle': (255, 255, 0),  # Cyan
    'truck': (255, 0, 255),    # Magenta
}

# Export formats
EXPORT_FORMAT = 'torchscript'  # Options: onnx, openvino, engine, coreml, etc. 