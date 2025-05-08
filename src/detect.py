#!/usr/bin/env python
"""
YOLO Inference Script using Ultralytics YOLO
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Import project config
from config import *
from utils.visualization import visualize_detection


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run YOLO inference")
    parser.add_argument('--model', type=str, default=os.path.join(MODELS_PATH, 'best.pt'),
                        help='Path to model.pt')
    parser.add_argument('--source', type=str, required=True,
                        help='Source for detection (file, folder, 0 for webcam)')
    parser.add_argument('--conf', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=IOU_THRESHOLD,
                        help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=MAX_DETECTIONS,
                        help='Maximum detections per image')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device to run inference on (cuda device, i.e. 0 or cpu)')
    parser.add_argument('--view-img', action='store_true',
                        help='Show results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in --save-txt labels')
    parser.add_argument('--project', type=str, default=RESULTS_PATH,
                        help='Save results to project/name')
    parser.add_argument('--name', type=str, default='exp',
                        help='Save results to project/name')
    parser.add_argument('--imgsz', type=int, default=IMAGE_SIZE,
                        help='Inference size (pixels)')
    
    return parser.parse_args()


def process_image(model, img_path, conf, iou, max_det, view_img, save_dir):
    """Process a single image with YOLO model"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return None
    
    # Run inference
    results = model(img, conf=conf, iou=iou, max_det=max_det)
    
    # Process results
    for result in results:
        # Extract detections
        boxes = result.boxes.cpu().numpy()
        class_names = result.names
        
        # Convert detections to format for visualization
        detections = []
        for box in boxes:
            # Extract box data
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Convert to YOLO format
            h, w = img.shape[:2]
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            detections.append([x_center, y_center, width, height, conf, cls])
        
        # Visualize detections
        img_result = visualize_detection(img, detections, class_names)
        
        # Display results
        if view_img:
            cv2.imshow(Path(img_path).name, img_result)
            cv2.waitKey(0)
        
        # Save results
        if save_dir:
            save_path = save_dir / Path(img_path).name
            cv2.imwrite(str(save_path), img_result)
            print(f"Result saved to {save_path}")
    
    return img_result


def process_video(model, video_path, conf, iou, max_det, view_img, save_dir):
    """Process a video with YOLO model"""
    # Open video file or webcam
    if video_path.isnumeric():
        cap = cv2.VideoCapture(int(video_path))
        video_path = f"webcam_{int(video_path)}"
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer if saving
    if save_dir:
        save_path = save_dir / f"{Path(video_path).stem}_result.mp4"
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
    
    # Process video frames
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Run inference
        results = model(frame, conf=conf, iou=iou, max_det=max_det)
        
        # Process results
        for result in results:
            # Extract detections
            boxes = result.boxes.cpu().numpy()
            class_names = result.names
            
            # Convert detections to format for visualization
            detections = []
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                
                # Convert to YOLO format
                h, w = frame.shape[:2]
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                detections.append([x_center, y_center, width, height, conf, cls])
            
            # Visualize detections
            frame_result = visualize_detection(frame, detections, class_names)
            
            # Display results
            if view_img:
                cv2.imshow("Detection", frame_result)
                if cv2.waitKey(1) == ord('q'):
                    break
            
            # Save results
            if save_dir and writer:
                writer.write(frame_result)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    if save_dir and writer:
        writer.release()
        print(f"Result saved to {save_path}")
    
    cv2.destroyAllWindows()


def main():
    """Main function for YOLO inference"""
    args = parse_args()
    
    # Print inference configuration
    print("=" * 50)
    print("YOLO Inference Configuration:")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"IoU Threshold: {args.iou}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Create output directory
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(args.model)
    
    # Determine source type
    source = args.source
    if source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        # Video/stream source
        process_video(
            model, source, args.conf, args.iou, args.max_det,
            args.view_img, save_dir
        )
    elif os.path.isfile(source) and source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video file
        process_video(
            model, source, args.conf, args.iou, args.max_det,
            args.view_img, save_dir
        )
    elif os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Image file
        process_image(
            model, source, args.conf, args.iou, args.max_det,
            args.view_img, save_dir
        )
    elif os.path.isdir(source):
        # Directory
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for root, _, files in os.walk(source):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_exts):
                    img_path = os.path.join(root, file)
                    process_image(
                        model, img_path, args.conf, args.iou, args.max_det,
                        args.view_img, save_dir
                    )
    else:
        print(f"Error: Source '{source}' is not supported")
        return
    
    print(f"Results saved to {save_dir}")
    

if __name__ == "__main__":
    main() 