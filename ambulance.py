from ultralytics import YOLO
from collections import Counter
import argparse
import os
import os.path as osp
import time
import torch
import cv2
import emoji
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print('\033[1m' + '\033[91m' + "Welcome to the Smart Traffic Management System with Ambulance Priority...\n")

from util.dynamic_signal_switching import switch_signal
from util.dynamic_signal_switching import avg_signal_oc_time

def arg_parse():
    parser = argparse.ArgumentParser(
        description='YOLOv8 Vehicle Detection with Ambulance Priority')
    parser.add_argument("--images",
                        dest='images',
                        help="Directory containing 4 lane images",
                        default="/Volumes/D/Aryan/Codes/ASEP_Project_(SEM-01)/ambulance final results/ASEP_ambulance/lanes",
                        type=str)
    parser.add_argument("--confidence_score",
                        dest="confidence",
                        help="Confidence Score to filter Vehicle Prediction",
                        default=0.1)
    parser.add_argument("--model",
                        dest='model',
                        help="Path to trained YOLOv8 model",
                        default="/Volumes/D/Aryan/Codes/ASEP_Project_(SEM-01)/ambulance final results/detect/train3/weights/best.pt",
                        type=str)
    return parser.parse_args()

class LaneStatus:
    def __init__(self):
        self.vehicle_count = 0
        self.has_ambulance = False
        self.processing_time = 0
        self.detections = {}

def process_lanes(model, image_paths, confidence):
    """Process all four lanes and return their status"""
    lane_statuses = []
    print("\nLoaded YOLO Model Classes:", model.names)

    for image_path in image_paths:
        status = LaneStatus()
        start = time.time()
        results = model(image_path, conf=confidence)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names.get(cls, "unknown")

                print(f"Detected {class_name} with confidence {conf:.2f}")
                
                if class_name == 'ambulance':
                    status.has_ambulance = True
                status.detections[class_name] = status.detections.get(class_name, 0) + 1
                if class_name != 'ambulance':  # Don't count ambulance in vehicle count
                    status.vehicle_count += 1
        
        status.processing_time = time.time() - start
        lane_statuses.append(status)
    
    return lane_statuses

def determine_priority_lane(lane_statuses):
    """Determine which lane gets priority based on ambulance presence and vehicle count"""
    for i, status in enumerate(lane_statuses):
        if status.has_ambulance:
            return i + 1, "Ambulance detected"

    max_count = 0
    priority_lane = 1
    for i, status in enumerate(lane_statuses):
        if status.vehicle_count > max_count:
            max_count = status.vehicle_count
            priority_lane = i + 1
    return priority_lane, "Highest traffic density"

def main():
    args = arg_parse()
    images_path = args.images
    confidence = float(args.confidence)

    # Load trained YOLOv8 model
    try:
        model = YOLO(args.model)
        print('\033[0m' + "YOLOv8 Neural Network Successfully Loaded..." + u'\N{check mark}')
        print("Model Classes:", model.names)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Verify we have exactly 4 images
    image_paths = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_paths) != 4:
        print(f"Error: Expected 4 lane images, found {len(image_paths)}")
        return
    
    image_paths = [os.path.join(images_path, img) for img in sorted(image_paths)]
    
    print('\033[1m' + '\033[92m' + 
          "Analyzing traffic and detecting ambulances..." + 
          '\033[0m' + u'\N{check mark}')

    # Process all lanes
    lane_statuses = process_lanes(model, image_paths, confidence)
    
    print("\nLANE ANALYSIS:")
    print("-" * 100)
    for i, status in enumerate(lane_statuses):
        print(f"\nLane {i+1}:")
        print(f"Total Vehicles: {status.vehicle_count}")
        print(f"Ambulance Present: {'Yes' if status.has_ambulance else 'No'}")
        print("Vehicle Distribution:")
        for vehicle_type, count in status.detections.items():
            print(f"  {vehicle_type}: {count}")
        print(f"Processing Time: {status.processing_time:.3f} seconds")
        print("-" * 50)

    priority_lane, reason = determine_priority_lane(lane_statuses)
    print('\033[1m' + "-" * 100)
    print(emoji.emojize(':vertical_traffic_light:') + 
          '\033[1m' + '\033[94m' + 
          f" Priority given to Lane {priority_lane} - Reason: {reason}" + 
          '\033[30m' + "\n")
    
    vehicle_counts = [status.vehicle_count for status in lane_statuses]
    switching_time = avg_signal_oc_time(vehicle_counts)
    if lane_statuses[priority_lane-1].has_ambulance:
        switching_time = min(switching_time, 30)

    switch_signal(priority_lane, switching_time)
    print('\033[1m' + "-" * 100)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
