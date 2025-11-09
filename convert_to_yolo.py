#!/usr/bin/env python3
"""
Convert VIA format annotations to YOLO format for shirt component detection.
Splits dataset into train/val sets and creates YOLO-compatible structure.
"""

import csv
import json
import shutil
from pathlib import Path
from PIL import Image
import random

# Configuration
CSV_FILE = "via_project_3Nov2025_10h20m_csv.csv"
IMG_DIR = Path("img")
OUTPUT_DIR = Path("yolo_dataset")
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

# Class mapping
CLASS_NAMES = ["hem", "left sleeve", "right sleeve", "neck"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def parse_annotations(csv_file):
    """Parse VIA CSV annotations and group by filename."""
    annotations = {}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']

            # Skip init images (no bounding boxes provided)
            if 'init' in filename:
                continue

            # Parse region shape attributes (bounding box coordinates)
            bbox_data = json.loads(row['region_shape_attributes'])
            x = int(bbox_data['x'])
            y = int(bbox_data['y'])
            width = int(bbox_data['width'])
            height = int(bbox_data['height'])

            # Parse region attributes (class name)
            region_attrs = json.loads(row['region_attributes'])
            class_name = region_attrs['name']

            if filename not in annotations:
                annotations[filename] = []

            annotations[filename].append({
                'class': class_name,
                'bbox': (x, y, width, height)
            })

    return annotations

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bbox from (x, y, w, h) to YOLO format.
    YOLO format: (x_center, y_center, width, height) normalized to [0, 1]
    """
    x, y, w, h = bbox

    # Calculate center point
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalize width and height
    norm_width = w / img_width
    norm_height = h / img_height

    return x_center, y_center, norm_width, norm_height

def create_yolo_dataset(annotations, train_split=0.8):
    """Create YOLO dataset structure with train/val split."""

    # Get list of all image files (excluding init images)
    all_files = [f for f in annotations.keys() if 'init' not in f]

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_split)

    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Total images: {len(all_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    # Process train and val sets
    for split_name, file_list in [('train', train_files), ('val', val_files)]:
        for filename in file_list:
            # Get image dimensions
            img_path = IMG_DIR / filename
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            img = Image.open(img_path)
            img_width, img_height = img.size

            # Copy image to output directory
            output_img_path = OUTPUT_DIR / split_name / "images" / filename
            shutil.copy(img_path, output_img_path)

            # Create YOLO label file
            label_filename = filename.replace('.png', '.txt')
            label_path = OUTPUT_DIR / split_name / "labels" / label_filename

            with open(label_path, 'w') as f:
                for ann in annotations[filename]:
                    class_id = CLASS_TO_ID[ann['class']]
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        ann['bbox'], img_width, img_height
                    )

                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"\nDataset created successfully in {OUTPUT_DIR}/")
    print(f"Classes: {CLASS_NAMES}")

def main():
    print("Converting VIA annotations to YOLO format...")
    print(f"Reading annotations from {CSV_FILE}")

    # Parse annotations
    annotations = parse_annotations(CSV_FILE)
    print(f"Found {len(annotations)} annotated images")

    # Create YOLO dataset
    create_yolo_dataset(annotations, TRAIN_SPLIT)

if __name__ == "__main__":
    main()
