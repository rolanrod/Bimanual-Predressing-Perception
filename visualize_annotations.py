#!/usr/bin/env python3
"""
Visualize YOLO annotations to verify conversion was correct.
Draws bounding boxes on images using the converted YOLO labels.
"""

import cv2
import random
from pathlib import Path
import argparse

# Class names and colors
CLASS_NAMES = ["hem", "left sleeve", "right sleeve", "neck"]
COLORS = [
    (255, 0, 0),      # hem - red
    (0, 255, 0),      # left sleeve - green
    (0, 0, 255),      # right sleeve - blue
    (255, 255, 0),    # neck - cyan
]

def yolo_to_bbox(yolo_coords, img_width, img_height):
    """Convert YOLO format back to pixel coordinates."""
    x_center, y_center, width, height = yolo_coords

    # Convert from normalized to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Convert to top-left corner format
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2

def visualize_sample(dataset_path='yolo_dataset', split='train', num_samples=5):
    """Visualize random samples from the dataset."""

    split_path = Path(dataset_path) / split
    images_path = split_path / 'images'
    labels_path = split_path / 'labels'

    # Get all image files
    image_files = list(images_path.glob('*.png'))

    if not image_files:
        print(f"No images found in {images_path}")
        return

    # Sample random images
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    output_dir = Path('visualization_output')
    output_dir.mkdir(exist_ok=True)

    print(f"Visualizing {len(samples)} samples from {split} set...")

    for img_file in samples:
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Failed to read {img_file}")
            continue

        img_height, img_width = img.shape[:2]

        # Read corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"

        if not label_file.exists():
            print(f"Warning: No label file for {img_file.name}")
            continue

        # Parse and draw bounding boxes
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                yolo_coords = [float(x) for x in parts[1:5]]

                # Convert to pixel coordinates
                x1, y1, x2, y2 = yolo_to_bbox(yolo_coords, img_width, img_height)

                # Draw bounding box
                color = COLORS[class_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Draw label
                label = CLASS_NAMES[class_id]
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save annotated image
        output_file = output_dir / f"annotated_{img_file.name}"
        cv2.imwrite(str(output_file), img)
        print(f"  Saved: {output_file}")

    print(f"\nVisualization complete! Check {output_dir}/ for results.")

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO annotations')
    parser.add_argument('--dataset', type=str, default='yolo_dataset',
                        help='Path to YOLO dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to visualize')
    parser.add_argument('--num', type=int, default=5,
                        help='Number of samples to visualize')

    args = parser.parse_args()

    visualize_sample(args.dataset, args.split, args.num)

if __name__ == "__main__":
    main()
