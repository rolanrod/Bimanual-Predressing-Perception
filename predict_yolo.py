#!/usr/bin/env python3
"""
Run inference with trained YOLOv8 model on new images.
"""

from ultralytics import YOLO
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference on shirt images')
    parser.add_argument('--model', type=str, default='runs/detect/shirt_detection/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, default='img/',
                        help='Path to image(s) or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                        help='Save results with bounding boxes drawn')
    parser.add_argument('--show', action='store_true',
                        help='Show results on screen')

    args = parser.parse_args()

    # Load trained model
    print(f"Loading model from: {args.model}")
    model = YOLO(args.model)

    # Run inference
    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=args.show,
        project='runs/detect',
        name='predict',
    )

    # Print detection results
    print("\n" + "="*50)
    print("Detection Results:")
    print("="*50)

    for i, result in enumerate(results):
        img_path = result.path
        num_detections = len(result.boxes)
        print(f"\nImage {i+1}: {Path(img_path).name}")
        print(f"  Detections: {num_detections}")

        if num_detections > 0:
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                class_name = result.names[class_id]
                print(f"    - {class_name}: {conf:.2f}")

    if args.save:
        print(f"\nResults saved to: runs/detect/predict/")

if __name__ == "__main__":
    main()
