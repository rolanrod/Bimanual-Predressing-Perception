#!/usr/bin/env python3
"""
Train YOLOv8 model for shirt component detection.
Detects: hem, left sleeve, right sleeve, and neck.
"""

from ultralytics import YOLO
import torch

def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pretrained YOLOv8 model (nano version for faster training)
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
    model = YOLO('yolov8n.pt')  # Start with nano model

    # Train the model
    results = model.train(
        data='dataset.yaml',           # Path to dataset config
        epochs=100,                     # Number of training epochs
        imgsz=640,                      # Image size (640x640)
        batch=16,                       # Batch size (adjust based on GPU memory)
        device=device,                  # Device to use
        project='runs/detect',          # Save results to this directory
        name='shirt_detection',         # Experiment name
        patience=20,                    # Early stopping patience
        save=True,                      # Save checkpoints
        plots=True,                     # Save training plots

        # Data augmentation (already enabled by default, but can be customized)
        hsv_h=0.015,                   # HSV-Hue augmentation
        hsv_s=0.7,                     # HSV-Saturation augmentation
        hsv_v=0.4,                     # HSV-Value augmentation
        degrees=10,                     # Rotation augmentation (degrees)
        translate=0.1,                  # Translation augmentation
        scale=0.5,                      # Scale augmentation
        flipud=0.0,                     # Vertical flip probability
        fliplr=0.5,                     # Horizontal flip probability
        mosaic=1.0,                     # Mosaic augmentation probability
    )

    # Validate the model
    print("\n" + "="*50)
    print("Validating trained model...")
    print("="*50)
    metrics = model.val()

    # Print final metrics
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"\nBest model saved at: runs/detect/shirt_detection/weights/best.pt")
    print(f"Last model saved at: runs/detect/shirt_detection/weights/last.pt")

if __name__ == "__main__":
    main()
