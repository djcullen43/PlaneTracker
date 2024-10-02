"""
Airplane Detection Project - Training Script
Author: Donovan Cullen
Description: Trains the YOLOv8 model on the dataset defined in the configuration files. Includes the
training loop, metrics tracking, and model saving.
"""

from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO('../tracking/yolov8n.pt')
    # Train the model
    model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # Use GPU
        amp=True
    )
    # Load the best model
    model = YOLO('runs/detect/train9/weights/best.pt')
    # Evaluate on the test set
    metrics = model.val(
        data='data.yaml',
        split='test'  # Specify the test split
    )
    # Print metrics
    print(metrics)
