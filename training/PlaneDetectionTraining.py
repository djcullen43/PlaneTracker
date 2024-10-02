"""
Airplane Detection Project - YOLOv8n Training Script
Author: Donovan Cullen
Description: This script trains a YOLOv8n model on the cropped airplane dataset. It specifies training parameters
such as epochs, image size, and batch size, and evaluates the trained model on a test dataset.
"""

from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('../tracking/yolov8n.pt')
    # Train the model
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=1024,
        batch=16,
        device=0,
        amp=True
    )
    # Load the best model
    model = YOLO('../runs/detect/train4/weights/best.pt')
    # Evaluate on the test set
    metrics = model.val(
        data='dataset.yaml',
        split='test'  # Specify the test split
    )
    print(metrics)
