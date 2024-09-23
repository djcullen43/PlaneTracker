from ultralytics import YOLO
if __name__ == '__main__':
    # Load a YOLOv8n model
    model = YOLO('yolov8n.pt')  # Or 'yolov8n.yaml' for training from scratch

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
