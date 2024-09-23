from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt')  # You can also use 'yolov8n.yaml' to train from scratch

    # Train the model
    model.train(
        data='dataset.yaml',  # Make sure this points to your updated dataset.yaml
        epochs=50,
        imgsz=1024,
        batch=16,
        device=0,  # Use GPU (set to 'cpu' if you don't have a GPU)
        amp=True  # Automatic Mixed Precision
    )

    # Load the best model after training
    model = YOLO('runs/detect/train/weights/best.pt')

    # Evaluate on the test set
    metrics = model.val(
        data='dataset.yaml',
        split='test'  # Specify the test split
    )

    # Print metrics
    print(metrics)
