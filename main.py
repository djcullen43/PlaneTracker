import cv2
from ultralytics import YOLO

# Load YOLOv8 model (YOLOv8n is the nano model for faster inference)
model = YOLO('runs/detect/train9/weights/best.pt')  # Use yolov8n, yolov8s, yolov8m, yolov8l, or yolov8x
model.to('cuda')  # This moves the model to the GPU

# Check if the model is on the GPU
if next(model.parameters()).is_cuda:
    print("YOLOv8 is using the GPU.")
else:
    print("YOLOv8 is using the CPU.")

# Specify the list of objects you want to track
target_classes = [
    'A10', 'A400M', 'AG600', 'AH64', 'An72', 'AV8B', 'B1', 'B2', 'B21', 'B52',
    'Be200', 'C130', 'C17', 'C2', 'C390', 'CH47', 'C5', 'E2', 'E7', 'EF2000',
    'F117', 'F14', 'F15', 'F16', 'FA18', 'F22', 'F35', 'F4', 'H6', 'J10', 'J20',
    'JAS39', 'JF17', 'JH7', 'KC135', 'KF21', 'KJ600', 'Ka52', 'MQ9', 'Mi24',
    'Mi28', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25',
    'Su34', 'Su57', 'TB001', 'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2',
    'UH60', 'US2', 'V22', 'Vulcan', 'WZ7', 'XB70', 'Y20', 'YF23', 'F18'  # Added 'F18'
]

# Function to track objects in the video using YOLOv8 and display using cv2
def track_objects_cv2(model, video_path, target_classes):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Filter detections for the target classes
        detections = results[0].boxes.data.cpu().numpy()  # YOLOv8 stores results in the 'boxes' attribute
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            class_id = int(class_id)
            class_name = model.names[class_id]

            if class_name in target_classes:
                # Draw bounding boxes on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Show the processed frame in a window using OpenCV
        cv2.imshow('Processed Video', frame)

        # Break if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    print("Processing complete.")

# Path to the video file
uploaded_video = 'militaryplanecomp.mp4'

# Process the video
track_objects_cv2(model, uploaded_video, target_classes)
