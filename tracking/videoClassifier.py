"""
Airplane Detection Project - Video Classifier Script
Author: Donovan Cullen
Description: This script processes video input for airplane classification. It uses a pre-trained YOLO
model to classify and detect airplanes in real-time or pre-recorded video footage, with options for tracking and logging results.
"""

import os
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
from collections import deque
import logging

# Configure logging for better monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define colors
COLOR_YELLOW = (0, 255, 255)  # BGR for Yellow
COLOR_GREEN = (0, 255, 0)     # BGR for Green
COLOR_RED = (0, 0, 255)       # BGR for Red (for 'Identifying...')
COLOR_WHITE = (255, 255, 255) # BGR for White (text color)

# Set device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load the YOLOv8 detection model
detection_model_path = '../runs/detect/train4/weights/best.pt'  # Update with your YOLOv8 model path
if not os.path.exists(detection_model_path):
    logging.error(f"YOLOv8 model not found at {detection_model_path}")
    exit(1)
detection_model = YOLO(detection_model_path)

# Set detection model parameters
detection_model.overrides['tracker'] = 'bytetrack.yaml'  # Use ByteTrack
detection_model.overrides['conf'] = 0.25  # Confidence threshold
detection_model.overrides['iou'] = 0.45   # IOU threshold
detection_model.overrides['imgsz'] = 640  # Input image size

# Load class names for classification
class_names = [
    'A10', 'A400M', 'AG600', 'AH64', 'An72', 'AV8B', 'B1', 'B2', 'B21', 'B52',
    'Be200', 'C130', 'C17', 'C2', 'C390', 'CH47', 'C5', 'E2', 'E7', 'EF2000',
    'F117', 'F14', 'F15', 'F16', 'FA18', 'F22', 'F35', 'F4', 'H6', 'J10', 'J20',
    'JAS39', 'JF17', 'JH7', 'KC135', 'KF21', 'KJ600', 'Ka52', 'MQ9', 'Mi24',
    'Mi28', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25',
    'Su34', 'Su57', 'TB001', 'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2',
    'UH60', 'US2', 'V22', 'Vulcan', 'XB70', 'Y20', 'YF23', 'F18'
]
num_classes = len(class_names)
logging.info(f"Number of classes for classification: {num_classes}")

# Load the trained classification model
classification_model_path = 'mobilenetv2_aircraft_classifier.pth'
if not os.path.exists(classification_model_path):
    logging.error(f"Classification model not found at {classification_model_path}")
    exit(1)
classification_model = models.mobilenet_v2()
classification_model.classifier[1] = nn.Linear(classification_model.last_channel, num_classes)
classification_model.load_state_dict(torch.load(classification_model_path, map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()
logging.info("Classification model loaded and set to evaluation mode.")

# Define the transformation for input images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Prediction function for classification
def predict_image_with_confidence(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classification_model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        predicted_class = class_names[preds[0]]
        confidence = confidence.item()
    return predicted_class, confidence

# Path to the input video file
video_path = '../militaryplanecomp.mp4'
if not os.path.exists(video_path):
    logging.error(f"Video file not found at {video_path}")
    exit(1)

# Initialize video capture
cap = cv2.VideoCapture(video_path)
logging.info(f"Opened video file: {video_path}")

# Set up video writer to save the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = '../output_video.mp4'  # Output video file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
logging.info(f"Output video will be saved to: {output_path}")

# Initialize object classification and position histories
object_class_history = {}
object_position_history = {}
object_last_label = {}
object_new_label_counter = {}

# Initialize parameters
history_length = 25  # Number of frames for classification
confidence_threshold = 0.85 # Required detection frequency for classification
label_change_confirmation = 25 # Number of consecutive frames to confirm label change
min_confidence = 0.7 # Minimum confidence for classifications

#Scene Change Detection Setup
previous_frame = None               # To store the previous frame for comparison
scene_change_threshold = 0.5        # Histogram correlation threshold to detect scene change

# Main Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.info("End of video stream reached.")
        break

    # Scene Change Detection
    if previous_frame is not None:
        try:
            # Convert frames to HSV color space
            hsv_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
            hsv_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Compute HSV histograms
            hist_prev = cv2.calcHist([hsv_prev], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [50, 60], [0, 180, 0, 256])

            # Normalize histograms
            cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)

            # Compare histograms using Correlation method
            similarity = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

            # Check if similarity is below the threshold
            if similarity < scene_change_threshold:
                # Reset tracking and classification histories
                object_class_history.clear()
                object_last_label.clear()
                object_new_label_counter.clear()
                object_position_history.clear()
                logging.info("Cleared object histories due to scene change.")
        except Exception as e:
            logging.error(f"Error during scene change detection: {e}")

    # Update previous_frame
    previous_frame = frame.copy()

    # Run Detection and Tracking
    results = detection_model.track(source=frame, persist=True, verbose=False)
    if not results:
        logging.warning("No detection results returned.")
        out.write(frame)
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    result = results[0]

    # Get detections
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        # No detections in this frame
        out.write(frame)
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    ids = boxes.id  # Object IDs assigned by the tracker

    # Get current object IDs
    current_ids = set(int(id.cpu().numpy()) for id in ids) if ids is not None else set()

    # Remove histories of objects that are no longer present
    for obj_id in list(object_class_history.keys()):
        if obj_id not in current_ids:
            del object_class_history[obj_id]
    for obj_id in list(object_position_history.keys()):
        if obj_id not in current_ids:
            del object_position_history[obj_id]
    for obj_id in list(object_last_label.keys()):
        if obj_id not in current_ids:
            del object_last_label[obj_id]
    for obj_id in list(object_new_label_counter.keys()):
        if obj_id not in current_ids:
            del object_new_label_counter[obj_id]

    # Loop Through Detections
    for i, box in enumerate(boxes):
        try:
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            # Object ID
            obj_id = int(ids[i].cpu().numpy()) if ids is not None else 0

            # Crop the detected airplane from the frame
            cropped_img = frame[y_min:y_max, x_min:x_max]

            # Check if the crop is valid
            if cropped_img.size == 0:
                logging.warning(f"Empty crop for Object ID {obj_id}. Skipping classification.")
                continue

            # Convert the cropped image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

            # Classify the cropped image
            plane_type, classification_confidence = predict_image_with_confidence(pil_image)

            # Classification Confidence Handling
            if classification_confidence >= min_confidence:
                # Update classification history
                if obj_id in object_class_history:
                    object_class_history[obj_id].append(plane_type)
                    if len(object_class_history[obj_id]) > history_length:
                        object_class_history[obj_id].popleft()
                else:
                    object_class_history[obj_id] = deque([plane_type], maxlen=history_length)

                # Determine the most frequent classification in history
                if len(object_class_history[obj_id]) == history_length:
                    # Count the occurrences of each classification
                    class_counts = {}
                    for cls in object_class_history[obj_id]:
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                    # Find the most frequent classification
                    most_common_type = max(class_counts, key=class_counts.get)
                    frequency = class_counts[most_common_type]
                    confidence_percent = (frequency / history_length) * 100  # Calculate confidence percentage

                    # Check if the frequency meets the confidence threshold
                    if frequency / history_length >= confidence_threshold:
                        if obj_id in object_last_label:
                            if most_common_type != object_last_label[obj_id]:
                                # Potential label change detected
                                if obj_id in object_new_label_counter:
                                    object_new_label_counter[obj_id] += 1
                                else:
                                    object_new_label_counter[obj_id] = 1

                                # Check if label change is confirmed
                                if object_new_label_counter[obj_id] >= label_change_confirmation:
                                    # Update the label
                                    object_last_label[obj_id] = most_common_type
                                    object_new_label_counter[obj_id] = 0
                                    label = f"{most_common_type} ({confidence_percent:.1f}%)"
                                    logging.info(f"Label for Object ID {obj_id} changed to {label}")
                                else:
                                    # Not enough confirmations yet; retain previous label
                                    if obj_id in object_last_label:
                                        existing_label = object_last_label[obj_id]
                                        # Retrieve the existing confidence if available
                                        existing_confidence = (class_counts[existing_label] / history_length) * 100
                                        label = f"{existing_label} ({existing_confidence:.1f}%)"
                                    else:
                                        label = 'Identifying...'
                            else:
                                # Same as current label
                                confidence_percent = (frequency / history_length) * 100
                                label = f"{object_last_label[obj_id]} ({confidence_percent:.1f}%)"
                                object_new_label_counter[obj_id] = 0
                        else:
                            # No previous label; set the current as the label
                            object_last_label[obj_id] = most_common_type
                            label = f"{most_common_type} ({confidence_percent:.1f}%)"
                            logging.info(f"Label for Object ID {obj_id} set to {label}")
                    else:
                        # Frequency does not meet threshold
                        if obj_id in object_last_label:
                            existing_label = object_last_label[obj_id]
                            existing_confidence = (class_counts[existing_label] / history_length) * 100
                            label = f"{existing_label} ({existing_confidence:.1f}%)"  # Retain previous label with confidence
                        else:
                            label = 'Identifying...'
                else:
                    # Not enough data yet
                    if obj_id in object_last_label:
                        existing_label = object_last_label[obj_id]
                        # Calculate current confidence for the existing label
                        class_counts = {}
                        for cls in object_class_history[obj_id]:
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        existing_confidence = (class_counts.get(existing_label, 0) / len(object_class_history[obj_id])) * 100
                        label = f"{existing_label} ({existing_confidence:.1f}%)"
                    else:
                        label = 'Identifying...'
            else:
                # Low-confidence classification
                if obj_id in object_last_label:
                    existing_label = object_last_label[obj_id]
                    # Calculate current confidence for the existing label
                    class_counts = {}
                    for cls in object_class_history.get(obj_id, []):
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    existing_confidence = (class_counts.get(existing_label, 0) / len(object_class_history.get(obj_id, []))) * 100 if object_class_history.get(obj_id) else 0
                    label = f"{existing_label} ({existing_confidence:.1f}%)"
                else:
                    label = 'Identifying...'

            # Smooth Bounding Box Positions
            if obj_id in object_position_history:
                prev_coords = object_position_history[obj_id]
                alpha = 0.7  # Smoothing factor
                x_min = int(alpha * x_min + (1 - alpha) * prev_coords[0])
                y_min = int(alpha * y_min + (1 - alpha) * prev_coords[1])
                x_max = int(alpha * x_max + (1 - alpha) * prev_coords[2])
                y_max = int(alpha * y_max + (1 - alpha) * prev_coords[3])
                object_position_history[obj_id] = (x_min, y_min, x_max, y_max)
            else:
                object_position_history[obj_id] = (x_min, y_min, x_max, y_max)


            # Determine box color based on label state
            if label.startswith('Identifying'):
                box_color = COLOR_YELLOW  # Yellow for identifying
            else:
                box_color = COLOR_GREEN    # Green for confirmed

            # Draw the bounding box with the assigned color
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, thickness=2)

            # Define font parameters for smaller text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8    # Reduced font size
            font_thickness = 2  # Reduced thickness

            # Put the label text above the bounding box without background
            cv2.putText(frame, label, (x_min, y_min - 5),
                        font, font_scale, COLOR_RED, font_thickness, cv2.LINE_AA)
        except Exception as e:
            logging.error(f"Error processing detection {i}: {e}")
            continue

    # Display and Save Frame
    cv2.imshow('Result', frame)

    # Write the frame to the output file
    out.write(frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Termination by user.")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
logging.info("Released all resources")
