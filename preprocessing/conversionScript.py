"""
Airplane Detection Project - Conversion Script
Author: Donovan Cullen
Description: This script converts annotations and data into YOLO-compatible formats, ensuring that the
images and labels can be properly used in training.
"""

import os
import pandas as pd
from tqdm import tqdm
import shutil
import random

DATASET_DIR = 'dataset'
IMAGES_DIR = 'dataset/images'
LABELS_DIR = 'dataset/labels'

# Create directories for each split
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.jpg')]

# Shuffle the dataset
random.shuffle(image_files)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Compute split indices
total_images = len(image_files)
train_end = int(total_images * train_ratio)
val_end = train_end + int(total_images * val_ratio)

# Split the dataset
train_images = image_files[:train_end]
val_images = image_files[train_end:val_end]
test_images = image_files[val_end:]

# Combine images into a dictionary for easy access
dataset_splits = {
    'train': train_images,
    'val': val_images,
    'test': test_images
}

# Class mapping
classes = [
    'A10', 'A400M', 'AG600', 'AH64', 'An72', 'AV8B', 'B1', 'B2', 'B21', 'B52',
    'Be200', 'C130', 'C17', 'C2', 'C390', 'CH47', 'C5', 'E2', 'E7', 'EF2000',
    'F117', 'F14', 'F15', 'F16', 'FA18', 'F22', 'F35', 'F4', 'H6', 'J10', 'J20',
    'JAS39', 'JF17', 'JH7', 'KC135', 'KF21', 'KJ600', 'Ka52', 'MQ9', 'Mi24',
    'Mi28', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25',
    'Su34', 'Su57', 'TB001', 'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2',
    'UH60', 'US2', 'V22', 'Vulcan', 'WZ7', 'XB70', 'Y20', 'YF23', 'F18'  # Added 'F18'
]
class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}

# Process each split
for split in splits:
    print(f'Processing {split} set...')
    for image_filename in tqdm(dataset_splits[split], desc=f'Processing {split} images'):
        image_id = os.path.splitext(image_filename)[0]
        image_path = os.path.join(DATASET_DIR, image_filename)
        csv_filename = image_id + '.csv'
        csv_path = os.path.join(DATASET_DIR, csv_filename)

        if not os.path.exists(csv_path):
            continue  # Skip if annotation does not exist

        # Read annotation CSV
        df = pd.read_csv(csv_path)

        # Clean and process annotations
        label_content = ''
        for idx, row in df.iterrows():
            # Clean class name
            class_name = row['class'].replace('-', '').replace('/', '')

            # Map class name to ID
            if class_name not in class_to_id:
                continue  # Skip if class not in mapping
            class_id = class_to_id[class_name]

            # Get image dimensions
            width = row['width']
            height = row['height']

            # Get bounding box coordinates
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Append to label content
            label_content += f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

        # If label content is empty, skip copying the image and label
        if label_content == '':
            continue

        # Copy image to the appropriate images directory
        dst_image_path = os.path.join(IMAGES_DIR, split, image_filename)
        if not os.path.exists(dst_image_path):
            shutil.copyfile(image_path, dst_image_path)

        # Write label file
        label_filename = image_id + '.txt'
        label_path = os.path.join(LABELS_DIR, split, label_filename)
        with open(label_path, 'w') as label_file:
            label_file.write(label_content)
