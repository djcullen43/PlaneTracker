"""
Airplane Detection Project - Dataset Validation Script
Author: Donovan Cullen
Description: This script validates the dataset, ensuring that all required files (images and annotations)
are in place and properly formatted for YOLO training. Any issues in the dataset will be logged for correction.
"""

import os
import cv2

def check_dataset_integrity(dataset_dir):
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    subsets = ['train', 'val']

    for subset in subsets:
        image_subset_dir = os.path.join(image_dir, subset)
        label_subset_dir = os.path.join(label_dir, subset)

        if not os.path.exists(image_subset_dir) or not os.path.exists(label_subset_dir):
            print(f"[WARNING] {subset} directory is missing in images or labels.")
            continue

        image_files = [f for f in os.listdir(image_subset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f for f in os.listdir(label_subset_dir) if f.endswith('.txt')]

        # Check for matching files
        image_basenames = set(os.path.splitext(f)[0] for f in image_files)
        label_basenames = set(os.path.splitext(f)[0] for f in label_files)

        unmatched_images = image_basenames - label_basenames
        unmatched_labels = label_basenames - image_basenames

        if unmatched_images:
            print(f"[ERROR] Images without labels in {subset}:")
            for name in unmatched_images:
                print(f" - {name}")
        if unmatched_labels:
            print(f"[ERROR] Labels without images in {subset}:")
            for name in unmatched_labels:
                print(f" - {name}")

        # Check each image and label
        for image_file in image_files:
            image_path = os.path.join(image_subset_dir, image_file)
            label_path = os.path.join(label_subset_dir, os.path.splitext(image_file)[0] + '.txt')

            # Check if image can be opened
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Cannot open image: {image_path}")
                continue

            height, width = img.shape[:2]

            # Check if label file exists
            if not os.path.exists(label_path):
                print(f"[ERROR] Missing label file for image: {image_file}")
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                print(f"[WARNING] Empty label file: {label_path}")
                continue

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"[ERROR] Incorrect annotation format in {label_path} at line {idx + 1}")
                    continue

                class_id, x_center, y_center, bbox_width, bbox_height = parts

                # Validate class_id
                try:
                    class_id = int(class_id)
                    if class_id < 0:
                        print(f"[ERROR] Negative class_id in {label_path} at line {idx + 1}")
                except ValueError:
                    print(f"[ERROR] Non-integer class_id in {label_path} at line {idx + 1}")

                # Validate bounding box coordinates
                try:
                    x_center = float(x_center)
                    y_center = float(y_center)
                    bbox_width = float(bbox_width)
                    bbox_height = float(bbox_height)
                    coords = [x_center, y_center, bbox_width, bbox_height]

                    if not all(0 <= coord <= 1 for coord in coords):
                        print(f"[ERROR] Bounding box out of bounds in {label_path} at line {idx + 1}")
                except ValueError:
                    print(f"[ERROR] Non-float coordinate in {label_path} at line {idx + 1}")

    print("\n[INFO] Dataset integrity check completed.")

dataset_directory = r'C:\Users\43lio\PyCharmProjects\TargeTracker\datasets'

check_dataset_integrity(dataset_directory)
