"""
Airplane Detection Project - CSV to YOLO Converter
Author: Donovan Cullen
Description: This script converts annotation data from CSV format to YOLO format (text files), making
them suitable for YOLO training.
"""

import os
import pandas as pd

class_mapping = {
    'A10': 0, 'A400M': 1, 'AG600': 2, 'AH64': 3, 'An72': 4, 'AV8B': 5, 'B1': 6, 'B2': 7,
    'B21': 8, 'B52': 9, 'Be200': 10, 'C130': 11, 'C17': 12, 'C2': 13, 'C390': 14, 'CH47': 15,
    'C5': 16, 'E2': 17, 'E7': 18, 'EF2000': 19, 'F117': 20, 'F14': 21, 'F15': 22, 'F16': 23,
    'FA18': 24, 'F22': 25, 'F35': 26, 'F4': 27, 'H6': 28, 'J10': 29, 'J20': 30, 'JAS39': 31,
    'JF17': 32, 'JH7': 33, 'KC135': 34, 'KF21': 35, 'KJ600': 36, 'Ka52': 37, 'MQ9': 38, 'Mi24': 39,
    'Mi28': 40, 'Mig31': 41, 'Mirage2000': 42, 'P3': 43, 'RQ4': 44, 'Rafale': 45, 'SR71': 46, 'Su24': 47,
    'Su25': 48, 'Su34': 49, 'Su57': 50, 'TB001': 51, 'TB2': 52, 'Tornado': 53, 'Tu160': 54, 'Tu22M': 55,
    'Tu95': 56, 'U2': 57, 'UH60': 58, 'US2': 59, 'V22': 60, 'Vulcan': 61, 'WZ7': 62, 'XB70': 63, 'Y20': 64,
    'YF23': 65, 'F18': 66
}

# Function to convert bounding box to YOLO format
def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


# Function to convert CSV annotations to YOLO format
def convert_csv_to_yolo(csv_folder, yolo_labels_folder):
    if not os.path.exists(yolo_labels_folder):
        os.makedirs(yolo_labels_folder)

    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            annotations = pd.read_csv(os.path.join(csv_folder, csv_file))
            for filename, group in annotations.groupby('filename'):
                label_file_path = os.path.join(yolo_labels_folder, filename + ".txt")

                img_width = group['width'].iloc[0]
                img_height = group['height'].iloc[0]

                with open(label_file_path, 'w') as f:
                    for _, row in group.iterrows():
                        class_id = class_mapping.get(row['class'], -1)
                        if class_id == -1:
                            print(f"Warning: class {row['class']} not found in class mapping.")
                            continue
                        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                        x_center, y_center, width, height = convert_bbox_to_yolo_format(bbox, img_width, img_height)
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            os.remove(csv_file)
    print(f"[INFO] Conversion complete for folder: {csv_folder}")


# Paths to CSV and YOLO labels directories
base_dir = r'C:\Users\43lio\PyCharmProjects\TargeTracker\datasets'

# Folders for CSV annotations
csv_train_folder = os.path.join(base_dir, 'labels', 'train')
csv_val_folder = os.path.join(base_dir, 'labels', 'val')
csv_test_folder = os.path.join(base_dir, 'labels', 'test')

# Folders for YOLO annotations
yolo_train_folder = os.path.join(base_dir, 'labels', 'train')
yolo_val_folder = os.path.join(base_dir, 'labels', 'val')
yolo_test_folder = os.path.join(base_dir, 'labels', 'test')

# Convert CSV annotations to YOLO format for each set (train, val, test)
convert_csv_to_yolo(csv_train_folder, yolo_train_folder)
convert_csv_to_yolo(csv_val_folder, yolo_val_folder)
convert_csv_to_yolo(csv_test_folder, yolo_test_folder)
