import os
import random
import shutil

# Set the path to your dataset folder containing images and CSV files
dataset_dir = 'C:\\Users\\43lio\\PyCharmProjects\\TargeTracker\\dataset'

# Get all image files (assuming .jpg or .png)
all_images = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png'))]

# Shuffle the images
random.seed(42)
random.shuffle(all_images)

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Compute split indices
total_images = len(all_images)
train_end_idx = int(train_ratio * total_images)
val_end_idx = train_end_idx + int(val_ratio * total_images)

# Split the images
train_images = all_images[:train_end_idx]
val_images = all_images[train_end_idx:val_end_idx]
test_images = all_images[val_end_idx:]


# Function to move images and CSV files
def move_files(image_list, split):
    images_dest = f'datasets/images/{split}/'
    labels_dest = f'datasets/labels/{split}/'
    for image_name in image_list:
        # Copy image
        shutil.copy(os.path.join(dataset_dir, image_name), images_dest)

        # Corresponding CSV annotation file
        csv_name = os.path.splitext(image_name)[0] + '.csv'
        csv_path = os.path.join(dataset_dir, csv_name)

        if os.path.exists(csv_path):
            shutil.copy(csv_path, labels_dest)
        else:
            print(f"No CSV file found for {image_name}")


# Move files to respective directories
move_files(train_images, 'train')
move_files(val_images, 'val')
move_files(test_images, 'test')
