import os
import shutil
import random

data_dir = 'C:\\Users\\43lio\\PyCharmProjects\\TargeTracker\\crop'  # Your crop folder path
classes = os.listdir(data_dir)

splits = ['train', 'val', 'test']
split_ratios = [0.7, 0.15, 0.15]  # Adjust as needed

# Create split directories
for split in splits:
    os.makedirs(os.path.join(data_dir, split), exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(data_dir, split, cls), exist_ok=True)

for cls in classes:
    cls_dir = os.path.join(data_dir, cls)
    if os.path.isdir(cls_dir):
        images = os.listdir(cls_dir)
        random.shuffle(images)
        num_images = len(images)
        train_end = int(split_ratios[0] * num_images)
        val_end = train_end + int(split_ratios[1] * num_images)

        for i, img in enumerate(images):
            src = os.path.join(cls_dir, img)
            if i < train_end:
                dst = os.path.join(data_dir, 'train', cls, img)
            elif i < val_end:
                dst = os.path.join(data_dir, 'val', cls, img)
            else:
                dst = os.path.join(data_dir, 'test', cls, img)
            shutil.copy(src, dst)
