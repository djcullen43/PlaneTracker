import os
from tqdm import tqdm  # For progress bar

# Path to your labels directory (adjust as needed)
labels_dir = 'datasets/datasets/labels/'

# Iterate over train, val, and test splits
splits = ['train', 'val', 'test']

for split in splits:
    split_dir = os.path.join(labels_dir, split)
    for root, dirs, files in os.walk(split_dir):
        for file in tqdm(files, desc=f'Processing {split} labels'):
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Update class IDs to 0
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = '0'  # Set class ID to 0
                        new_line = ' '.join(parts)
                        new_lines.append(new_line + '\n')

                # Write updated labels back to file
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
