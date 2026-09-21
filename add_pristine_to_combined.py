import os
import shutil
import csv
import random
from PIL import Image
import numpy as np

# Paths
PRISTINE_SRC = r"C:\Users\dines\Downloads\OneDrive_1_9-1-2026\image"
COMBINED_ROOT = r"C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed for reproducibility
random.seed(42)

# Get pristine files (without 't' suffix)
all_files = [f for f in os.listdir(PRISTINE_SRC) if f.endswith('.tif')]
pristine_files = [f for f in all_files if not f.endswith('t.tif')]
print(f"Found {len(pristine_files)} pristine images")

# Shuffle and split
random.shuffle(pristine_files)
n_train = int(len(pristine_files) * TRAIN_RATIO)
n_val = int(len(pristine_files) * VAL_RATIO)
n_test = len(pristine_files) - n_train - n_val

train_files = pristine_files[:n_train]
val_files = pristine_files[n_train:n_train + n_val]
test_files = pristine_files[n_train + n_val:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

def process_split(files, split_name):
    """Process a split: copy images, create zero masks, update manifest."""
    split_root = os.path.join(COMBINED_ROOT, split_name)
    img_dir = os.path.join(split_root, 'images')
    mc_dir = os.path.join(split_root, 'masks_3class')
    mm_dir = os.path.join(split_root, 'masks_binary')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mc_dir, exist_ok=True)
    os.makedirs(mm_dir, exist_ok=True)
    
    manifest_path = os.path.join(split_root, 'manifest.csv')
    
    # Read existing manifest
    existing_rows = []
    if os.path.exists(manifest_path):
        with open(manifest_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
    
    fieldnames = ['sample_id', 'category', 'file_name', 'forged_file_name', 
                  'mc_file_name', 'mm_file_name', 'mt_file_name', 'mask_file_name',
                  'clone_type', 'rotation', 'scale', 'translation_x', 'translation_y',
                  'overlap_ratio', 'image_id']
    
    new_rows = []
    for idx, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]  # e.g., '1', '100'
        
        # Copy and convert image to jpg
        src_img = os.path.join(PRISTINE_SRC, fname)
        dst_img_name = f"pristine_{stem}.jpg"
        dst_img_path = os.path.join(img_dir, dst_img_name)
        
        img = Image.open(src_img).convert('RGB')
        img.save(dst_img_path, 'JPEG', quality=95)
        
        # Create all-zero masks (256x256)
        mc_arr = np.zeros((256, 256), dtype=np.uint8)
        mm_arr = np.zeros((256, 256), dtype=np.uint8)
        
        mc_name = f"masks_3class/pristine_{stem}_mc.png"
        mm_name = f"masks_binary/pristine_{stem}_mm.png"
        
        Image.fromarray(mc_arr).save(os.path.join(split_root, mc_name))
        Image.fromarray(mm_arr).save(os.path.join(split_root, mm_name))
        
        new_rows.append({
            'sample_id': f"PRISTINE_{stem}",
            'category': 'PRISTINE',
            'file_name': f"images/{dst_img_name}",
            'forged_file_name': f"images/{dst_img_name}",
            'mc_file_name': mc_name,
            'mm_file_name': mm_name,
            'mt_file_name': '',
            'mask_file_name': '',
            'clone_type': '',
            'rotation': '',
            'scale': '',
            'translation_x': '',
            'translation_y': '',
            'overlap_ratio': '',
            'image_id': ''
        })
    
    # Write updated manifest
    all_rows = existing_rows + new_rows
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"  {split_name}: added {len(new_rows)} pristine samples, total {len(all_rows)}")

# Process each split
process_split(train_files, 'train')
process_split(val_files, 'val')
process_split(test_files, 'test')

print("Done!")