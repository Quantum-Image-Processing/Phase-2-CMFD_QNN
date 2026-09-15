import os
import shutil
import csv
import glob
import random
import cv2
import numpy as np
from PIL import Image

def setup_dirs(base_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'masks_3class'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'masks_binary'), exist_ok=True)

def copy_coco_subset(coco_dir, base_out_dir, ratios=(0.8, 0.1, 0.1)):
    print("Processing COCO subset...")
    manifest_path = os.path.join(coco_dir, "forgery_manifest_with_masks.csv")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    
    splits = {
        'train': rows[:n_train],
        'val': rows[n_train:n_train+n_val],
        'test': rows[n_train+n_val:]
    }
    
    out_rows = {'train': [], 'val': [], 'test': []}
    
    for split, split_rows in splits.items():
        for row in split_rows:
            def get_abs(path_str):
                if os.path.exists(os.path.join(coco_dir, path_str)):
                    return os.path.join(coco_dir, path_str)
                base = os.path.basename(path_str)
                for d in ['images_forged', 'images', 'masks_3class', 'masks_binary']:
                    p = os.path.join(coco_dir, d, base)
                    if os.path.exists(p): return p
                return None
            
            img_path = None
            for col in ['forged_file_name', 'file_name', 'image']:
                if row.get(col):
                    p = get_abs(row[col])
                    if p:
                        img_path = p
                        break
            
            mc_path = get_abs(row.get('mc_file_name', ''))
            mm_path = get_abs(row.get('mm_file_name', ''))
            
            if not (img_path and mc_path and mm_path):
                continue
                
            base_name = f"coco_{os.path.basename(img_path)}"
            base_mc = f"coco_{os.path.basename(mc_path)}"
            base_mm = f"coco_{os.path.basename(mm_path)}"
            
            out_img = os.path.join(base_out_dir, split, 'images', base_name)
            out_mc = os.path.join(base_out_dir, split, 'masks_3class', base_mc)
            out_mm = os.path.join(base_out_dir, split, 'masks_binary', base_mm)
            
            # Use PIL to resize COCO to 256x256 consistently for simplicity in loading
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img.save(out_img)
            
            mc_img = Image.open(mc_path).convert('L').resize((256, 256), Image.NEAREST)
            mc_img.save(out_mc)
            
            mm_img = Image.open(mm_path).convert('L').resize((256, 256), Image.NEAREST)
            mm_img.save(out_mm)
            
            new_row = row.copy()
            new_row['file_name'] = f"images/{base_name}"
            new_row['forged_file_name'] = f"images/{base_name}"
            new_row['mc_file_name'] = f"masks_3class/{base_mc}"
            new_row['mm_file_name'] = f"masks_binary/{base_mm}"
            out_rows[split].append(new_row)
            
    return out_rows

def process_defacto(defacto_dir, base_out_dir, ratios=(0.8, 0.1, 0.1)):
    print("Processing DEFACTO subset...")
    img_dir = os.path.join(defacto_dir, "copymove_img", "img")
    donor_dir = os.path.join(defacto_dir, "copymove_annotations", "donor_mask")
    probe_dir = os.path.join(defacto_dir, "copymove_annotations", "probe_mask")
    
    all_imgs = glob.glob(os.path.join(img_dir, "*.tif"))
    random.shuffle(all_imgs)
    
    n = len(all_imgs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    
    splits = {
        'train': all_imgs[:n_train],
        'val': all_imgs[n_train:n_train+n_val],
        'test': all_imgs[n_train+n_val:]
    }
    
    out_rows = {'train': [], 'val': [], 'test': []}
    
    for split, split_imgs in splits.items():
        for img_path in split_imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            donor_path = os.path.join(donor_dir, f"{stem}.jpg")
            probe_path = os.path.join(probe_dir, f"{stem}.jpg")
            
            # Load image and resize to 256x256
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            base_name = f"defacto_{stem}.png"
            base_mc = f"defacto_{stem}_mc.png"
            base_mm = f"defacto_{stem}_mm.png"
            
            img.save(os.path.join(base_out_dir, split, 'images', base_name))
            
            mc_mask = np.zeros((256, 256), dtype=np.uint8)
            mm_mask = np.zeros((256, 256), dtype=np.uint8)
            
            if os.path.exists(donor_path):
                donor_img = cv2.imread(donor_path, cv2.IMREAD_GRAYSCALE)
                donor_img = cv2.resize(donor_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                mc_mask[donor_img > 127] = 127 # 1 (donor)
                
            if os.path.exists(probe_path):
                probe_img = cv2.imread(probe_path, cv2.IMREAD_GRAYSCALE)
                probe_img = cv2.resize(probe_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                mc_mask[probe_img > 127] = 255 # 2 (probe)
                mm_mask[probe_img > 127] = 255 # binary mask
                
            cv2.imwrite(os.path.join(base_out_dir, split, 'masks_3class', base_mc), mc_mask)
            cv2.imwrite(os.path.join(base_out_dir, split, 'masks_binary', base_mm), mm_mask)
            
            out_rows[split].append({
                "sample_id": f"defacto_{stem}",
                "category": "defacto",
                "file_name": f"images/{base_name}",
                "forged_file_name": f"images/{base_name}",
                "mc_file_name": f"masks_3class/{base_mc}",
                "mm_file_name": f"masks_binary/{base_mm}"
            })
            
    return out_rows

def process_coverage(coverage_dir, base_out_dir, ratios=(0.8, 0.1, 0.1)):
    print("Processing Coverage SGO subset...")
    img_dir = os.path.join(coverage_dir, "image")
    all_tifs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    # Only original images
    original_imgs = [f for f in all_tifs if not os.path.basename(f).replace(".tif", "").endswith("t")]
    
    random.shuffle(original_imgs)
    n = len(original_imgs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    
    splits = {
        'train': original_imgs[:n_train],
        'val': original_imgs[n_train:n_train+n_val],
        'test': original_imgs[n_train+n_val:]
    }
    
    out_rows = {'train': [], 'val': [], 'test': []}
    
    for split, split_imgs in splits.items():
        for img_path in split_imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            base_name = f"coverage_sgo_{stem}.png"
            base_mc = f"coverage_sgo_{stem}_mc.png"
            base_mm = f"coverage_sgo_{stem}_mm.png"
            
            img.save(os.path.join(base_out_dir, split, 'images', base_name))
            
            mc_mask = np.zeros((256, 256), dtype=np.uint8)
            mm_mask = np.zeros((256, 256), dtype=np.uint8)
            
            cv2.imwrite(os.path.join(base_out_dir, split, 'masks_3class', base_mc), mc_mask)
            cv2.imwrite(os.path.join(base_out_dir, split, 'masks_binary', base_mm), mm_mask)
            
            out_rows[split].append({
                "sample_id": f"coverage_sgo_{stem}",
                "category": "coverage_sgo",
                "file_name": f"images/{base_name}",
                "forged_file_name": f"images/{base_name}",
                "mc_file_name": f"masks_3class/{base_mc}",
                "mm_file_name": f"masks_binary/{base_mm}"
            })
            
    return out_rows

def main():
    # Set seed for reproducibility
    random.seed(42)
    
    base_out_dir = r"C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset"
    setup_dirs(base_out_dir)
    
    coco_dir = r"C:\ml_work\synthetic_datasets\coco_subset"
    defacto_dir = r"C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\data\DEFACTO"
    coverage_dir = r"C:\Users\dines\Downloads\OneDrive_1_9-1-2026"
    
    all_rows = {'train': [], 'val': [], 'test': []}
    
    coco_rows = copy_coco_subset(coco_dir, base_out_dir)
    defacto_rows = process_defacto(defacto_dir, base_out_dir)
    coverage_rows = process_coverage(coverage_dir, base_out_dir)
    
    for split in ['train', 'val', 'test']:
        all_rows[split].extend(coco_rows[split])
        all_rows[split].extend(defacto_rows[split])
        all_rows[split].extend(coverage_rows[split])
        
        # Shuffle within each split so they are mixed
        random.shuffle(all_rows[split])
        
        # Write manifest
        manifest_path = os.path.join(base_out_dir, split, 'manifest.csv')
        
        # Get all fieldnames
        fieldnames = set()
        for r in all_rows[split]:
            fieldnames.update(r.keys())
        fieldnames = list(fieldnames)
        
        # Make sure important ones are first
        priority = ['sample_id', 'category', 'file_name', 'forged_file_name', 'mc_file_name', 'mm_file_name']
        final_fields = [f for f in priority if f in fieldnames] + [f for f in fieldnames if f not in priority]
        
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_fields)
            writer.writeheader()
            writer.writerows(all_rows[split])
            
        print(f"Wrote {len(all_rows[split])} samples to {manifest_path}")

if __name__ == "__main__":
    main()
