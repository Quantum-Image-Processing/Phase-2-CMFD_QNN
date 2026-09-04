import os
import glob
import random
import csv
import cv2
import numpy as np
from PIL import Image

def generate_custom_test_dataset(
    coco_forged_dir=r"C:\ml_work\synthetic_datasets\coco_subset\images_forged",
    coco_masks_3class_dir=r"C:\ml_work\synthetic_datasets\coco_subset\masks_3class",
    coco_masks_binary_dir=r"C:\ml_work\synthetic_datasets\coco_subset\masks_binary",
    defacto_img_dir="data/DEFACTO/copymove_img/img",
    defacto_probe_dir="data/DEFACTO/copymove_annotations/probe_mask",
    defacto_donor_dir="data/DEFACTO/copymove_annotations/donor_mask",
    pristine_img_dir=r"C:\Users\dines\Downloads\OneDrive_1_9-1-2026\image",
    out_root="custom_test_dataset",
    num_coco=50,
    num_defacto=50,
    num_pristine=50,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    # Directories
    img_out_dir = os.path.join(out_root, "images")
    mc_out_dir = os.path.join(out_root, "masks_3class")
    mm_out_dir = os.path.join(out_root, "masks_binary")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mc_out_dir, exist_ok=True)
    os.makedirs(mm_out_dir, exist_ok=True)

    manifest_rows = []

    print("--- 1. Processing COCO Subset (Synthetic copy-move forgeries) ---")
    coco_img_files = sorted(glob.glob(os.path.join(coco_forged_dir, "*.jpg")))
    selected_coco = random.sample(coco_img_files, min(num_coco, len(coco_img_files)))

    for idx, img_path in enumerate(selected_coco):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # base_name format: 000000377254_forge_1
        mc_path = os.path.join(coco_masks_3class_dir, base_name + "_mc.png")
        mm_path = os.path.join(coco_masks_binary_dir, base_name + "_mm.png")

        out_img_name = f"coco_{base_name}.jpg"
        out_mc_name = f"masks_3class/coco_{base_name}_mc.png"
        out_mm_name = f"masks_binary/coco_{base_name}_mm.png"

        # Load forged image & resize to 256x256
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        img.save(os.path.join(img_out_dir, out_img_name))

        # Load 3-class mask (source=127, target=255) and binary mask
        if os.path.exists(mc_path) and os.path.exists(mm_path):
            mc_img = Image.open(mc_path).convert("L").resize((256, 256), Image.NEAREST)
            mm_img = Image.open(mm_path).convert("L").resize((256, 256), Image.NEAREST)
            mc_arr = np.array(mc_img)
            mm_arr = np.array(mm_img)
        else:
            mc_arr = np.zeros((256, 256), dtype=np.uint8)
            mm_arr = np.zeros((256, 256), dtype=np.uint8)

        cv2.imwrite(os.path.join(out_root, out_mm_name), mm_arr)
        cv2.imwrite(os.path.join(out_root, out_mc_name), mc_arr)

        manifest_rows.append({
            "sample_id": f"COCO_{base_name}",
            "category": "COCO_Subset",
            "file_name": out_img_name,
            "forged_file_name": out_img_name,
            "mc_file_name": out_mc_name,
            "mm_file_name": out_mm_name
        })

    print(f"Processed {len(selected_coco)} COCO samples.")

    print("\n--- 2. Processing DEFACTO Dataset (Combining probe & donor masks) ---")
    defacto_img_files = sorted(glob.glob(os.path.join(defacto_img_dir, "*.tif")))
    selected_defacto = random.sample(defacto_img_files, min(num_defacto, len(defacto_img_files)))

    for idx, img_path in enumerate(selected_defacto):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        probe_path = os.path.join(defacto_probe_dir, stem + ".jpg")
        if not os.path.exists(probe_path):
            probe_path = os.path.join(defacto_probe_dir, stem + ".tif")

        donor_path = os.path.join(defacto_donor_dir, stem + ".jpg")
        if not os.path.exists(donor_path):
            donor_path = os.path.join(defacto_donor_dir, stem + ".tif")

        out_img_name = f"defacto_{stem}.jpg"
        out_mc_name = f"masks_3class/defacto_{stem}_mc.png"
        out_mm_name = f"masks_binary/defacto_{stem}_mm.png"

        # Load image & resize 256x256
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        img.save(os.path.join(img_out_dir, out_img_name))

        # Build 3-class and binary mask from donor (source=127) and probe (target=255)
        mc_arr = np.zeros((256, 256), dtype=np.uint8)
        
        if os.path.exists(donor_path):
            donor_img = Image.open(donor_path).convert("L").resize((256, 256), Image.NEAREST)
            donor_arr = np.array(donor_img) > 127
            mc_arr[donor_arr] = 127  # Source region

        if os.path.exists(probe_path):
            probe_img = Image.open(probe_path).convert("L").resize((256, 256), Image.NEAREST)
            probe_arr = np.array(probe_img) > 127
            mc_arr[probe_arr] = 255  # Target region (overwrites donor if overlap)

        mm_arr = np.where(mc_arr > 0, 255, 0).astype(np.uint8)

        cv2.imwrite(os.path.join(out_root, out_mc_name), mc_arr)
        cv2.imwrite(os.path.join(out_root, out_mm_name), mm_arr)

        manifest_rows.append({
            "sample_id": f"DEFACTO_{stem}",
            "category": "DEFACTO",
            "file_name": out_img_name,
            "forged_file_name": out_img_name,
            "mc_file_name": out_mc_name,
            "mm_file_name": out_mm_name
        })

    print(f"Processed {len(selected_defacto)} DEFACTO samples.")

    print("\n--- 3. Processing Pristine/Symmetrical Images (Category 3 - False Positive Test) ---")
    # Get pristine images (without 't' suffix - these are original images with natural repetitions)
    all_pristine_files = sorted(glob.glob(os.path.join(pristine_img_dir, "*.tif")))
    pristine_files = [f for f in all_pristine_files if not os.path.basename(f).endswith('t.tif')]
    selected_pristine = random.sample(pristine_files, min(num_pristine, len(pristine_files)))

    for idx, img_path in enumerate(selected_pristine):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_img_name = f"pristine_{stem}.jpg"
        out_mc_name = f"masks_3class/pristine_{stem}_mc.png"
        out_mm_name = f"masks_binary/pristine_{stem}_mm.png"

        # Load image & resize 256x256
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        img.save(os.path.join(img_out_dir, out_img_name))

        # All-zero masks (pristine - no forgery)
        mc_arr = np.zeros((256, 256), dtype=np.uint8)
        mm_arr = np.zeros((256, 256), dtype=np.uint8)

        cv2.imwrite(os.path.join(out_root, out_mc_name), mc_arr)
        cv2.imwrite(os.path.join(out_root, out_mm_name), mm_arr)

        manifest_rows.append({
            "sample_id": f"PRISTINE_{stem}",
            "category": "PRISTINE_Symmetrical",
            "file_name": out_img_name,
            "forged_file_name": out_img_name,
            "mc_file_name": out_mc_name,
            "mm_file_name": out_mm_name
        })

    print(f"Processed {len(selected_pristine)} Pristine/Symmetrical samples.")

    # Write Manifest CSV
    csv_path = os.path.join(out_root, "manifest_test_dataset.csv")
    fieldnames = ["sample_id", "category", "file_name", "forged_file_name", "mc_file_name", "mm_file_name"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nCustom Test Dataset manifest written to: {csv_path}")
    print(f"\nTotal samples: {len(manifest_rows)}")
    print(f"  COCO_Subset: {num_coco}")
    print(f"  DEFACTO: {num_defacto}")
    print(f"  PRISTINE_Symmetrical: {num_pristine}")

if __name__ == "__main__":
    generate_custom_test_dataset()
