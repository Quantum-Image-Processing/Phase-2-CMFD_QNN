"""
Add COVERAGE SGO (Similar but Genuine Objects) original images to the
custom test dataset as Category 3 -- false-positive check.

These are the ORIGINAL (un-tampered) images from the COVERAGE dataset.
They contain natural objects that look similar to each other but are NOT
forged.  Ground-truth masks are all-black (no forgery).
"""

import os
import glob
import csv
import re
import cv2
import numpy as np
from PIL import Image


def add_coverage_sgo_category(
    coverage_dir=r"C:\Users\dines\Downloads\OneDrive_1_9-1-2026",
    out_root="custom_test_dataset",
):
    img_dir = os.path.join(coverage_dir, "image")

    # Output dirs (already created by previous script)
    img_out_dir = os.path.join(out_root, "images")
    mc_out_dir = os.path.join(out_root, "masks_3class")
    mm_out_dir = os.path.join(out_root, "masks_binary")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mc_out_dir, exist_ok=True)
    os.makedirs(mm_out_dir, exist_ok=True)

    # Collect ONLY original images: N.tif  (NOT Nt.tif which are tampered)
    all_tifs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    original_imgs = [f for f in all_tifs if not os.path.basename(f).replace(".tif", "").endswith("t")]
    # Sort numerically
    original_imgs.sort(key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))

    print(f"Found {len(original_imgs)} original (SGO) images in COVERAGE dataset")

    new_rows = []
    for img_path in original_imgs:
        stem = os.path.splitext(os.path.basename(img_path))[0]  # e.g. "1", "2", ...

        out_img_name = f"coverage_sgo_{stem}.png"
        out_mc_name = f"masks_3class/coverage_sgo_{stem}_mc.png"
        out_mm_name = f"masks_binary/coverage_sgo_{stem}_mm.png"

        # Load original image & resize to 256x256
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        img.save(os.path.join(img_out_dir, out_img_name))

        # All-black masks (no forgery)
        black_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.imwrite(os.path.join(out_root, out_mc_name), black_mask)
        cv2.imwrite(os.path.join(out_root, out_mm_name), black_mask)

        new_rows.append({
            "sample_id": f"COVERAGE_SGO_{stem}",
            "category": "COVERAGE_SGO",
            "file_name": out_img_name,
            "forged_file_name": out_img_name,
            "mc_file_name": out_mc_name,
            "mm_file_name": out_mm_name,
        })

    print(f"Processed {len(new_rows)} COVERAGE SGO samples (all-black masks).")

    # ---- Append to existing manifest CSV ----
    csv_path = os.path.join(out_root, "manifest_test_dataset.csv")
    fieldnames = ["sample_id", "category", "file_name", "forged_file_name",
                   "mc_file_name", "mm_file_name"]

    # Read existing rows
    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    # Remove any previous COVERAGE_SGO rows (idempotent re-run)
    existing_rows = [r for r in existing_rows if r.get("category") != "COVERAGE_SGO"]

    all_rows = existing_rows + new_rows
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary
    cats = {}
    for r in all_rows:
        c = r.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
    print(f"\nUpdated manifest: {csv_path}")
    print(f"Total samples: {len(all_rows)}")
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    add_coverage_sgo_category()
