import os
import glob
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

def process_single(item):
    mc_path, mm_path, donor_dir, probe_dir = item
    base = os.path.basename(mc_path)
    stem = base[len('defacto_'):-len('_mc.png')]

    donor_path = os.path.join(donor_dir, f"{stem}.tif")
    if not os.path.exists(donor_path):
        donor_path = os.path.join(donor_dir, f"{stem}.jpg")

    probe_path = os.path.join(probe_dir, f"{stem}.jpg")
    if not os.path.exists(probe_path):
        probe_path = os.path.join(probe_dir, f"{stem}.tif")

    mc_mask = np.zeros((256, 256), dtype=np.uint8)

    if os.path.exists(donor_path):
        donor_img = cv2.imread(donor_path, cv2.IMREAD_GRAYSCALE)
        if donor_img is not None:
            donor_r = cv2.resize(donor_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            mc_mask[donor_r > 127] = 127

    if os.path.exists(probe_path):
        probe_img = cv2.imread(probe_path, cv2.IMREAD_GRAYSCALE)
        if probe_img is not None:
            probe_r = cv2.resize(probe_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            mc_mask[probe_r > 127] = 255

    mm_mask = np.zeros((256, 256), dtype=np.uint8)
    mm_mask[mc_mask > 0] = 255

    cv2.imwrite(mc_path, mc_mask)
    cv2.imwrite(mm_path, mm_mask)
    return stem

def main():
    base_dir = r"C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset"
    defacto_dir = r"C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\data\DEFACTO"
    donor_dir = os.path.join(defacto_dir, "copymove_annotations", "donor_mask")
    probe_dir = os.path.join(defacto_dir, "copymove_annotations", "probe_mask")

    tasks = []
    for split in ['train', 'val', 'test']:
        mc_files = glob.glob(os.path.join(base_dir, split, 'masks_3class', 'defacto_*_mc.png'))
        for mc_p in mc_files:
            mm_p = mc_p.replace('masks_3class', 'masks_binary').replace('_mc.png', '_mm.png')
            tasks.append((mc_p, mm_p, donor_dir, probe_dir))

    print(f"Total DEFACTO masks to regenerate: {len(tasks)}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single, tasks))
    t1 = time.time()
    print(f"Regenerated {len(results)} DEFACTO masks in {t1 - t0:.2f}s!")

if __name__ == "__main__":
    main()
