"""Check the REAL distribution of patch/mask overlap before picking a
labeling threshold. Run this before training -- answers "what fraction of
a typical pasted region's patches actually have meaningful mask coverage"
using your actual synthetic dataset, instead of guessing a percentage.

Only touches synthetic_coco (forged images only -- authentic images have
no mask signal to check). Does not train anything.
"""
import os, json, random, sys
import numpy as np
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)

IMG_DIR  = 'data/synthetic_coco/images'
MASK_DIR = 'data/synthetic_coco/masks'
MANIFEST = 'data/synthetic_coco/manifest.json'

FULL_IMG_SIZE = 256
PATCH_SIZE    = 128
STRIDE        = 8
SAMPLE_IMAGES = 500   # enough for a stable histogram, cheap to run


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    samples = [s for s in manifest['samples'] if s['split'] == 'train']

    # Only images that actually have a forged region are informative here
    forged_samples = []
    for s in samples:
        mask = Image.open(os.path.join(MASK_DIR, s['id'] + '.png')).convert('L')
        mask = mask.resize((FULL_IMG_SIZE, FULL_IMG_SIZE), Image.NEAREST)
        if (np.array(mask) > 0).any():
            forged_samples.append(s)

    print(f"{len(forged_samples)} / {len(samples)} train images have any forged pixels")
    random.seed(42)
    check_set = random.sample(forged_samples, min(SAMPLE_IMAGES, len(forged_samples)))

    coords = list(range(0, FULL_IMG_SIZE - PATCH_SIZE + 1, STRIDE))
    overlaps = []   # mask_patch.mean() for every patch that has ANY overlap at all

    for s in check_set:
        mask = Image.open(os.path.join(MASK_DIR, s['id'] + '.png')).convert('L')
        mask = mask.resize((FULL_IMG_SIZE, FULL_IMG_SIZE), Image.NEAREST)
        mask_np = np.array(mask) > 0

        for y in coords:
            for x in coords:
                patch_mask = mask_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                cov = patch_mask.mean()
                if cov > 0:
                    overlaps.append(cov)

    overlaps = np.array(overlaps)
    print(f"\n{len(overlaps)} patches (across {len(check_set)} images) have ANY mask overlap")
    print(f"\nCoverage distribution (fraction of patch pixels that are masked, given >0 overlap):")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {pct:3d}th percentile: {np.percentile(overlaps, pct):.4f}")

    print(f"\nHow many patches would be labeled 'forged' at different thresholds:")
    for thresh in [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]:
        n_forged = (overlaps > thresh).sum()
        pct_of_overlapping = 100 * n_forged / len(overlaps)
        print(f"  threshold={thresh:.2f}: {n_forged}/{len(overlaps)} "
              f"({pct_of_overlapping:.1f}%) of overlapping patches pass")

    print(f"\nAlso checking overall class balance implication:")
    n_patches_per_img = len(coords) ** 2
    total_patches_checked = len(check_set) * n_patches_per_img
    for thresh in [0.0, 0.05, 0.10]:
        n_forged = (overlaps > thresh).sum()
        pct_overall = 100 * n_forged / total_patches_checked
        print(f"  threshold={thresh:.2f}: {n_forged}/{total_patches_checked} total patches "
              f"({pct_overall:.2f}%) would be labeled forged -- watch for severe imbalance")


if __name__ == '__main__':
    main()