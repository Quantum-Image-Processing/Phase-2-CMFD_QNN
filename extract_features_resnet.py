"""Extract per-image feature vectors for MICC-F2000, using the trained
ResNet18 patch extractor (models/patch_extractor_best.pth) specifically.

STRIDE HERE IS 128, NOT 8 -- confirmed from the paper (page 7):
    "The patches that extract the image features are obtained using a
    stride s of 128 for MICC-F600. The proposed model selected a larger
    stride to maintain the same number of patches per image."

This is DIFFERENT from the stride used to TRAIN the CNN's weights
(stride 8, in train_extractor_patch_classifier.py, page 5's "CNN Training"
section). Two different stages of the pipeline, two different strides,
both confirmed directly from the paper text.

Output: features/micc_features_resnet.npz (32-dim, mean- and max-pooled,
saved separately from the Al-khuzaie-native version's features).

Label convention CONFIRMED against actual MICC-F2000 filenames:
    'tamp' in filename  -> forged     (700 images)
    '_scale' in filename -> authentic (1300 images)
"""
import os, sys
import numpy as np
import torch
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)

from train_extractor_patch_classifier import ResNetExtractor, MEAN, STD

FULL_IMG_SIZE     = 256
PATCH_SIZE        = 128
EXTRACTION_STRIDE = 128    # confirmed paper value for feature extraction (NOT training stride 8)
MICC_DIR          = 'data/MICC-F2000'
EXTRACTOR_PATH    = 'models/patch_extractor_best.pth'
OUT_PATH          = 'features/micc_features_resnet.npz'
FEATURE_DIM       = 32


def get_label(filename: str) -> int:
    fname = filename.lower()
    if 'tamp' in fname:
        return 1
    if '_scale' in fname:
        return 0
    raise ValueError(
        f"Filename '{filename}' matches neither 'tamp' nor '_scale' -- "
        f"check this file manually before proceeding.")


def extract_patches(img, coords):
    return [img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE)) for y in coords for x in coords]


def main():
    os.makedirs('features', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Extractor: ResNet18  ({FEATURE_DIM}-dim)", flush=True)

    extractor = ResNetExtractor(pretrained=False).to(device)
    ckpt = torch.load(EXTRACTOR_PATH, map_location=device)
    extractor.load_state_dict(ckpt['extractor_state'])
    extractor.eval()
    print(f"Loaded extractor from epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})", flush=True)

    mean_t = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1)
    coords = list(range(0, FULL_IMG_SIZE - PATCH_SIZE + 1, EXTRACTION_STRIDE))
    n_patches_per_img = len(coords) ** 2
    print(f"Extraction stride {EXTRACTION_STRIDE} -> {n_patches_per_img} patches/image "
          f"(paper-confirmed value, different from the training stride of 8)", flush=True)

    filenames = sorted(f for f in os.listdir(MICC_DIR)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')))
    print(f"Found {len(filenames)} images in {MICC_DIR}", flush=True)

    mean_feats, max_feats, labels, ids, skipped = [], [], [], [], []
    with torch.no_grad():
        for i, fname in enumerate(filenames):
            try:
                label = get_label(fname)
            except ValueError as e:
                print(f"  SKIPPING {fname}: {e}", flush=True)
                skipped.append(fname)
                continue

            img = Image.open(os.path.join(MICC_DIR, fname)).convert('RGB')
            img = img.resize((FULL_IMG_SIZE, FULL_IMG_SIZE), Image.BILINEAR)
            patches = extract_patches(img, coords)

            batch = torch.stack([
                (torch.tensor(np.array(p, dtype=np.float32)).permute(2, 0, 1) / 255.0 - mean_t) / std_t
                for p in patches
            ]).to(device)

            feat_maps = extractor(batch)                    # (n_patches, 32, H, W)
            per_patch_vec = feat_maps.mean(dim=(2, 3))         # GAP per patch -> (n_patches, 32)

            mean_feats.append(per_patch_vec.mean(dim=0).cpu().numpy())
            max_feats.append(per_patch_vec.amax(dim=0).cpu().numpy())
            labels.append(label)
            ids.append(fname)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(filenames)} processed", flush=True)

    if skipped:
        print(f"\nWARNING: {len(skipped)} files skipped: {skipped[:10]}", flush=True)

    mean_feats = np.stack(mean_feats)
    max_feats  = np.stack(max_feats)
    labels     = np.array(labels)
    n_forged = int(labels.sum())
    print(f"\nTotal used: {len(labels)}  |  forged: {n_forged}  |  authentic: {len(labels)-n_forged}",
          flush=True)
    print(f"(Expected ~700 forged, ~1300 authentic -- investigate if these differ meaningfully.)")

    np.savez(OUT_PATH, features_mean=mean_feats, features_max=max_feats,
              labels=labels, ids=np.array(ids), feature_dim=FEATURE_DIM)
    print(f"Saved BOTH mean- and max-pooled {FEATURE_DIM}-dim features to {OUT_PATH}", flush=True)


if __name__ == '__main__':
    main()