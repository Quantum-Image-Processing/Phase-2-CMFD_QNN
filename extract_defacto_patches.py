"""Extract VGG16 block-4 patch features from DEFACTO copy-move images.

Feature extractor: torchvision VGG16 (ImageNet pretrained), frozen,
blocks 1-4 only (features[0:24]). Input 256×256 → output 16×16×512.
Preprocessing: resize to 256×256, ImageNet normalize
  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

Patch labels (256 per image, one per 16×16 grid cell):
  probe_mask : forged/target region  (label 1)
  donor_mask : source/copied region  (label 1)
  combined   : probe OR donor → label 1, else 0

Balanced sampling: for each image, keep all forged patches and sample
an equal number of pristine patches from the same image (1:1 ratio).
Images with n_forged == 0 after 16×16 downsampling are dropped.

Usage:
  python extract_defacto_patches.py --pilot   # 10 images, no cache write
  python extract_defacto_patches.py           # full extraction + cache
"""
import os, glob, json, hashlib, argparse, random, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from PIL import Image

import torch
import torchvision.models as tvm
import torchvision.transforms as T

IMG_DIR    = 'data/DEFACTO/copymove_img/img'
PROBE_DIR  = 'data/DEFACTO/copymove_annotations/probe_mask'
DONOR_DIR  = 'data/DEFACTO/copymove_annotations/donor_mask'
CACHE_DIR  = 'cache_defacto'
N_PATCHES  = 256   # 16 × 16
FEAT_DIM   = 512
SEED       = 42
TEST_FRAC  = 0.20


# ------------------------------------------------------------------ extractor
def build_extractor():
    vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
    extractor = torch.nn.Sequential(*list(vgg.features.children())[:24])
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return extractor


_imagenet_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_patches(extractor, img_path):
    """Return (256, 512) float32 feature array."""
    img  = Image.open(img_path).convert('RGB')
    t    = _imagenet_tf(img).unsqueeze(0)
    feat = extractor(t)                             # (1, 512, 16, 16)
    feat = feat.squeeze(0).permute(1, 2, 0).numpy()
    return feat.reshape(N_PATCHES, FEAT_DIM).astype(np.float32)


def load_mask_labels(mask_path):
    """Load .tif mask → resize 16×16 nearest → threshold >0 → (256,) int32."""
    if not os.path.exists(mask_path):
        return np.zeros(N_PATCHES, dtype=np.int32)
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((16, 16), Image.NEAREST)
    arr  = np.array(mask, dtype=np.uint8)
    return (arr.ravel() > 0).astype(np.int32)


def get_labels(stem):
    probe_p = os.path.join(PROBE_DIR, stem + '.jpg')
    if not os.path.exists(probe_p):
        probe_p = os.path.join(PROBE_DIR, stem + '.tif')
    donor_p = os.path.join(DONOR_DIR, stem + '.tif')
    if not os.path.exists(donor_p):
        donor_p = os.path.join(DONOR_DIR, stem + '.jpg')
    probe = load_mask_labels(probe_p)
    donor = load_mask_labels(donor_p)
    return np.clip(probe + donor, 0, 1)


# ------------------------------------------------------------------ split
def make_split(image_ids, seed, test_frac):
    """Split by image_id, 80/20, stratified (all images are label=1 here
    since we drop label=0 images, but keep the interface general)."""
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_set  = set(ids[:n_test])
    return {iid: ('test' if iid in test_set else 'train') for iid in ids}


# ------------------------------------------------------------------ pilot
def pilot(extractor, img_paths, n=10):
    for img_path in img_paths[:n]:
        stem   = os.path.splitext(os.path.basename(img_path))[0]
        labels = get_labels(stem)
        n_forged   = int(labels.sum())
        n_pristine = N_PATCHES - n_forged

        print(f"\n{'='*60}")
        print(f"Image : {stem}")
        print(f"Patches: {n_forged} forged (probe+donor), {n_pristine} pristine")

        grid = labels.reshape(16, 16)
        print("Label grid (X=forged, .=pristine):")
        for row in grid:
            print('  ' + ''.join('X' if v else '.' for v in row))

        patches = extract_patches(extractor, img_path)
        if n_forged > 0 and n_pristine > 0:
            mean_forged   = patches[labels == 1].mean(axis=0)
            mean_pristine = patches[labels == 0].mean(axis=0)
            diff_norm     = float(np.linalg.norm(mean_forged - mean_pristine))
            rel            = diff_norm / (float(np.linalg.norm(mean_pristine)) + 1e-8)
            print(f"||mean_forged - mean_pristine||_2 = {diff_norm:.4f}  "
                  f"(relative to ||mean_pristine||: {rel:.4f})")
        elif n_forged == 0:
            print("  (all patches pristine — no forged patches to compare)")
        else:
            print("  (all patches forged — no pristine patches to compare)")


# ------------------------------------------------------------------ full extraction
def sha256_npy(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def full_extraction(extractor, img_paths):
    rng = random.Random(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)

    all_feats   = []
    all_labels  = []
    all_img_ids = []

    n_total    = len(img_paths)
    n_dropped  = 0
    n_kept     = 0
    kept_stems = []

    print(f"\nExtracting features from {n_total} images ...")
    for i, img_path in enumerate(img_paths):
        stem   = os.path.splitext(os.path.basename(img_path))[0]
        labels = get_labels(stem)
        n_forged = int(labels.sum())

        if n_forged == 0:
            n_dropped += 1
            continue

        patches = extract_patches(extractor, img_path)

        forged_idx   = np.where(labels == 1)[0]
        pristine_idx = np.where(labels == 0)[0]

        # balanced 1:1: keep all forged, sample equal pristine from same image
        n_sample = min(n_forged, len(pristine_idx))
        sampled_pristine = rng.sample(list(pristine_idx), n_sample)

        sel_idx = list(forged_idx) + sampled_pristine
        sel_lbl = [1] * len(forged_idx) + [0] * n_sample

        all_feats.append(patches[sel_idx])
        all_labels.extend(sel_lbl)
        all_img_ids.extend([stem] * len(sel_idx))

        n_kept += 1
        kept_stems.append(stem)

        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            pct = 100 * (i + 1) / n_total
            print(f"  [{i+1}/{n_total}] {pct:.1f}%  kept={n_kept}  dropped={n_dropped}", flush=True)

    print(f"\nDone. Kept {n_kept} images, dropped {n_dropped} "
          f"({100*n_dropped/n_total:.1f}%) — forged region too small for 16×16.")

    features  = np.vstack(all_feats).astype(np.float32)
    labels_np = np.array(all_labels, dtype=np.int32)
    img_ids   = np.array(all_img_ids)

    # leakage-safe split by image_id
    split = make_split(kept_stems, seed=SEED, test_frac=TEST_FRAC)

    split_arr = np.array([split[iid] for iid in img_ids])
    train_mask = split_arr == 'train'
    test_mask  = split_arr == 'test'

    # save cache
    np.save(os.path.join(CACHE_DIR, 'patches.npy'),   features)
    np.save(os.path.join(CACHE_DIR, 'labels.npy'),    labels_np)
    np.save(os.path.join(CACHE_DIR, 'image_ids.npy'), img_ids)

    manifest = dict(
        extractor       = 'vgg16_imagenet_blocks1-4',
        total_images    = n_total,
        images_dropped  = n_dropped,
        images_kept     = n_kept,
        drop_rate_pct   = round(100 * n_dropped / n_total, 2),
        total_patches   = int(len(labels_np)),
        forged_patches  = int((labels_np == 1).sum()),
        pristine_patches= int((labels_np == 0).sum()),
        train_patches   = int(train_mask.sum()),
        test_patches    = int(test_mask.sum()),
        train_forged    = int((labels_np[train_mask] == 1).sum()),
        train_pristine  = int((labels_np[train_mask] == 0).sum()),
        test_forged     = int((labels_np[test_mask] == 1).sum()),
        test_pristine   = int((labels_np[test_mask] == 0).sum()),
        seed            = SEED,
        test_frac       = TEST_FRAC,
        patches_sha256  = sha256_npy(features),
    )
    with open(os.path.join(CACHE_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n--- Cache written to {CACHE_DIR}/ ---")
    print(f"  patches.npy   : {features.shape}  float32")
    print(f"  labels.npy    : {labels_np.shape}  (forged={manifest['forged_patches']}, pristine={manifest['pristine_patches']})")
    print(f"  image_ids.npy : {img_ids.shape}")
    print(f"\n--- Split summary ---")
    print(f"  Train: {manifest['train_patches']} patches  "
          f"({manifest['train_forged']} forged / {manifest['train_pristine']} pristine)")
    print(f"  Test : {manifest['test_patches']} patches  "
          f"({manifest['test_forged']} forged / {manifest['test_pristine']} pristine)")

    return manifest


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true', help='Run 10-image pilot only')
    ap.add_argument('--n-pilot', type=int, default=10)
    args = ap.parse_args()

    print("Building VGG16 extractor (blocks 1-4, ImageNet pretrained) ...")
    extractor = build_extractor()
    print(f"Extractor ready. Params: "
          f"{sum(p.numel() for p in extractor.parameters()):,}  [all frozen]")

    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, '*.tif')))
    print(f"Found {len(img_paths)} images in {IMG_DIR}")

    if args.pilot:
        pilot(extractor, img_paths, n=args.n_pilot)
        return

    full_extraction(extractor, img_paths)


if __name__ == '__main__':
    main()
