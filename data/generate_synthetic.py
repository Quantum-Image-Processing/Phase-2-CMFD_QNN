"""D2PRL-style synthetic copy-move forgery generator (COCO val2017).

Pipeline per sample:
  1. Load a random COCO image, resize to 1024x1024 (bilinear)
  2. Choose source region: 80% random polygon, 20% COCO instance mask
  3. Affine-transform (rotate + scale) the source region about its centroid
  4. Translate the transformed region to a random, non-overlapping location
  5. Paste it into the image (simple overwrite) and build the binary mask
  6. Post-process: JPEG compression + Gaussian noise
  7. Downscale image + mask to 256x256 and save

Output:
  data/synthetic_coco/images/{i:05d}.jpg
  data/synthetic_coco/masks/{i:05d}.png
  data/synthetic_coco/manifest.json   -- {id, split, coco_image_id, ...meta}
    split is assigned per source COCO image (not per sample) so the same base
    photo never appears in both train and test.

Usage:
  python data/generate_synthetic.py --pilot       # 5 samples + visualizations
  python data/generate_synthetic.py --n 15000     # full generation
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

REPO_ROOT     = Path(__file__).resolve().parent.parent
COCO_IMG_DIR  = REPO_ROOT / 'data/coco/val2017'
COCO_ANN_FILE = REPO_ROOT / 'data/coco/annotations/instances_val2017.json'
OUT_IMG_DIR   = REPO_ROOT / 'data/synthetic_coco/images'
OUT_MASK_DIR  = REPO_ROOT / 'data/synthetic_coco/masks'
PILOT_DIR     = REPO_ROOT / 'data/synthetic_coco/pilot'

SEED = 42
WORK_DIM  = 1024   # working resolution: all forgery ops happen here
FINAL_DIM = 256    # final saved resolution

POLY_FRAC     = 0.80     # fraction of samples using a random polygon source
N_VERTS_RANGE = (5, 10)
AREA_FRAC_MIN = 0.05
AREA_FRAC_MAX = 0.15
ROT_RANGE     = (-180, 180)
SCALE_RANGE   = (0.5, 2.0)
MIN_ANN_AREA  = 2000      # COCO instance min area, at original resolution

MIN_OFFSET_PX    = 50     # required gap between source and paste bboxes
MAX_PASTE_TRIES  = 20

JPEG_Q_RANGE    = (70, 95)
NOISE_STD_RANGE = (1, 5)

TEST_FRAC = 0.20   # fraction of *source images* (not samples) held out for test


# ------------------------------------------------------------------ setup
def clear_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def load_coco():
    print("Loading COCO annotations ...", flush=True)
    coco = COCO(str(COCO_ANN_FILE))
    img_ids = sorted(coco.getImgIds())
    print(f"  {len(img_ids)} images", flush=True)
    return coco, img_ids


# ------------------------------------------------------------------ mask utils
def random_polygon_mask(canvas_dim, rng):
    """Random convex-ish polygon, 5-10 vertices, covering ~5-15% of canvas area."""
    area_frac = rng.uniform(AREA_FRAC_MIN, AREA_FRAC_MAX)
    target_area = area_frac * canvas_dim * canvas_dim
    r = np.sqrt(target_area / np.pi)
    cx = rng.integers(int(canvas_dim * 0.2), int(canvas_dim * 0.8))
    cy = rng.integers(int(canvas_dim * 0.2), int(canvas_dim * 0.8))
    n_verts = rng.integers(N_VERTS_RANGE[0], N_VERTS_RANGE[1] + 1)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_verts))
    radii = rng.uniform(0.7 * r, 1.3 * r, n_verts)
    xs = np.clip(cx + radii * np.cos(angles), 0, canvas_dim - 1)
    ys = np.clip(cy + radii * np.sin(angles), 0, canvas_dim - 1)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    mask = np.zeros((canvas_dim, canvas_dim), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def coco_instance_mask(coco, img_id, canvas_dim, rng):
    """Pick a random non-crowd annotation with area > MIN_ANN_AREA; resize its
    mask to the working canvas. Returns None if no valid annotation exists."""
    anns = [a for a in coco.imgToAnns.get(img_id, [])
            if a.get('iscrowd', 0) == 0 and a.get('area', 0) > MIN_ANN_AREA]
    if not anns:
        return None
    ann = anns[rng.integers(0, len(anns))]
    m = coco.annToMask(ann)  # (orig_h, orig_w), values 0/1
    m = cv2.resize(m.astype(np.uint8) * 255, (canvas_dim, canvas_dim),
                    interpolation=cv2.INTER_NEAREST)
    return m


def bbox_of_mask(mask):
    """(x0, y0, x1, y1) inclusive bounding box of nonzero region, or None."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_overlaps(b1, b2, margin=0):
    x0a, y0a, x1a, y1a = b1[0] - margin, b1[1] - margin, b1[2] + margin, b1[3] + margin
    x0b, y0b, x1b, y1b = b2
    return not (x1a < x0b or x1b < x0a or y1a < y0b or y1b < y0a)


# ------------------------------------------------------------------ affine transform
def rotate_scale_about_centroid(layer, mask, angle, scale):
    """Rotate + scale a (image-layer, mask) pair about the mask's centroid,
    keeping the same WORK_DIM x WORK_DIM canvas (cv2.warpAffine, BORDER_CONSTANT)."""
    ys, xs = np.where(mask > 0)
    ccx, ccy = float(xs.mean()), float(ys.mean())
    M = cv2.getRotationMatrix2D((ccx, ccy), angle, scale)
    h, w = mask.shape[:2]
    warped_layer = cv2.warpAffine(layer, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_layer, warped_mask


def translate(layer, mask, dx, dy):
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = mask.shape[:2]
    out_layer = cv2.warpAffine(layer, T, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    out_mask = cv2.warpAffine(mask, T, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out_layer, out_mask


# ------------------------------------------------------------------ post-process
def post_process(img_bgr, rng):
    noise_std = rng.uniform(*NOISE_STD_RANGE)
    noisy = img_bgr.astype(np.float32) + rng.normal(0, noise_std, img_bgr.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    jpeg_q = int(rng.integers(JPEG_Q_RANGE[0], JPEG_Q_RANGE[1] + 1))
    ok, enc = cv2.imencode('.jpg', noisy, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec, noise_std, jpeg_q


# ------------------------------------------------------------------ main generator
def generate_one(coco, img_id, rng, py_rng):
    """Generate one synthetic copy-move sample. Returns (img_256, mask_256, meta) or None."""
    info = coco.loadImgs(img_id)[0]
    img_path = COCO_IMG_DIR / info['file_name']
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (WORK_DIM, WORK_DIM), interpolation=cv2.INTER_LINEAR)

    # --- Source region mask
    use_instance = rng.random() > POLY_FRAC
    src_mask = None
    if use_instance:
        src_mask = coco_instance_mask(coco, img_id, WORK_DIM, rng)
    if src_mask is None:
        use_instance = False
        src_mask = random_polygon_mask(WORK_DIM, rng)

    src_bbox = bbox_of_mask(src_mask)
    if src_bbox is None:
        return None

    # --- Extract source region pixels
    source_layer = np.zeros_like(img)
    source_layer[src_mask > 0] = img[src_mask > 0]

    # --- Affine transform about the mask centroid
    angle = rng.uniform(*ROT_RANGE)
    scale = rng.uniform(*SCALE_RANGE)
    warped_layer, warped_mask = rotate_scale_about_centroid(source_layer, src_mask, angle, scale)

    warped_bbox = bbox_of_mask(warped_mask)
    if warped_bbox is None:
        return None
    x0, y0, x1, y1 = warped_bbox
    bbox_w, bbox_h = x1 - x0 + 1, y1 - y0 + 1
    if bbox_w >= WORK_DIM or bbox_h >= WORK_DIM:
        return None
    bbox_cx, bbox_cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # --- Find a valid, non-overlapping paste location
    placed = False
    for _ in range(MAX_PASTE_TRIES):
        cx = py_rng.randint(bbox_w // 2, WORK_DIM - bbox_w // 2)
        cy = py_rng.randint(bbox_h // 2, WORK_DIM - bbox_h // 2)
        dx, dy = cx - bbox_cx, cy - bbox_cy
        new_bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        if not bbox_overlaps(src_bbox, new_bbox, margin=MIN_OFFSET_PX):
            placed = True
            break
    if not placed:
        return None

    dx_i, dy_i = int(round(dx)), int(round(dy))
    paste_layer, paste_mask = translate(warped_layer, warped_mask, dx_i, dy_i)
    paste_bbox = bbox_of_mask(paste_mask)
    if paste_bbox is None:
        return None

    # --- Paste (simple overwrite, no blending)
    result = img.copy()
    valid = paste_mask > 0
    result[valid] = paste_layer[valid]

    # --- Binary mask: source OR target
    bin_mask = np.zeros((WORK_DIM, WORK_DIM), dtype=np.uint8)
    bin_mask[src_mask > 0] = 255
    bin_mask[valid] = 255

    # --- Post-process
    result, noise_std, jpeg_q = post_process(result, rng)

    # --- Downscale to FINAL_DIM
    img_final = cv2.resize(result, (FINAL_DIM, FINAL_DIM), interpolation=cv2.INTER_AREA)
    mask_final = cv2.resize(bin_mask, (FINAL_DIM, FINAL_DIM), interpolation=cv2.INTER_NEAREST)
    mask_final = np.where(mask_final > 0, 255, 0).astype(np.uint8)

    coverage = float((mask_final > 0).mean())
    meta = dict(
        source_type='instance' if use_instance else 'polygon',
        angle=round(float(angle), 1), scale=round(float(scale), 3),
        jpeg_q=jpeg_q, noise_std=round(float(noise_std), 2),
        coverage=coverage,
    )
    return img_final, mask_final, meta


# ------------------------------------------------------------------ pilot
def run_pilot(coco, img_ids, rng, py_rng):
    print("\n" + "=" * 65)
    print("PILOT: 5 synthetic copy-move samples")
    print("=" * 65, flush=True)
    clear_dir(PILOT_DIR)

    generated = 0
    tries = 0
    while generated < 5 and tries < 50:
        tries += 1
        img_id = img_ids[rng.integers(0, len(img_ids))]
        result = generate_one(coco, img_id, rng, py_rng)
        if result is None:
            continue
        img_final, mask_final, meta = result
        generated += 1

        print(f"\n  Sample {generated}: image_id={img_id}")
        print(f"    source_type={meta['source_type']}  angle={meta['angle']:+.1f}  scale={meta['scale']:.2f}")
        print(f"    jpeg_q={meta['jpeg_q']}  noise_std={meta['noise_std']}")
        print(f"    mask coverage: {meta['coverage']*100:.1f}%")

        mask_bgr = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
        side_by_side = cv2.hconcat([img_final, mask_bgr])
        cv2.imwrite(str(PILOT_DIR / f'{generated:02d}_{img_id}.jpg'), side_by_side)

    print(f"\nPilot done. {generated}/5 samples generated successfully.", flush=True)
    print(f"Visualizations saved to {PILOT_DIR}")
    if generated == 5:
        print("=> Run with --n 15000 for full generation.")
    else:
        print("=> Some samples failed placement. Check MAX_PASTE_TRIES or area fractions.")


# ------------------------------------------------------------------ full run
def run_full(coco, img_ids, rng, py_rng, n_total):
    clear_dir(OUT_IMG_DIR)
    clear_dir(OUT_MASK_DIR)

    # Leakage-safe split: assign each *source* COCO image to train/test up
    # front, so every sample derived from that image stays on the same side.
    split_rng = random.Random(SEED)
    shuffled = list(img_ids)
    split_rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * TEST_FRAC)))
    test_ids = set(shuffled[:n_test])

    generated = skipped = 0
    coverages = []
    manifest_rows = []
    pbar = tqdm(total=n_total, desc="generating")
    while generated < n_total:
        img_id = img_ids[rng.integers(0, len(img_ids))]
        result = generate_one(coco, img_id, rng, py_rng)
        if result is None:
            skipped += 1
            continue

        img_final, mask_final, meta = result
        sample_id = f'{generated:05d}'
        cv2.imwrite(str(OUT_IMG_DIR / f'{sample_id}.jpg'), img_final,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT_MASK_DIR / f'{sample_id}.png'), mask_final)

        split = 'test' if img_id in test_ids else 'train'
        manifest_rows.append(dict(id=sample_id, coco_image_id=int(img_id),
                                   split=split, **meta))

        coverages.append(meta['coverage'])
        generated += 1
        pbar.update(1)
    pbar.close()

    n_train = sum(1 for r in manifest_rows if r['split'] == 'train')
    n_test_s = sum(1 for r in manifest_rows if r['split'] == 'test')
    manifest = dict(total=generated, skipped=skipped, train=n_train, test=n_test_s,
                     samples=manifest_rows)
    with open(OUT_IMG_DIR.parent / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    mean_cov = 100.0 * float(np.mean(coverages)) if coverages else 0.0
    print(f"\nDone. {generated} samples saved to {OUT_IMG_DIR.parent}", flush=True)
    print(f"  skipped={skipped}  mean mask coverage={mean_cov:.1f}%", flush=True)
    print(f"  train={n_train}  test={n_test_s}  (split by source COCO image)", flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--n', type=int, default=15000, help='Total samples to generate')
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    coco, img_ids = load_coco()

    if args.pilot:
        run_pilot(coco, img_ids, rng, py_rng)
    else:
        run_full(coco, img_ids, rng, py_rng, n_total=args.n)


if __name__ == '__main__':
    main()
