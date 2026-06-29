"""Generate synthetic copy-move forgery images from COCO val2017.

Following D2PRL Section V-B:
  - Source region: 80% random polygon (5-8 vertices, 10-30% area),
                   20% COCO instance mask (non-crowd, area > 2000px)
  - Affine: rotation in [-180, 180], scale in [0.5, 2.0]
  - Target: source and target centroids must map to different cells in 16×16 grid
  - Post-process: JPEG quality in [85, 95] + Gaussian noise std in [1, 3]
  - Binary mask: source AND target regions marked 255

Output:
  data/synthetic_coco/images/<id>.jpg
  data/synthetic_coco/masks/<id>.png

Usage:
  python generate_synthetic_coco.py --pilot       # 5 samples, show grids
  python generate_synthetic_coco.py --n 15000     # full generation
"""
import os, io, json, random, argparse, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.stdout.reconfigure(line_buffering=True)

COCO_IMG_DIR  = 'data/coco/val2017'
COCO_ANN_FILE = 'data/coco/annotations/instances_val2017.json'
OUT_IMG_DIR   = 'data/synthetic_coco/images'
OUT_MASK_DIR  = 'data/synthetic_coco/masks'

N_PER_IMAGE   = 3
SEED          = 42
TEST_FRAC     = 0.20

# D2PRL Section V-B parameters
POLY_FRAC     = 0.80   # fraction using polygon source
N_VERTS_RANGE = (5, 8)
AREA_FRAC_MIN = 0.10
AREA_FRAC_MAX = 0.30
ROT_RANGE     = (-180, 180)
SCALE_RANGE   = (0.5, 2.0)
JPEG_Q_RANGE  = (85, 95)
NOISE_STD_RANGE = (1, 3)
MAX_PLACE_TRIES = 700   # combined budget (no fallback needed with grid-cell check)


# ------------------------------------------------------------------ COCO load
def load_coco(ann_file):
    print("Loading COCO annotations ...", flush=True)
    with open(ann_file) as f:
        data = json.load(f)
    img_info = {img['id']: img for img in data['images']}
    ann_by_img = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        if ann['area'] <= 2000:
            continue
        iid = ann['image_id']
        ann_by_img.setdefault(iid, []).append(ann)
    print(f"  {len(img_info)} images, "
          f"{sum(len(v) for v in ann_by_img.values())} usable annotations", flush=True)
    return img_info, ann_by_img


# ------------------------------------------------------------------ mask utils
def polygon_mask(img_w, img_h, n_verts, area_frac, rng):
    """Random convex-ish polygon covering area_frac of the image."""
    target_area = area_frac * img_w * img_h
    r = np.sqrt(target_area / np.pi)
    cx = rng.randint(int(img_w * 0.2), int(img_w * 0.8))
    cy = rng.randint(int(img_h * 0.2), int(img_h * 0.8))
    angles = sorted(rng.uniform(0, 2 * np.pi, n_verts))
    radii  = rng.uniform(0.7 * r, 1.3 * r, n_verts)
    pts = []
    for a, ri in zip(angles, radii):
        x = int(np.clip(cx + ri * np.cos(a), 0, img_w - 1))
        y = int(np.clip(cy + ri * np.sin(a), 0, img_h - 1))
        pts.append((x, y))
    mask = Image.new('L', (img_w, img_h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    return np.array(mask, dtype=np.uint8)


def coco_seg_to_mask(segmentation, img_w, img_h):
    """Convert COCO polygon segmentation list to binary mask."""
    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in segmentation:
        pts = [(poly[i], poly[i+1]) for i in range(0, len(poly) - 1, 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.array(mask, dtype=np.uint8)


def mask_bbox(mask):
    """(x, y, w, h) bounding box of nonzero region."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1


def centroid(mask):
    ys, xs = np.where(mask > 0)
    return float(xs.mean()), float(ys.mean())


# ------------------------------------------------------------------ affine transform
def affine_transform(img_pil, mask_arr, angle, scale):
    """Rotate + scale image crop and its mask."""
    h, w = mask_arr.shape
    mask_pil = Image.fromarray(mask_arr)

    img_rot  = img_pil.rotate(angle, expand=True, resample=Image.BILINEAR)
    mask_rot = mask_pil.rotate(angle, expand=True, resample=Image.NEAREST)

    nw = max(1, int(img_rot.width  * scale))
    nh = max(1, int(img_rot.height * scale))
    img_scaled  = img_rot.resize((nw, nh), Image.BILINEAR)
    mask_scaled = mask_rot.resize((nw, nh), Image.NEAREST)

    return img_scaled, np.array(mask_scaled, dtype=np.uint8)


# ------------------------------------------------------------------ post-process
def post_process(img_pil, noise_std, jpeg_q):
    arr = np.array(img_pil, dtype=np.float32)
    arr += np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=jpeg_q)
    buf.seek(0)
    return Image.open(buf).copy()


# ------------------------------------------------------------------ main generator
def generate_one(img_path, img_w, img_h, annotations, rng, variant_idx):
    """Generate one synthetic copy-move image. Returns (forged_img, binary_mask) or None."""
    img = Image.open(img_path).convert('RGB')

    # --- Source region mask
    use_instance = (len(annotations) > 0) and (rng.random() > POLY_FRAC)
    if use_instance:
        ann = rng.choice(annotations)
        seg = ann.get('segmentation', [])
        if not seg or not isinstance(seg, list):
            use_instance = False
    if not use_instance:
        n_verts   = rng.randint(*N_VERTS_RANGE)
        area_frac = rng.uniform(AREA_FRAC_MIN, AREA_FRAC_MAX)
        src_mask  = polygon_mask(img_w, img_h, n_verts, area_frac, np.random.RandomState(rng.randint(0, 2**31)))
    else:
        src_mask = coco_seg_to_mask(seg, img_w, img_h)

    bbox = mask_bbox(src_mask)
    if bbox is None or bbox[2] < 8 or bbox[3] < 8:
        return None
    bx, by, bw, bh = bbox

    src_cx, src_cy = centroid(src_mask)

    # --- Affine transform
    img_crop  = img.crop((bx, by, bx + bw, by + bh))
    mask_crop = src_mask[by:by + bh, bx:bx + bw]
    angle = rng.uniform(*ROT_RANGE)
    scale = rng.uniform(*SCALE_RANGE)
    patch_img, patch_mask = affine_transform(img_crop, mask_crop, angle, scale)
    pw, ph = patch_img.size   # PIL: (w, h)

    # --- Find target location: source and target centroids in different 16×16 grid cells
    pm_ys, pm_xs = np.where(patch_mask > 127)
    if len(pm_xs) == 0:
        return None
    patch_cx_off = float(pm_xs.mean())
    patch_cy_off = float(pm_ys.mean())

    src_gc = (int(src_cx / img_w * 16), int(src_cy / img_h * 16))

    target_x = target_y = None
    for _ in range(MAX_PLACE_TRIES):
        tx = rng.randint(0, max(1, img_w - pw))
        ty = rng.randint(0, max(1, img_h - ph))
        tcx = tx + patch_cx_off
        tcy = ty + patch_cy_off
        tgt_gc = (int(tcx / img_w * 16), int(tcy / img_h * 16))
        if tgt_gc != src_gc:
            target_x, target_y = tx, ty
            break
    if target_x is None:
        return None

    # --- Paste transformed patch
    result = np.array(img, dtype=np.uint8)
    patch_arr = np.array(patch_img, dtype=np.uint8)
    tx2 = min(target_x + pw, img_w)
    ty2 = min(target_y + ph, img_h)
    aw  = tx2 - target_x
    ah  = ty2 - target_y
    if aw <= 0 or ah <= 0:
        return None
    valid_patch_mask = patch_mask[:ah, :aw] > 127
    result[target_y:ty2, target_x:tx2][valid_patch_mask] = patch_arr[:ah, :aw][valid_patch_mask]

    # --- Binary mask: source + target
    bin_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    bin_mask[src_mask > 0] = 255
    bin_mask[target_y:ty2, target_x:tx2][valid_patch_mask] = 255

    # --- Post-process
    noise_std = rng.uniform(*NOISE_STD_RANGE)
    jpeg_q    = rng.randint(*JPEG_Q_RANGE)
    forged_img = post_process(Image.fromarray(result), noise_std, jpeg_q)

    tgt_cx_final = float(target_x + patch_cx_off)
    tgt_cy_final = float(target_y + patch_cy_off)
    final_tgt_gc = (int(tgt_cx_final / img_w * 16), int(tgt_cy_final / img_h * 16))
    meta = dict(
        source_centroid=(round(src_cx, 1), round(src_cy, 1)),
        target_centroid=(round(tgt_cx_final, 1), round(tgt_cy_final, 1)),
        src_grid_cell=src_gc,
        tgt_grid_cell=final_tgt_gc,
        angle=round(angle, 1),
        scale=round(scale, 3),
        noise_std=round(noise_std, 2),
        jpeg_q=jpeg_q,
        source_type='instance' if use_instance else 'polygon',
        forged_patches=int((np.array(Image.fromarray(bin_mask).resize((16, 16), Image.NEAREST)) > 0).sum()),
    )
    return forged_img, bin_mask, meta


# ------------------------------------------------------------------ 16×16 grid
def print_grid(bin_mask, img_h, img_w, src_cx, src_cy, tgt_cx, tgt_cy):
    """Downsample mask to 16×16 and print as text grid."""
    m16 = np.array(Image.fromarray(bin_mask).resize((16, 16), Image.NEAREST)) > 0

    src_gx, src_gy = int(src_cx / img_w * 16), int(src_cy / img_h * 16)
    tgt_gx, tgt_gy = int(tgt_cx / img_w * 16), int(tgt_cy / img_h * 16)

    print("  16×16 mask grid (X=forged, S=source centroid, T=target centroid):")
    for gy in range(16):
        row = ""
        for gx in range(16):
            if gy == src_gy and gx == src_gx:
                row += "S"
            elif gy == tgt_gy and gx == tgt_gx:
                row += "T"
            elif m16[gy, gx]:
                row += "X"
            else:
                row += "."
        print(f"    {row}")


# ------------------------------------------------------------------ pilot
def run_pilot(img_info, ann_by_img, rng):
    print("\n" + "="*65)
    print("PILOT: 5 synthetic copy-move samples")
    print("="*65, flush=True)

    img_ids = [info['id'] for info in list(img_info.values())[:5]]
    generated = 0

    for img_id in img_ids:
        info  = img_info[img_id]
        fname = info['file_name']
        iw, ih = info['width'], info['height']
        anns  = ann_by_img.get(img_id, [])
        img_path = os.path.join(COCO_IMG_DIR, fname)

        result = generate_one(img_path, iw, ih, anns, rng, variant_idx=0)
        if result is None:
            print(f"  {fname}: SKIP (no valid placement)", flush=True)
            continue

        forged_img, bin_mask, meta = result
        generated += 1

        separated = (meta['src_grid_cell'] != meta['tgt_grid_cell'])
        print(f"\n  Sample {generated}: {fname}  ({iw}x{ih})")
        print(f"    source_type  : {meta['source_type']}")
        print(f"    angle={meta['angle']:+.1f}  scale={meta['scale']:.2f}")
        print(f"    source centroid: {meta['source_centroid']}  grid cell: {meta['src_grid_cell']}")
        print(f"    target centroid: {meta['target_centroid']}  grid cell: {meta['tgt_grid_cell']}")
        print(f"    grid cell constraint: {'Different OK' if separated else 'SAME -- FAIL'}")
        print(f"    forged patches (16x16): {meta['forged_patches']}/256")
        print(f"    post-process: jpeg_q={meta['jpeg_q']}  noise_std={meta['noise_std']}")

        print_grid(bin_mask, ih, iw,
                   meta['source_centroid'][0], meta['source_centroid'][1],
                   meta['target_centroid'][0], meta['target_centroid'][1])

    print(f"\nPilot done. {generated}/5 samples generated successfully.", flush=True)
    if generated == 5:
        print("=> All checks passed. Run with --n 15000 for full generation.")
    else:
        print("=> Some samples failed placement. Check image dimensions or increase MAX_PLACE_TRIES.")


# ------------------------------------------------------------------ full generation
def run_full(img_info, ann_by_img, rng, n_total):
    os.makedirs(OUT_IMG_DIR,  exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    img_ids = list(img_info.keys())
    split_rng = random.Random(SEED)
    split_rng.shuffle(img_ids)
    n_test = max(1, int(round(len(img_ids) * TEST_FRAC)))
    test_set = set(img_ids[:n_test])

    manifest_rows = []
    generated = skipped = 0
    target = n_total

    outer_idx = 0
    while generated < target:
        img_id = img_ids[outer_idx % len(img_ids)]
        outer_idx += 1
        variant = (outer_idx - 1) // len(img_ids)

        info  = img_info[img_id]
        fname = info['file_name']
        iw, ih = info['width'], info['height']
        anns  = ann_by_img.get(img_id, [])
        img_path = os.path.join(COCO_IMG_DIR, fname)

        result = generate_one(img_path, iw, ih, anns, rng, variant_idx=variant)
        if result is None:
            skipped += 1
            continue

        forged_img, bin_mask, meta = result

        sample_id = f"{img_id:012d}_v{variant}"
        forged_img.save(os.path.join(OUT_IMG_DIR,  sample_id + '.jpg'), quality=95)
        Image.fromarray(bin_mask).save(os.path.join(OUT_MASK_DIR, sample_id + '.png'))

        manifest_rows.append(dict(
            id=sample_id, coco_image_id=img_id, filename=fname,
            split='test' if img_id in test_set else 'train', **meta))
        generated += 1

        if generated % 500 == 0 or generated == target:
            print(f"  [{generated}/{target}] generated={generated}  skipped={skipped}", flush=True)

    manifest = dict(total=generated, skipped=skipped,
                    train=sum(1 for r in manifest_rows if r['split']=='train'),
                    test=sum(1 for r in manifest_rows if r['split']=='test'),
                    samples=manifest_rows)
    with open('data/synthetic_coco/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {generated} samples saved to data/synthetic_coco/", flush=True)
    print(f"  train={manifest['train']}  test={manifest['test']}", flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--n', type=int, default=15000, help='Total samples to generate')
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED)
    rng = random.Random(SEED)

    img_info, ann_by_img = load_coco(COCO_ANN_FILE)

    if args.pilot:
        run_pilot(img_info, ann_by_img, rng)
    else:
        run_full(img_info, ann_by_img, rng, n_total=args.n)


if __name__ == '__main__':
    main()
