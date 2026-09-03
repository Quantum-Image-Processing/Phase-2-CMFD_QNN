"""
evaluate_experiment_a.py

Batch-evaluates a trained Experiment A checkpoint (BusterNetWithAux, as saved by
train_comparison.py) against the custom multi-category test dataset Devansh pushed
(custom_test_dataset/: COCO_Subset, DEFACTO, PRISTINE_Symmetrical, COVERAGE_SGO).

Reuses train_comparison.py's BusterNetWithAux + compute_metrics, and dataset.py's
SyntheticCMFDataset, directly -- no reimplementation.

Self-heals a known gap in the committed manifest: 100 COVERAGE_SGO rows are missing
even though the image/mask files are present on disk (see plan notes) -- patched at
runtime, original manifest left untouched.

Usage (smoke test first):
    python evaluate_experiment_a.py --checkpoint outputs/comparison/best_model_expA.pth --limit-per-category 3

Usage (full run, only after the smoke test looks right):
    python evaluate_experiment_a.py --checkpoint outputs/comparison/best_model_expA.pth
"""
import os
import csv
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from dataset import SyntheticCMFDataset
from train_comparison import BusterNetWithAux, compute_metrics

# blue = pristine/background, green = source copy, red = target copy
# (same convention as make_diagram_assets.py's CMAP, for visual consistency
# with the architecture diagram)
CMAP = np.array([[47, 79, 168], [47, 168, 79], [192, 57, 43]], dtype=np.uint8)

# Categories whose ground truth is all-background (false-positive check, not a
# real forgery-localization case) -- reported differently from COCO/DEFACTO.
FALSE_POSITIVE_CATEGORIES = {"PRISTINE_Symmetrical", "COVERAGE_SGO"}


def parse_args():
    p = argparse.ArgumentParser(description="Batch-evaluate an Experiment A checkpoint on the custom test dataset")
    p.add_argument('--checkpoint', type=str, required=True, help='Path to the Experiment A checkpoint (.pth)')
    p.add_argument('--manifest', type=str, default='../custom_test_dataset/manifest_test_dataset.csv',
                    help='custom_test_dataset/ lives at the Phase-2-CMFD_QNN repo root, one level above this script')
    p.add_argument('--root', type=str, default='../custom_test_dataset')
    p.add_argument('--img-size', type=int, default=256)
    p.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--save-dir', type=str, default='outputs/exp_a_custom_test')
    p.add_argument('--samples-per-category', type=int, default=5, help='How many visualizations to save per category')
    p.add_argument('--limit-per-category', type=int, default=None,
                    help='Cap each category to this many images (for a quick smoke test before the full run)')
    return p.parse_args()


def self_heal_manifest(manifest_csv, root):
    """Return a manifest CSV path that includes COVERAGE_SGO rows even if the
    committed manifest is missing them (known gap -- see plan notes). Writes a
    patched copy alongside the original; leaves the original untouched."""
    with open(manifest_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else \
            ['sample_id', 'category', 'file_name', 'forged_file_name', 'mc_file_name', 'mm_file_name']

    have_coverage_sgo = {r['sample_id'] for r in rows if r.get('category') == 'COVERAGE_SGO'}

    images_dir = os.path.join(root, 'images')
    on_disk = sorted(glob.glob(os.path.join(images_dir, 'coverage_sgo_*.*')))

    patched = []
    for img_path in on_disk:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]  # e.g. "coverage_sgo_1"
        sample_id = f"COVERAGE_SGO_{stem.replace('coverage_sgo_', '')}"
        if sample_id in have_coverage_sgo:
            continue
        patched.append({
            'sample_id': sample_id,
            'category': 'COVERAGE_SGO',
            'file_name': fname,
            'forged_file_name': fname,
            'mc_file_name': f'masks_3class/{stem}_mc.png',
            'mm_file_name': f'masks_binary/{stem}_mm.png',
        })

    if not patched:
        print('[manifest self-heal] no missing COVERAGE_SGO rows found -- manifest already complete.')
        return manifest_csv

    print(f'[manifest self-heal] WARNING: committed manifest was missing {len(patched)} COVERAGE_SGO rows '
          f'(images/masks exist on disk but the manifest was never regenerated after they were added). '
          f'Patching a local copy -- flag this to Devansh.')

    all_rows = rows + patched
    patched_path = os.path.join(root, 'manifest_test_dataset_patched.csv')
    with open(patched_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'[manifest self-heal] wrote {patched_path} ({len(all_rows)} rows total)')
    return patched_path


def colorize_mc(mc_class_map):
    """[H,W] array of class indices {0,1,2} -> [H,W,3] RGB using CMAP."""
    return CMAP[mc_class_map]


def save_visualization(save_dir, category, sample_idx, image_tensor, gt_mc, pred_mc):
    out_dir = os.path.join(save_dir, category)
    os.makedirs(out_dir, exist_ok=True)

    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_vis = colorize_mc(gt_mc)
    pred_vis = colorize_mc(pred_mc)

    Image.fromarray(img).save(os.path.join(out_dir, f'sample_{sample_idx}_img.png'))
    Image.fromarray(gt_vis).save(os.path.join(out_dir, f'sample_{sample_idx}_gt.png'))
    Image.fromarray(pred_vis).save(os.path.join(out_dir, f'sample_{sample_idx}_pred.png'))


def main():
    args = parse_args()

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda requested but CUDA is not available on this machine.')
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    manifest_csv = self_heal_manifest(args.manifest, args.root)

    dataset = SyntheticCMFDataset(manifest_csv=manifest_csv, root=args.root, img_size=args.img_size)

    if args.limit_per_category is not None:
        per_cat_count = defaultdict(int)
        limited_rows = []
        for row in dataset.rows:
            cat = row.get('category', 'unknown')
            if per_cat_count[cat] < args.limit_per_category:
                limited_rows.append(row)
                per_cat_count[cat] += 1
        dataset.rows = limited_rows
        print(f'[smoke test] limited to {args.limit_per_category} images/category -> {len(dataset)} total')

    print(f'Evaluating {len(dataset)} images from {manifest_csv}')

    model = BusterNetWithAux(target_size=args.img_size).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint {args.checkpoint} (epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss')})")

    os.makedirs(args.save_dir, exist_ok=True)

    # per-category running sums
    sums = defaultdict(lambda: {'pixel_acc': 0.0, 'mean_iou': 0.0, 'recall': [0.0, 0.0, 0.0], 'n': 0})
    saved_per_category = defaultdict(int)

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            category = sample['meta']['manifest_row'].get('category', 'unknown')

            image = sample['image'].unsqueeze(0).to(device)          # [1,3,H,W]
            mc = sample['mc'].to(device)                              # [1,H,W] (channel dim doubles as batch=1)
            loss_mask = sample['loss_mask'].to(device)                # [1,H,W]

            outputs = model(image)
            pixel_acc, mean_iou, recalls = compute_metrics(outputs['mc_logits'], mc, loss_mask)

            bucket = sums[category]
            bucket['pixel_acc'] += pixel_acc
            bucket['mean_iou'] += mean_iou if not np.isnan(mean_iou) else 0.0
            for i in range(3):
                bucket['recall'][i] += recalls[i] if not np.isnan(recalls[i]) else 0.0
            bucket['n'] += 1

            overall = sums['__overall__']
            overall['pixel_acc'] += pixel_acc
            overall['mean_iou'] += mean_iou if not np.isnan(mean_iou) else 0.0
            for i in range(3):
                overall['recall'][i] += recalls[i] if not np.isnan(recalls[i]) else 0.0
            overall['n'] += 1

            if saved_per_category[category] < args.samples_per_category:
                pred_class = torch.argmax(outputs['mc_pred'][0], dim=0).cpu().numpy().astype(np.uint8)
                gt_class = mc[0].cpu().numpy().astype(np.uint8)
                save_visualization(args.save_dir, category, saved_per_category[category],
                                    sample['image'], gt_class, pred_class)
                saved_per_category[category] += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(dataset):
                print(f'  [{idx + 1}/{len(dataset)}] ...')

    # ---- report ----
    print('\n' + '=' * 100)
    print(f"{'Category':<24}{'N':>5}{'PixelAcc%':>12}{'mIoU%':>10}{'Recall[BG,SRC,TGT]':>32}{'FalsePos%':>12}")
    print('-' * 100)

    summary = {}
    categories = sorted(k for k in sums.keys() if k != '__overall__')
    for cat in categories + ['__overall__']:
        b = sums[cat]
        n = max(b['n'], 1)
        pixel_acc = b['pixel_acc'] / n
        mean_iou = b['mean_iou'] / n
        recall = [r / n for r in b['recall']]
        label = 'OVERALL' if cat == '__overall__' else cat
        fp_str = ''
        fp_rate = None
        if cat in FALSE_POSITIVE_CATEGORIES:
            fp_rate = 100.0 - pixel_acc
            fp_str = f'{fp_rate:.2f}'
        recall_str = f"[{recall[0]:.1f}, {recall[1]:.1f}, {recall[2]:.1f}]"
        print(f"{label:<24}{b['n']:>5}{pixel_acc:>12.2f}{mean_iou:>10.2f}{recall_str:>32}{fp_str:>12}")

        summary[label] = {
            'n': b['n'],
            'pixel_acc': pixel_acc,
            'mean_iou': mean_iou,
            'recall_bg_src_tgt': recall,
            'false_positive_pixel_rate': fp_rate,
        }
    print('=' * 100)

    summary_path = os.path.join(args.save_dir, 'metrics_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {summary_path}')
    print(f'Sample visualizations saved under {args.save_dir}/<category>/')


if __name__ == '__main__':
    main()
