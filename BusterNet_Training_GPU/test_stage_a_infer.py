# test_stage_a_infer.py
# Forward-only verification script for ManiDet and SimiDet branches.
# Requires: dataset.SyntheticCMFDataset, model_manidet.ManiDetBranch, model_simidet.SimiDetBranch
# Place in project root and run with Python.

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import project modules (assumes these files exist in project)
from dataset import SyntheticCMFDataset
from model_manidet import ManiDetBranch
from model_simidet import SimiDetBranch

def to_np(t):
    return t.detach().cpu().numpy()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_png_from_tensor_uint8(arr, path):    # arr expected HxW or HxWxC with values 0..255
    im = Image.fromarray(arr.astype(np.uint8))
    im.save(path)

def save_mask_png(mask_tensor, path):
    # mask_tensor: [H,W] values small ints 0/1/2
    arr = mask_tensor.astype(np.uint8)
    save_png_from_tensor_uint8(arr, path)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def compare_checkpoint_to_model(model, checkpoint_state):
    model_state = model.state_dict()
    common_keys = sorted(set(checkpoint_state.keys()) & set(model_state.keys()))
    total = len(common_keys)
    equal_count = 0
    diff_count = 0
    first_diffs = []

    print(f"Comparing checkpoint to model after load: {total} common keys")
    for key in common_keys:
        ckpt_tensor = checkpoint_state[key]
        model_tensor = model_state[key]
        if ckpt_tensor.shape != model_tensor.shape:
            print(f"KEY SHAPE MISMATCH: {key} ckpt={tuple(ckpt_tensor.shape)} model={tuple(model_tensor.shape)}")
            diff_count += 1
            first_diffs.append((key, ckpt_tensor, model_tensor, 'shape_mismatch'))
            continue

        are_equal = torch.equal(ckpt_tensor, model_tensor)
        max_diff = float((ckpt_tensor.detach().cpu() - model_tensor.detach().cpu()).abs().max().item())
        print(f"key={key} ckpt_shape={tuple(ckpt_tensor.shape)} model_shape={tuple(model_tensor.shape)} equal={are_equal} max_abs_diff={max_diff:.6e}")
        if are_equal:
            equal_count += 1
        else:
            diff_count += 1
            if len(first_diffs) < 5:
                first_diffs.append((key, ckpt_tensor.detach().cpu(), model_tensor.detach().cpu(), max_diff))

    print(f"Total common keys={total} equal_keys={equal_count} different_keys={diff_count}")
    if first_diffs:
        print("First differing keys:")
        for key, ckpt_tensor, model_tensor, info in first_diffs:
            if info == 'shape_mismatch':
                print(f"  {key}: shape mismatch")
                continue
            diff = (ckpt_tensor - model_tensor).ravel()
            print(f"  {key}: max_abs_diff={info:.6e}")
            print(f"    ckpt first values: {diff[:5].tolist()}")
            # compute actual first values, not diff
            print(f"    ckpt first tensor values: {ckpt_tensor.flatten()[:5].tolist()}")
            print(f"    model first tensor values: {model_tensor.flatten()[:5].tolist()}")
    return total, equal_count, diff_count


def load_weights_if_given(model, path, name):
    if path is None:
        return None
    print(f"Loading weights for {name} from {path} ...")
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format for {path}: expected dict-like state_dict")

    # If state has 'module.' prefixes strip them
    new_state = {}
    for k,v in state.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]

        # Map common community-repo branch names into our backbone module names.
        # Example:
        #   manipulation_net.features.0.weight -> backbone.0.weight
        #   similarity_net.features.0.weight -> backbone.0.weight
        #   features.0.weight -> backbone.0.weight
        if name == 'ManiDet' and new_k.startswith('manipulation_net.features.'):
            suffix = new_k[len('manipulation_net.features.'):]
            new_k = 'backbone.' + suffix
        elif name == 'SimiDet' and new_k.startswith('similarity_net.features.'):
            suffix = new_k[len('similarity_net.features.'):]
            new_k = 'backbone.' + suffix
        elif new_k.startswith('features.'):
            new_k = 'backbone.' + new_k[len('features.'):]

        new_state[new_k] = v

    res = model.load_state_dict(new_state, strict=False)
    print(f" -> load_state_dict result: missing_keys={len(res.missing_keys)}, unexpected_keys={len(res.unexpected_keys)}")
    if res.missing_keys:
        print("  missing_keys:")
        for k in res.missing_keys:
            print(f"    {k}")
    if res.unexpected_keys:
        print("  unexpected_keys:")
        for k in res.unexpected_keys:
            print(f"    {k}")

    # Compare common checkpoint keys against the loaded model state
    compare_checkpoint_to_model(model, new_state)
    return res

def backbone_param_stats(module):
    vals = []
    for n,p in module.named_parameters():
        if p is None: continue
        vals.append(p.detach().cpu().view(-1))
    if not vals:
        return None
    cat = torch.cat(vals)
    return float(cat.mean().item()), float(cat.std().item()), int(cat.numel())

def compare_backbones_exact(a_state, b_state, prefix_a=None, prefix_b=None, max_pairs=500):
    # a_state, b_state are dicts or state_dict-like
    # align keys that contain likely backbone substrings; fall back to order.
    ka = sorted([k for k in a_state.keys() if (prefix_a is None or prefix_a in k)])
    kb = sorted([k for k in b_state.keys() if (prefix_b is None or prefix_b in k)])
    if not ka or not kb:
        # fallback to all keys if backbone keys not detected
        ka = sorted(list(a_state.keys()))[:max_pairs]
        kb = sorted(list(b_state.keys()))[:max_pairs]
    pairs = min(len(ka), len(kb), max_pairs)
    equal = 0
    compared = 0
    for i in range(pairs):
        A = a_state[ka[i]]
        B = b_state[kb[i]]
        if A.shape != B.shape:
            continue
        a_np = A.detach().cpu().numpy().ravel() if isinstance(A, torch.Tensor) else np.array(A).ravel()
        b_np = B.detach().cpu().numpy().ravel() if isinstance(B, torch.Tensor) else np.array(B).ravel()
        compared += 1
        if np.array_equal(a_np, b_np):
            equal += 1
    pct = 100.0 * equal / compared if compared > 0 else 0.0
    return compared, equal, pct

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='coco_subset/forgery_manifest_with_masks.csv')
    p.add_argument('--outdir', default='outputs/py_infer')
    p.add_argument('--device', default='cpu')
    p.add_argument('--weights_manidet', default=None)
    p.add_argument('--weights_simidet', default=None)
    p.add_argument('--n-samples', type=int, default=8)
    p.add_argument('--save_inputs', action='store_true', help='Save input images (PNG)')
    p.add_argument('--save_masks', action='store_true', help='Save ground-truth masks (PNG)')
    p.add_argument('--save_preds', action='store_true', help='Save aux predictions (numpy). Default dtype float16 to reduce size when set.')
    p.add_argument('--pred_dtype', choices=['float32','float16'], default='float16', help='dtype for saved predictions')
    p.add_argument('--threshold', type=float, default=0.5, help='threshold to binarize predicted probability maps')
    p.add_argument('--save_S_P', action='store_true', help='Save S_raw and P (very large) - default OFF')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    log_path = outdir / 'run_log.jsonl'
    print("Loading dataset...")
    ds = SyntheticCMFDataset(manifest_csv=args.manifest, root='coco_subset', img_size=256)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    print("Instantiating models...")
    manidet = ManiDetBranch().to(device)
    simidet = SimiDetBranch().to(device)

    # optional weight loading
    if args.weights_manidet:
        load_weights_if_given(manidet, args.weights_manidet, 'ManiDet')
    if args.weights_simidet:
        load_weights_if_given(simidet, args.weights_simidet, 'SimiDet')

    # basic backbone stats (if accessible)
    def get_backbone_state(model):
        # try to build a state dict only for backbone parameters if exist
        sd = {}
        for n,p in model.named_parameters():
            sd[n] = p.detach().cpu()
        return sd

    sd_man = get_backbone_state(manidet)
    sd_sim = get_backbone_state(simidet)

    mean_man, std_man, cnt_man = backbone_param_stats(manidet)
    mean_sim, std_sim, cnt_sim = backbone_param_stats(simidet)
    print(f"ManiDet backbone params: count={cnt_man} mean={mean_man:.6f} std={std_man:.6f}")
    print(f"SimiDet backbone params: count={cnt_sim} mean={mean_sim:.6f} std={std_sim:.6f}")

    # compare loaded states if available by attempting to load checkpoint dicts (if any)
    # (We already loaded via load_state_if_given which printed missing/unexpected keys)

    # If user passed the same checkpoint for both, try to compare parameter equality anyway:
    try:
        # quick equality heuristic of backbone param arrays
        compared, equal, pct = compare_backbones_exact(sd_man, sd_sim)
        print(f"Backbone exact-equality check: compared_pairs={compared}, equal_pairs={equal}, pct_identical={pct:.2f}%")
        if pct > 90.0:
            print("WARNING: >90% backbone params are identical. This suggests the checkpoint may have loaded the same VGG into both branches.")
    except Exception as e:
        print("Backbone comparison failed:", e)

    # Evaluation loop
    manidet.eval()
    simidet.eval()
    saved = 0
    failures = 0
    start_time = time.time()

    with open(str(log_path), 'w') as logf:
        for idx, sample in enumerate(dl):
            if idx >= args.n_samples:
                break
            try:
                image = sample['image'].to(device)        # [1,3,256,256]
                mc = sample['mc'].to(device)             # [1,256,256] int
                mm = sample['mm'].to(device)             # [1,1,256,256] or [1,256,256]
                simi_binary = sample['simi_binary'].to(device)  # [1,1,256,256]
                loss_mask = sample['loss_mask'].to(device)      # [1,1,256,256]
                meta = sample.get('meta', {})
            except Exception as e:
                print("Failed to load sample", idx, "error:", e)
                failures += 1
                continue

            with torch.no_grad():
                out_m = manidet(image)
                out_s = simidet(image)

            # Extract tensors (expected shapes)
            f_m = out_m.get('f')   # [1,512,16,16]
            d_m = out_m.get('d')   # [1,6,256,256]
            mm_pred = out_m.get('m_pred')  # [1,1,256,256]

            f_s = out_s.get('f')   # [1,512,16,16]
            S_raw = out_s.get('S') # [1,16,16,256]
            P = out_s.get('P')     # [1,16,16,K]
            d_s = out_s.get('d')   # [1,6,256,256]
            ms_pred = out_s.get('m_pred')  # [1,1,256,256]

            # Basic shape assertions
            def assert_shape(tensor, expected, name):
                if tensor is None:
                    raise AssertionError(f"{name} is None")
                shp = tuple(tensor.shape)
                if shp != expected:
                    raise AssertionError(f"{name} shape {shp} != expected {expected}")

            try:
                assert_shape(f_m, (1,512,16,16), 'f_X_m')
                assert_shape(f_s, (1,512,16,16), 'f_X_s')
                assert_shape(S_raw, (1,16,16,256), 'S_raw')
                assert_shape(P, (1,16,16,P.shape[-1]), 'P')  # only check H/W/L dims; K may vary
                K = P.shape[-1]
                assert_shape(d_m, (1,6,256,256), 'd_X_m')
                assert_shape(d_s, (1,6,256,256), 'd_X_s')
                assert_shape(mm_pred, (1,1,256,256), 'M_m_pred')
                assert_shape(ms_pred, (1,1,256,256), 'M_s_pred')
            except AssertionError as e:
                print(f"Assertion failed on sample {idx}: {e}")
                failures += 1
                # log and continue to next sample
                logf.write(json.dumps({'sample_idx': idx, 'error': str(e)}) + '\n')
                continue

            # Numeric sanity
            def check_nans(name, tensor):
                if not torch.isfinite(tensor).all():
                    raise AssertionError(f"{name} contains NaN/Inf")

            try:
                check_nans('f_X_m', f_m)
                check_nans('f_X_s', f_s)
                check_nans('S_raw', S_raw)
                check_nans('P', P)
                check_nans('d_X_m', d_m)
                check_nans('d_X_s', d_s)
                check_nans('M_m_pred', mm_pred)
                check_nans('M_s_pred', ms_pred)
            except AssertionError as e:
                print(f"Numeric assertion failed on sample {idx}: {e}")
                failures += 1
                logf.write(json.dumps({'sample_idx': idx, 'error': str(e)}) + '\n')
                continue

            # mm should be a binary mask over the same spatial grid
            try:
                mm_np = mm.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                unique_vals = set(np.unique(mm_np).tolist())
                if unique_vals - {0, 1}:
                    raise AssertionError(f"mm contains non-binary values: {unique_vals}")
            except AssertionError as e:
                print(f"Mask alignment assertion failed on sample {idx}: {e}")
                failures += 1
                logf.write(json.dumps({'sample_idx': idx, 'error': str(e)}) + '\n')
                continue

            # Compute the saved prediction exactly once from the model output.
            pred_prob_tensor = ms_pred.clone().detach().cpu().squeeze(0).squeeze(0)
            # If the branch returns logits, apply sigmoid; otherwise this is identity for already-sigmoided output.
            pred_prob = torch.sigmoid(pred_prob_tensor) if pred_prob_tensor.dtype.is_floating_point else pred_prob_tensor
            pred_prob = pred_prob.to(torch.float32)
            pred_prob_np = pred_prob.numpy()

            gt_mask = mm.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            threshold = args.threshold
            pred_bin = (pred_prob_np >= threshold).astype(np.uint8)

            tp = int(((pred_bin == 1) & (gt_mask == 1)).sum())
            fp = int(((pred_bin == 1) & (gt_mask == 0)).sum())
            tn = int(((pred_bin == 0) & (gt_mask == 0)).sum())
            fn = int(((pred_bin == 0) & (gt_mask == 1)).sum())
            total = tp + fp + tn + fn
            gt_forged = tp + fn
            gt_pristine = tn + fp
            pred_forged = tp + fp
            pred_pristine = tn + fn

            def safe_div(a, b):
                return float(a) / float(b) if b != 0 else 0.0

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            specificity = safe_div(tn, tn + fp)
            iou = safe_div(tp, tp + fp + fn)
            dice = safe_div(2 * tp, 2 * tp + fp + fn)
            accuracy = safe_div(tp + tn, total)

            perc = lambda x: f"{100.0 * x:.2f}%"

            def tensor_stats(arr):
                return {
                    'min': float(arr.min()),
                    'max': float(arr.max()),
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'ge_0_5': int((arr >= 0.5).sum()),
                    'lt_0_5': int((arr < 0.5).sum())
                }

            before_stats = tensor_stats(pred_prob_np)

            out_base = outdir / f"sample{idx:05d}"
            ensure_dir(out_base)
            pred_save_path = str(out_base / 'simidet_m_pred.npy')
            np.save(pred_save_path, pred_prob_np)
            loaded_arr = np.load(pred_save_path)
            after_stats = tensor_stats(loaded_arr)

            match = bool(np.allclose(pred_prob_np, loaded_arr, atol=1e-6))
            verification_summary = {
                'before': before_stats,
                'after': after_stats,
                'match': match
            }

            confusion = {
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'GT_forged_pixels': gt_forged,
                'GT_pristine_pixels': gt_pristine,
                'Pred_forged_pixels': pred_forged,
                'Pred_pristine_pixels': pred_pristine,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'IoU': iou,
                'Dice': dice,
                'PixelAccuracy': accuracy,
                'Precision_pct': perc(precision),
                'Recall_pct': perc(recall),
                'Specificity_pct': perc(specificity),
                'IoU_pct': perc(iou),
                'Dice_pct': perc(dice),
                'PixelAccuracy_pct': perc(accuracy)
            }

            sample_log = {
                'sample_idx': idx,
                'meta': to_jsonable(meta),
                'shapes': {
                    'f_X_m': tuple(f_m.shape),
                    'f_X_s': tuple(f_s.shape),
                    'S_raw': tuple(S_raw.shape),
                    'P': tuple(P.shape),
                    'd_X_m': tuple(d_m.shape),
                    'd_X_s': tuple(d_s.shape),
                    'M_m_pred': tuple(mm_pred.shape),
                    'M_s_pred': tuple(ms_pred.shape)
                },
                'confusion': confusion,
                'verification': verification_summary,
                'pred_prob_summary': {
                    'min': before_stats['min'],
                    'max': before_stats['max'],
                    'mean': before_stats['mean'],
                    'std': before_stats['std'],
                    'threshold_used': threshold
                }
            }

            print('===== Verification =====')
            print('Tensor before saving')
            print(f"min={before_stats['min']}, max={before_stats['max']}, mean={before_stats['mean']}, std={before_stats['std']}, >=0.5={before_stats['ge_0_5']}, <0.5={before_stats['lt_0_5']}")
            print('Tensor after loading')
            print(f"min={after_stats['min']}, max={after_stats['max']}, mean={after_stats['mean']}, std={after_stats['std']}, >=0.5={after_stats['ge_0_5']}, <0.5={after_stats['lt_0_5']}")
            print(f"MATCH = {'YES' if match else 'NO'}")

            if not match:
                print('Verification mismatch detected for sample', idx)
                print('Stopping execution to preserve diagnostics.')
                return

            # Save according to flags
            base = outdir / f"sample{idx:05d}"
            ensure_dir(base)

            if args.save_inputs:
                inp = image.squeeze(0).permute(1,2,0).cpu().numpy()  # HWC [0,1]
                inp_img = (np.clip(inp*255.0, 0, 255)).astype(np.uint8)
                save_png_from_tensor_uint8(inp_img, str(base / 'input.png'))

            if args.save_masks:
                mc_arr = mc.squeeze().cpu().numpy().astype(np.uint8)  # [H,W] with values 0/1/2
                mm_arr = mm.squeeze().cpu().numpy().astype(np.uint8)
                save_mask_png(mc_arr * 127, str(base / 'mc.png'))
                save_mask_png(mm_arr * 255, str(base / 'mm.png'))
                # save loss_mask if present in meta or sample
                loss_mask_np = loss_mask.squeeze().cpu().numpy().astype(np.uint8)
                save_mask_png(loss_mask_np*255, str(base / 'loss_mask.png'))

            if args.save_preds:
                dtype = np.float16 if args.pred_dtype=='float16' else np.float32
                mm_np = mm_pred.squeeze().detach().cpu().numpy().astype(dtype)
                ms_np = ms_pred.squeeze().detach().cpu().numpy().astype(dtype)
                np.save(str(base / 'manidet_m_pred.npy'), mm_np)
                np.save(str(base / 'simidet_m_pred.npy'), ms_np)
                # save a binary PNG of the predicted mask for quick visual inspection
                pred_png = (pred_bin * 255).astype(np.uint8)
                Image.fromarray(pred_png).save(str(base / 'simidet_m_pred_bin.png'))

            if args.save_S_P:
                # optional large saves
                np.save(str(base / 'S_raw.npy'), to_np(S_raw))
                np.save(str(base / 'P.npy'), to_np(P))

            # Write per-sample JSON log
            logf.write(json.dumps(sample_log) + '\n')
            saved += 1
            if (idx+1) % 10 == 0:
                print(f"Processed {idx+1}/{args.n_samples} samples ...")

    end_time = time.time()
    print(f"Done. Processed {saved} samples successfully. Failures: {failures}. Time elapsed: {end_time-start_time:.1f}s")
    print(f"Per-sample logs written to {log_path}")

if __name__ == '__main__':
    main()
