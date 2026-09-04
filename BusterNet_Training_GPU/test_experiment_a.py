"""
Test script for Experiment A checkpoint on:
1. 50 COCO synthetic copy-move forgeries from custom_test_dataset
2. 10 training images from Experiment A train split (seed=42, val_split=0.1)

Uses checkpoint: C:/Users/dines/Downloads/best_model_expA.pth
Architecture: BusterNet from BusterNet_Training_GPU
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms as T

# Import BusterNet components
import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')

from model_busternet import BusterNet, build_busternet_full
from model_simidet import build_simidet
from model_manidet import build_manidet
from preprocessing import BusterNetPreprocess


# =============================================================================
# Dataset for custom_test_dataset (all categories)
# =============================================================================
class CustomTestDataset(Dataset):
    def __init__(self, manifest_csv, root_dir, img_size=256, category=None):
        self.manifest_csv = manifest_csv
        self.root_dir = root_dir
        self.img_size = img_size
        self.category = category
        
        with open(manifest_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if category:
                self.rows = [r for r in rows if r['category'] == category]
            else:
                self.rows = rows
        
        cat_str = f" ({category})" if category else ""
        print(f"Loaded {len(self.rows)} samples from custom_test_dataset{cat_str}")
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, 'images', row['file_name'])
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = T.functional.to_tensor(img).type(torch.float32)  # [3,H,W] in [0,1] RGB
        
        # Load 3-class mask
        mc_path = os.path.join(self.root_dir, row['mc_file_name'])
        mc_img = Image.open(mc_path).convert('L')
        mc_img = mc_img.resize((self.img_size, self.img_size), Image.NEAREST)
        mc_np = np.array(mc_img, dtype=np.int64)
        
        # Map {0, 127, 255} -> {0, 1, 2}
        if set(np.unique(mc_np)).issubset({0, 127, 255}):
            mapping = {0: 0, 127: 1, 255: 2}
            mc_np = np.vectorize(lambda v: mapping.get(int(v), 0))(mc_np).astype(np.int64)
        elif set(np.unique(mc_np)).issubset({0, 1, 2}):
            pass  # already mapped
        else:
            raise ValueError(f"Unexpected mask values: {np.unique(mc_np)}")
        
        mc_tensor = torch.from_numpy(mc_np)[None, ...]  # [1,H,W]
        
        # Load binary mask
        mm_path = os.path.join(self.root_dir, row['mm_file_name'])
        mm_img = Image.open(mm_path).convert('L')
        mm_img = mm_img.resize((self.img_size, self.img_size), Image.NEAREST)
        mm_np = np.array(mm_img, dtype=np.uint8)
        mm_np = (mm_np > 0).astype(np.int64)
        mm_tensor = torch.from_numpy(mm_np)[None, ...]
        
        # Loss mask (all ones - no padding)
        loss_mask = torch.ones((self.img_size, self.img_size), dtype=torch.float32)
        
        return {
            'image': img_tensor,           # [3,H,W] in [0,1] RGB
            'mc': mc_tensor,               # [1,H,W] in {0,1,2}
            'mm': mm_tensor.float(),       # [1,H,W] in {0,1}
            'loss_mask': loss_mask[None, ...],  # [1,H,W]
            'sample_id': row['sample_id'],
            'original_size': original_size,
            'category': row['category'],
        }


# =============================================================================
# Dataset for Experiment A training data (to get the 10 training images)
# =============================================================================
class ExperimentATrainDataset(Dataset):
    """
    Recreates the exact train/val split from train_comparison.py Experiment A:
    - manifest: coco_subset/forgery_manifest_subset_2000.csv
    - root: coco_subset
    - val_split: 0.1
    - seed: 42
    """
    def __init__(self, manifest_csv, root_dir, img_size=256, val_split=0.1, seed=42, get_train=True):
        self.manifest_csv = manifest_csv
        self.root_dir = root_dir
        self.img_size = img_size
        self.get_train = get_train
        
        # Load full dataset (same as SyntheticCMFDataset)
        from dataset import SyntheticCMFDataset
        full_dataset = SyntheticCMFDataset(
            manifest_csv=manifest_csv,
            root=root_dir,
            img_size=img_size
        )
        
        # Recreate exact same split
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        if get_train:
            self.dataset = train_dataset
            print(f"Experiment A train split: {len(self.dataset)} samples")
        else:
            self.dataset = val_dataset
            print(f"Experiment A val split: {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# =============================================================================
# Model wrapper for inference (matches train_comparison.py BusterNetWithAux)
# =============================================================================
class BusterNetWithAux(torch.nn.Module):
    def __init__(self, target_size=256):
        super().__init__()
        self.busternet = build_busternet_full(target_size=target_size)
        self.simidet = self.busternet.simidet
        self.manidet = self.busternet.manidet
        self.preprocess_layer = self.busternet.preprocess_layer
        self.fusion = self.busternet.fusion
        self.pred = self.busternet.pred
        self.target_size = self.busternet.target_size

    def preprocess(self, x):
        return self.preprocess_layer(x)

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        x = self.preprocess(x)
        
        simi_out = self.simidet(x)
        mani_out = self.manidet(x)
        simi_feat = simi_out['d']
        mani_feat = mani_out['d']
        simi_m_pred = simi_out['m_pred']
        mani_m_pred = mani_out['m_pred']
        
        fused = torch.cat([simi_feat, mani_feat], dim=1)
        fused = self.fusion(fused)
        pred_logits = self.pred(fused)
        pred_prob = torch.softmax(pred_logits, dim=1)
        
        if orig_h != self.target_size or orig_w != self.target_size:
            pred_prob = F.interpolate(
                pred_prob, size=(orig_h, orig_w), mode='bilinear', align_corners=True
            )
            simi_m_pred = F.interpolate(
                simi_m_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=True
            )
            mani_m_pred = F.interpolate(
                mani_m_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=True
            )
            pred_logits = F.interpolate(
                pred_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=True
            )
        
        return {
            'mc_pred': pred_prob,
            'simi_m_pred': simi_m_pred,
            'mani_m_pred': mani_m_pred,
            'mc_logits': pred_logits,
        }


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(pred_logits, target_mask, loss_mask):
    """Standard metrics for forgery images (COCO, DEFACTO, train) with all 3 classes."""
    pred_prob = torch.softmax(pred_logits, dim=1)
    pred_class = torch.argmax(pred_prob, dim=1)
    
    if loss_mask.dim() == 4:
        valid = loss_mask.squeeze(1) > 0.5
    else:
        valid = loss_mask > 0.5
    
    pred_valid = pred_class[valid]
    target_valid = target_mask[valid]
    
    # Pixel accuracy
    correct = (pred_valid == target_valid).float().sum()
    total = valid.float().sum()
    pixel_acc = (correct / total * 100).item() if total > 0 else 0.0
    
    # IoU and Recall per class
    ious = []
    recalls = []
    for cls in range(3):
        pred_inds = (pred_valid == cls)
        target_inds = (target_valid == cls)
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        iou = (intersection / union * 100).item() if union > 0 else float('nan')
        ious.append(iou)
        
        tp = intersection
        fn = (~pred_inds & target_inds).float().sum()
        recall = (tp / (tp + fn) * 100).item() if (tp + fn) > 0 else float('nan')
        recalls.append(recall)
    
    mean_iou = np.nanmean(ious)
    return pixel_acc, mean_iou, recalls, ious


def compute_pristine_metrics(pred_logits, target_mask, loss_mask):
    """
    Metrics for pristine images where GT only has background (class 0).
    
    For pristine images:
    - GT has only class 0 (background)
    - Source (class 1) and Target (class 2) have NO ground truth pixels
    - Standard IoU/Recall for src/tgt are undefined (denominator = 0)
    
    Appropriate metrics:
    - Pixel Accuracy (background accuracy)
    - Background IoU and Recall
    - False Positive Rate (FPR) for source and target
    - Specificity (True Negative Rate) for source and target
    """
    pred_prob = torch.softmax(pred_logits, dim=1)
    pred_class = torch.argmax(pred_prob, dim=1)
    
    if loss_mask.dim() == 4:
        valid = loss_mask.squeeze(1) > 0.5
    else:
        valid = loss_mask > 0.5
    
    pred_valid = pred_class[valid]
    target_valid = target_mask[valid]
    
    # Pixel accuracy (background accuracy)
    correct = (pred_valid == target_valid).float().sum()
    total = valid.float().sum()
    pixel_acc = (correct / total * 100).item() if total > 0 else 0.0
    
    # Background metrics (class 0)
    bg_pred = (pred_valid == 0)
    bg_target = (target_valid == 0)
    bg_intersection = (bg_pred & bg_target).float().sum()
    bg_union = (bg_pred | bg_target).float().sum()
    bg_iou = (bg_intersection / bg_union * 100).item() if bg_union > 0 else float('nan')
    
    bg_tp = bg_intersection
    bg_fn = (~bg_pred & bg_target).float().sum()
    bg_recall = (bg_tp / (bg_tp + bg_fn) * 100).item() if (bg_tp + bg_fn) > 0 else float('nan')
    
    # For source (class 1) and target (class 2): GT has NO positive pixels
    # So we compute False Positive Rate and Specificity instead
    src_pred = (pred_valid == 1)
    tgt_pred = (pred_valid == 2)
    
    # Total background pixels in GT (all valid pixels are background for pristine)
    total_bg = bg_target.float().sum()
    
    # False positives for src and tgt
    src_fp = (src_pred & bg_target).float().sum()
    tgt_fp = (tgt_pred & bg_target).float().sum()
    
    # True negatives for src and tgt
    src_tn = (~src_pred & bg_target).float().sum()
    tgt_tn = (~tgt_pred & bg_target).float().sum()
    
    # False Positive Rate = FP / (FP + TN) = FP / total_bg
    src_fpr = (src_fp / total_bg * 100).item() if total_bg > 0 else 0.0
    tgt_fpr = (tgt_fp / total_bg * 100).item() if total_bg > 0 else 0.0
    
    # Specificity (True Negative Rate) = TN / (TN + FP) = TN / total_bg
    src_specificity = (src_tn / total_bg * 100).item() if total_bg > 0 else 100.0
    tgt_specificity = (tgt_tn / total_bg * 100).item() if total_bg > 0 else 100.0
    
    # Precision for src/tgt is undefined (TP = 0), but we can report it as 0 if FP > 0
    # Or better: don't report precision for pristine
    
    return {
        'pixel_acc': pixel_acc,
        'bg_iou': bg_iou,
        'bg_recall': bg_recall,
        'src_fpr': src_fpr,
        'tgt_fpr': tgt_fpr,
        'src_specificity': src_specificity,
        'tgt_specificity': tgt_specificity,
    }


def save_prediction_visual(pred_tensor, image_tensor, mc_tensor, sample_id, out_dir, prefix=''):
    """Save prediction visualization."""
    os.makedirs(out_dir, exist_ok=True)
    
    # pred_tensor: [3, H, W] softmax -> class indices
    pred_class = torch.argmax(pred_tensor, dim=0).cpu().numpy().astype(np.uint8)
    pred_vis = np.vectorize({0:0, 1:127, 2:255}.get)(pred_class).astype(np.uint8)
    
    # GT mask
    gt_mask = mc_tensor.squeeze().cpu().numpy().astype(np.uint8)
    if gt_mask.max() > 1:
        gt_vis = np.vectorize({0:0, 1:127, 2:255}.get)(gt_mask).astype(np.uint8)
    else:
        gt_vis = (gt_mask * 255).astype(np.uint8)
    
    # Image
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    Image.fromarray(img).save(os.path.join(out_dir, f'{prefix}{sample_id}_img.png'))
    Image.fromarray(gt_vis).save(os.path.join(out_dir, f'{prefix}{sample_id}_gt.png'))
    Image.fromarray(pred_vis).save(os.path.join(out_dir, f'{prefix}{sample_id}_pred.png'))


# =============================================================================
# Main testing function
# =============================================================================
def run_test(model, dataloader, device, save_dir, prefix='', save_vis=True):
    model.eval()
    
    all_metrics = {
        'pixel_acc': [],
        'mean_iou': [],
        'recall_bg': [],
        'recall_src': [],
        'recall_tgt': [],
        'iou_bg': [],
        'iou_src': [],
        'iou_tgt': [],
    }
    
    sample_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            mc_masks = batch['mc'].to(device).squeeze(1).long()
            loss_masks = batch['loss_mask'].to(device).float()
            
            outputs = model(images)
            pred_logits = outputs['mc_logits']
            
            # Compute metrics per sample
            for i in range(images.shape[0]):
                pixel_acc, mean_iou, recalls, ious = compute_metrics(
                    pred_logits[i:i+1], mc_masks[i:i+1], loss_masks[i:i+1]
                )
                
                # Get sample_id (different datasets return it differently)
                if 'sample_id' in batch:
                    sample_ids = batch['sample_id']
                    sample_id = sample_ids[i] if isinstance(sample_ids, list) else sample_ids[i].item() if hasattr(sample_ids[i], 'item') else sample_ids[i]
                elif 'meta' in batch:
                    meta = batch['meta']
                    if isinstance(meta, list):
                        row = meta[i]
                    elif isinstance(meta, dict) and 'manifest_row' in meta:
                        row = meta['manifest_row'][i] if isinstance(meta['manifest_row'], list) else meta['manifest_row']
                    else:
                        row = meta[i] if isinstance(meta, list) else meta
                    sample_id = row.get('sample_id', f'sample_{i}')
                else:
                    sample_id = f'sample_{i}'
                
                all_metrics['pixel_acc'].append(pixel_acc)
                all_metrics['mean_iou'].append(mean_iou)
                all_metrics['recall_bg'].append(recalls[0])
                all_metrics['recall_src'].append(recalls[1])
                all_metrics['recall_tgt'].append(recalls[2])
                all_metrics['iou_bg'].append(ious[0])
                all_metrics['iou_src'].append(ious[1])
                all_metrics['iou_tgt'].append(ious[2])
                
                sample_results.append({
                    'sample_id': sample_id,
                    'pixel_acc': pixel_acc,
                    'mean_iou': mean_iou,
                    'recall_bg': recalls[0],
                    'recall_src': recalls[1],
                    'recall_tgt': recalls[2],
                    'iou_bg': ious[0],
                    'iou_src': ious[1],
                    'iou_tgt': ious[2],
                })
                
                if save_vis:
                    save_prediction_visual(
                        outputs['mc_pred'][i].cpu(),
                        images[i].cpu(),
                        mc_masks[i:i+1].cpu(),
                        sample_id,
                        save_dir,
                        prefix=prefix
                    )
    
    # Aggregate metrics
    print(f"\n=== {prefix} Results ({len(sample_results)} samples) ===")
    print(f"Pixel Acc:  {np.mean(all_metrics['pixel_acc']):.2f}%")
    print(f"Mean IoU:   {np.mean(all_metrics['mean_iou']):.2f}%")
    print(f"Recall BG:  {np.nanmean(all_metrics['recall_bg']):.2f}%")
    print(f"Recall SRC: {np.nanmean(all_metrics['recall_src']):.2f}%")
    print(f"Recall TGT: {np.nanmean(all_metrics['recall_tgt']):.2f}%")
    print(f"IoU BG:     {np.nanmean(all_metrics['iou_bg']):.2f}%")
    print(f"IoU SRC:    {np.nanmean(all_metrics['iou_src']):.2f}%")
    print(f"IoU TGT:    {np.nanmean(all_metrics['iou_tgt']):.2f}%")
    
    # Save detailed results
    import json
    results_path = os.path.join(save_dir, f'{prefix}results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'pixel_acc': np.mean(all_metrics['pixel_acc']),
                'mean_iou': np.mean(all_metrics['mean_iou']),
                'recall_bg': float(np.nanmean(all_metrics['recall_bg'])),
                'recall_src': float(np.nanmean(all_metrics['recall_src'])),
                'recall_tgt': float(np.nanmean(all_metrics['recall_tgt'])),
                'iou_bg': float(np.nanmean(all_metrics['iou_bg'])),
                'iou_src': float(np.nanmean(all_metrics['iou_src'])),
                'iou_tgt': float(np.nanmean(all_metrics['iou_tgt'])),
            },
            'per_sample': sample_results
        }, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    return all_metrics


def run_test_pristine(model, dataloader, device, save_dir, prefix='', save_vis=True):
    """
    Test function for pristine images with appropriate metrics.
    GT only has background (class 0), so src/tgt IoU/Recall are undefined.
    Uses FPR and Specificity for src/tgt instead.
    """
    model.eval()
    
    all_metrics = {
        'pixel_acc': [],
        'bg_iou': [],
        'bg_recall': [],
        'src_fpr': [],
        'tgt_fpr': [],
        'src_specificity': [],
        'tgt_specificity': [],
    }
    
    sample_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            mc_masks = batch['mc'].to(device).squeeze(1).long()
            loss_masks = batch['loss_mask'].to(device).float()
            
            outputs = model(images)
            pred_logits = outputs['mc_logits']
            
            # Compute metrics per sample
            for i in range(images.shape[0]):
                metrics = compute_pristine_metrics(
                    pred_logits[i:i+1], mc_masks[i:i+1], loss_masks[i:i+1]
                )
                
                # Get sample_id
                if 'sample_id' in batch:
                    sample_ids = batch['sample_id']
                    sample_id = sample_ids[i] if isinstance(sample_ids, list) else sample_ids[i].item() if hasattr(sample_ids[i], 'item') else sample_ids[i]
                elif 'meta' in batch:
                    meta = batch['meta']
                    if isinstance(meta, list):
                        row = meta[i]
                    elif isinstance(meta, dict) and 'manifest_row' in meta:
                        row = meta['manifest_row'][i] if isinstance(meta['manifest_row'], list) else meta['manifest_row']
                    else:
                        row = meta[i] if isinstance(meta, list) else meta
                    sample_id = row.get('sample_id', f'sample_{i}')
                else:
                    sample_id = f'sample_{i}'
                
                all_metrics['pixel_acc'].append(metrics['pixel_acc'])
                all_metrics['bg_iou'].append(metrics['bg_iou'])
                all_metrics['bg_recall'].append(metrics['bg_recall'])
                all_metrics['src_fpr'].append(metrics['src_fpr'])
                all_metrics['tgt_fpr'].append(metrics['tgt_fpr'])
                all_metrics['src_specificity'].append(metrics['src_specificity'])
                all_metrics['tgt_specificity'].append(metrics['tgt_specificity'])
                
                sample_results.append({
                    'sample_id': sample_id,
                    'pixel_acc': metrics['pixel_acc'],
                    'bg_iou': metrics['bg_iou'],
                    'bg_recall': metrics['bg_recall'],
                    'src_fpr': metrics['src_fpr'],
                    'tgt_fpr': metrics['tgt_fpr'],
                    'src_specificity': metrics['src_specificity'],
                    'tgt_specificity': metrics['tgt_specificity'],
                })
                
                if save_vis:
                    save_prediction_visual(
                        outputs['mc_pred'][i].cpu(),
                        images[i].cpu(),
                        mc_masks[i:i+1].cpu(),
                        sample_id,
                        save_dir,
                        prefix=prefix
                    )
    
    # Aggregate metrics
    print(f"\n=== {prefix} Results ({len(sample_results)} samples) ===")
    print(f"Pixel Acc (BG Acc):  {np.mean(all_metrics['pixel_acc']):.2f}%")
    print(f"BG IoU:              {np.nanmean(all_metrics['bg_iou']):.2f}%")
    print(f"BG Recall:           {np.nanmean(all_metrics['bg_recall']):.2f}%")
    print(f"SRC False Positive Rate: {np.mean(all_metrics['src_fpr']):.2f}%")
    print(f"TGT False Positive Rate: {np.mean(all_metrics['tgt_fpr']):.2f}%")
    print(f"SRC Specificity:     {np.mean(all_metrics['src_specificity']):.2f}%")
    print(f"TGT Specificity:     {np.mean(all_metrics['tgt_specificity']):.2f}%")
    
    # Save detailed results
    import json
    results_path = os.path.join(save_dir, f'{prefix}results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'pixel_acc': np.mean(all_metrics['pixel_acc']),
                'bg_iou': float(np.nanmean(all_metrics['bg_iou'])),
                'bg_recall': float(np.nanmean(all_metrics['bg_recall'])),
                'src_fpr': np.mean(all_metrics['src_fpr']),
                'tgt_fpr': np.mean(all_metrics['tgt_fpr']),
                'src_specificity': np.mean(all_metrics['src_specificity']),
                'tgt_specificity': np.mean(all_metrics['tgt_specificity']),
            },
            'per_sample': sample_results
        }, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    return all_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Paths
    checkpoint_path = r'C:\Users\dines\Downloads\best_model_expA.pth'
    custom_test_manifest = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\custom_test_dataset\manifest_test_dataset.csv'
    custom_test_root = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\custom_test_dataset'
    expA_manifest = r'C:\ml_work\synthetic_datasets\coco_subset\forgery_manifest_subset_2000.csv'
    expA_root = r'C:\ml_work\synthetic_datasets\coco_subset'
    output_dir = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\test_results_expA'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print('Loading model...')
    model = BusterNetWithAux(target_size=256).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f'Checkpoint epoch: {checkpoint["epoch"]}, val_loss: {checkpoint["loss"]:.4f}')
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('Model loaded successfully.')
    
    # =========================================================================
    # Test 1: COCO synthetic copy-move forgeries from custom_test_dataset
    # =========================================================================
    print('\n' + '='*60)
    print('TEST 1: COCO synthetic copy-move forgeries (custom_test_dataset)')
    print('='*60)
    
    coco_dataset = CustomTestDataset(custom_test_manifest, custom_test_root, img_size=256, category='COCO_Subset')
    coco_loader = DataLoader(coco_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    run_test(model, coco_loader, device, output_dir, prefix='coco50_', save_vis=True)
    
    # =========================================================================
    # Test 2: DEFACTO from custom_test_dataset
    # =========================================================================
    print('\n' + '='*60)
    print('TEST 2: DEFACTO copy-move forgeries (custom_test_dataset)')
    print('='*60)
    
    defacto_dataset = CustomTestDataset(custom_test_manifest, custom_test_root, img_size=256, category='DEFACTO')
    defacto_loader = DataLoader(defacto_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    run_test(model, defacto_loader, device, output_dir, prefix='defacto50_', save_vis=True)
    
    # =========================================================================
    # Test 3: PRISTINE/Symmetrical from custom_test_dataset
    # =========================================================================
    print('\n' + '='*60)
    print('TEST 3: PRISTINE/Symmetrical (custom_test_dataset)')
    print('='*60)
    print('NOTE: Using pristine-specific metrics (FPR/Specificity for src/tgt)')
    print('      since GT only has background (class 0)')
    
    pristine_dataset = CustomTestDataset(custom_test_manifest, custom_test_root, img_size=256, category='PRISTINE_Symmetrical')
    pristine_loader = DataLoader(pristine_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    run_test_pristine(model, pristine_loader, device, output_dir, prefix='pristine50_', save_vis=True)
    
    # =========================================================================
    # Test 4: 10 training images from Experiment A train split
    # =========================================================================
    print('\n' + '='*60)
    print('TEST 4: 10 training images from Experiment A train split (seed=42)')
    print('='*60)
    
    # Get Experiment A train split
    expA_train_dataset = ExperimentATrainDataset(
        expA_manifest, expA_root, img_size=256, val_split=0.1, seed=42, get_train=True
    )
    
    # Take first 10 samples
    subset_10 = Subset(expA_train_dataset, list(range(10)))
    expA_loader = DataLoader(subset_10, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Testing on {len(subset_10)} training samples from Experiment A split")
    
    run_test(model, expA_loader, device, output_dir, prefix='expA_train10_', save_vis=True)
    
    print('\n' + '='*60)
    print('ALL TESTS COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()