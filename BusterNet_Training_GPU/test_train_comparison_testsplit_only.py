"""
Test script for train_comparison.py checkpoints on combined_dataset TEST SPLIT ONLY.
Uses:
- Pristine images: FPR and Specificity (no forgery in GT)
- COCO subset: per-class recall (BG, SRC, TGT)
- DEFACTO: per-class recall + manipulation recall Rm = TPm / (TPm + FNm)
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')

from model_busternet import BusterNet, build_busternet_full
from model_simidet import build_simidet
from model_manidet import build_manidet
from preprocessing import BusterNetPreprocess
from dataset import SyntheticCMFDataset


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
def compute_forgery_metrics(pred_logits, target_mask, loss_mask):
    """
    Standard metrics for forgery images (COCO, DEFACTO) with all 3 classes.
    Returns: pixel_acc, mean_iou, recalls[3], ious[3]
    """
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
    
    return {
        'pixel_acc': pixel_acc,
        'bg_iou': bg_iou,
        'bg_recall': bg_recall,
        'src_fpr': src_fpr,
        'tgt_fpr': tgt_fpr,
        'src_specificity': src_specificity,
        'tgt_specificity': tgt_specificity,
    }


def compute_manipulation_recall(pred_logits, target_mask, loss_mask):
    """
    Manipulation recall for DEFACTO: Rm = TPm / (TPm + FNm)
    
    TPm = pixels correctly predicted as either source (1) or target (2)
    FNm = pixels that are source or target in GT but predicted as background (0)
    
    This measures: of all manipulated pixels (src + tgt), how many were detected?
    """
    pred_prob = torch.softmax(pred_logits, dim=1)
    pred_class = torch.argmax(pred_prob, dim=1)
    
    if loss_mask.dim() == 4:
        valid = loss_mask.squeeze(1) > 0.5
    else:
        valid = loss_mask > 0.5
    
    pred_valid = pred_class[valid]
    target_valid = target_mask[valid]
    
    # Manipulated pixels in GT: source (1) OR target (2)
    manipulated_gt = (target_valid == 1) | (target_valid == 2)
    
    # Predicted as manipulated: source (1) OR target (2)
    manipulated_pred = (pred_valid == 1) | (pred_valid == 2)
    
    # TPm: correctly detected manipulated pixels
    tp_m = (manipulated_pred & manipulated_gt).float().sum()
    
    # FNm: manipulated in GT but predicted as background
    fn_m = (~manipulated_pred & manipulated_gt).float().sum()
    
    if (tp_m + fn_m).item() == 0:
        return float('nan')
    
    return (tp_m / (tp_m + fn_m) * 100).item()


def save_prediction_visual(pred_tensor, image_tensor, mc_tensor, sample_id, out_dir, prefix=''):
    """Save prediction visualization."""
    os.makedirs(out_dir, exist_ok=True)
    
    # pred_tensor: [3, H, W] softmax -> class indices
    pred_class = torch.argmax(pred_tensor, dim=0).cpu().numpy().astype(np.uint8)
    pred_vis = np.vectorize({0:0, 1:127, 2:255}.get)(pred_class).astype(np.uint8)
    
    # GT mask
    gt_mask = mc_tensor.squeeze().cpu().numpy().astype(np.uint8)
    if gt_mask.max() > 2:
        gt_vis = np.vectorize({0:0, 1:127, 2:255}.get)(gt_mask).astype(np.uint8)
    else:
        gt_vis = (gt_mask * 127).astype(np.uint8)
    
    # Image
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    Image.fromarray(img).save(os.path.join(out_dir, f'{prefix}{sample_id}_img.png'))
    Image.fromarray(gt_vis).save(os.path.join(out_dir, f'{prefix}{sample_id}_gt.png'))
    Image.fromarray(pred_vis).save(os.path.join(out_dir, f'{prefix}{sample_id}_pred.png'))


# =============================================================================
# Main testing functions
# =============================================================================
def run_test_coco(model, dataloader, device, save_dir, prefix='', save_vis=True):
    """Test on COCO subset - standard per-class recall."""
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
            
            for i in range(images.shape[0]):
                pixel_acc, mean_iou, recalls, ious = compute_forgery_metrics(
                    pred_logits[i:i+1], mc_masks[i:i+1], loss_masks[i:i+1]
                )
                
                meta = batch['meta']
                row = meta['manifest_row'][i] if isinstance(meta['manifest_row'], list) else meta['manifest_row']
                sample_id = row.get('sample_id', f'sample_{i}')
                
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
    print(f"Mean IoU:   {np.nanmean(all_metrics['mean_iou']):.2f}%")
    print(f"Recall BG:  {np.nanmean(all_metrics['recall_bg']):.2f}%")
    print(f"Recall SRC: {np.nanmean(all_metrics['recall_src']):.2f}%")
    print(f"Recall TGT: {np.nanmean(all_metrics['recall_tgt']):.2f}%")
    print(f"IoU BG:     {np.nanmean(all_metrics['iou_bg']):.2f}%")
    print(f"IoU SRC:    {np.nanmean(all_metrics['iou_src']):.2f}%")
    print(f"IoU TGT:    {np.nanmean(all_metrics['iou_tgt']):.2f}%")
    
    # Save detailed results
    results_path = os.path.join(save_dir, f'{prefix}results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'pixel_acc': np.mean(all_metrics['pixel_acc']),
                'mean_iou': np.nanmean(all_metrics['mean_iou']),
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
    """Test on PRISTINE images - FPR and Specificity for src/tgt."""
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
            
            for i in range(images.shape[0]):
                metrics = compute_pristine_metrics(
                    pred_logits[i:i+1], mc_masks[i:i+1], loss_masks[i:i+1]
                )
                
                meta = batch['meta']
                row = meta['manifest_row'][i] if isinstance(meta['manifest_row'], list) else meta['manifest_row']
                sample_id = row.get('sample_id', f'sample_{i}')
                
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


def run_test_defacto(model, dataloader, device, save_dir, prefix='', save_vis=True):
    """Test on DEFACTO - per-class recall + manipulation recall Rm."""
    model.eval()
    
    all_metrics = {
        'pixel_acc': [],
        'mean_iou': [],
        'recall_bg': [],
        'recall_src': [],
        'recall_tgt': [],
        'manipulation_recall': [],  # Rm = TPm / (TPm + FNm)
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
            
            for i in range(images.shape[0]):
                pixel_acc, mean_iou, recalls, ious = compute_forgery_metrics(
                    pred_logits[i:i+1], mc_masks[i:i+1], loss_masks[i:i+1]
                )
                
                # Manipulation recall
                r_m = compute_manipulation_recall(
                    pred_logits[i:i+1], mc_masks[i:i+1], loss_masks[i:i+1]
                )
                
                meta = batch['meta']
                row = meta['manifest_row'][i] if isinstance(meta['manifest_row'], list) else meta['manifest_row']
                sample_id = row.get('sample_id', f'sample_{i}')
                
                all_metrics['pixel_acc'].append(pixel_acc)
                all_metrics['mean_iou'].append(mean_iou)
                all_metrics['recall_bg'].append(recalls[0])
                all_metrics['recall_src'].append(recalls[1])
                all_metrics['recall_tgt'].append(recalls[2])
                all_metrics['manipulation_recall'].append(r_m)
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
                    'manipulation_recall': r_m,
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
    print(f"Pixel Acc:           {np.mean(all_metrics['pixel_acc']):.2f}%")
    print(f"Mean IoU:            {np.nanmean(all_metrics['mean_iou']):.2f}%")
    print(f"Recall BG:           {np.nanmean(all_metrics['recall_bg']):.2f}%")
    print(f"Recall SRC:          {np.nanmean(all_metrics['recall_src']):.2f}%")
    print(f"Recall TGT:          {np.nanmean(all_metrics['recall_tgt']):.2f}%")
    print(f"Manipulation Recall: {np.nanmean(all_metrics['manipulation_recall']):.2f}%")
    print(f"IoU BG:              {np.nanmean(all_metrics['iou_bg']):.2f}%")
    print(f"IoU SRC:             {np.nanmean(all_metrics['iou_src']):.2f}%")
    print(f"IoU TGT:             {np.nanmean(all_metrics['iou_tgt']):.2f}%")
    
    results_path = os.path.join(save_dir, f'{prefix}results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'pixel_acc': np.mean(all_metrics['pixel_acc']),
                'mean_iou': np.nanmean(all_metrics['mean_iou']),
                'recall_bg': float(np.nanmean(all_metrics['recall_bg'])),
                'recall_src': float(np.nanmean(all_metrics['recall_src'])),
                'recall_tgt': float(np.nanmean(all_metrics['recall_tgt'])),
                'manipulation_recall': float(np.nanmean(all_metrics['manipulation_recall'])),
                'iou_bg': float(np.nanmean(all_metrics['iou_bg'])),
                'iou_src': float(np.nanmean(all_metrics['iou_src'])),
                'iou_tgt': float(np.nanmean(all_metrics['iou_tgt'])),
            },
            'per_sample': sample_results
        }, f, indent=2)
    print(f"Detailed results saved to: {results_path}")
    
    return all_metrics


# =============================================================================
# Dataset class for combined_dataset (filter by category)
# =============================================================================
class CombinedDataset(Dataset):
    """Wrapper around SyntheticCMFDataset with category filtering."""
    def __init__(self, manifest_csv, root_dir, img_size=256, category=None):
        self.base_dataset = SyntheticCMFDataset(manifest_csv, root_dir, img_size)
        self.category = category
        
        if category:
            candidate_indices = [
                i for i, row in enumerate(self.base_dataset.rows)
                if (row.get('category') or 'coco').lower() == category.lower()
            ]
        else:
            candidate_indices = list(range(len(self.base_dataset)))
        
        # Verify files exist
        self.indices = []
        for i in candidate_indices:
            try:
                _ = self.base_dataset[i]
                self.indices.append(i)
            except FileNotFoundError:
                pass  # Skip missing files
        
        cat_str = f" ({category})" if category else ""
        print(f"Loaded {len(self.indices)} valid samples from {manifest_csv}{cat_str} (skipped {len(candidate_indices) - len(self.indices)} missing)")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def main():
    parser = argparse.ArgumentParser(description='Test train_comparison.py checkpoint on combined_dataset TEST SPLIT ONLY')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--test-manifest', type=str, default=r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test\manifest.csv')
    parser.add_argument('--test-root', type=str, default=r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default=r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\test_results_train_comparison_testsplit')
    parser.add_argument('--save-vis', action='store_true', help='Save prediction visualizations')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print('Loading model...')
    model = BusterNetWithAux(target_size=args.img_size).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    epoch = checkpoint.get("epoch", "N/A")
    loss = checkpoint.get("loss", checkpoint.get("best_val_acc", "N/A"))
    if isinstance(loss, (int, float)):
        print(f'Checkpoint epoch: {epoch}, val_loss: {loss:.4f}')
    else:
        print(f'Checkpoint epoch: {epoch}, val_loss: {loss}')
    
    state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model_state'
    model.load_state_dict(checkpoint[state_key], strict=True)
    print('Model loaded successfully.')
    
    # =========================================================================
    # Test 1: COCO subset (test split)
    # =========================================================================
    print('\n' + '='*60)
    print('TEST 1: COCO Subset (test split)')
    print('='*60)
    
    coco_dataset = CombinedDataset(args.test_manifest, args.test_root, args.img_size, category='COCO_Subset')
    coco_loader = DataLoader(coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    run_test_coco(model, coco_loader, device, args.output_dir, prefix='coco_test_', save_vis=args.save_vis)
    
    # =========================================================================
    # Test 2: DEFACTO (test split)
    # =========================================================================
    print('\n' + '='*60)
    print('TEST 2: DEFACTO (test split)')
    print('='*60)
    
    defacto_dataset = CombinedDataset(args.test_manifest, args.test_root, args.img_size, category='DEFACTO')
    defacto_loader = DataLoader(defacto_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    run_test_defacto(model, defacto_loader, device, args.output_dir, prefix='defacto_test_', save_vis=args.save_vis)
    
    # =========================================================================
    # Test 3: PRISTINE (test split)
    # =========================================================================
    print('\n' + '='*60)
    print('TEST 3: PRISTINE (test split)')
    print('='*60)
    print('NOTE: Using pristine-specific metrics (FPR/Specificity for src/tgt)')
    print('      since GT only has background (class 0)')
    
    pristine_dataset = CombinedDataset(args.test_manifest, args.test_root, args.img_size, category='PRISTINE_Symmetrical')
    pristine_loader = DataLoader(pristine_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    run_test_pristine(model, pristine_loader, device, args.output_dir, prefix='pristine_test_', save_vis=args.save_vis)
    
    print('\n' + '='*60)
    print('ALL TEST SPLIT TESTS COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()