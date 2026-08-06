import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from dataset import SyntheticCMFDataset
from model_busternet import BusterNet, build_busternet_full
from model_utils import masked_bce_loss
import random
import glob

# Helper function for masked cross-entropy loss using logits
def masked_cross_entropy_loss(logits, target, mask):
    """
    logits: [B, C, H, W] raw logits
    target: [B, H, W] long tensor with class indices
    mask:   [B, 1, H, W] or [B, H, W] float mask with 1 for valid pixels, 0 for ignored
    """
    loss = F.cross_entropy(logits, target, reduction='none')  # [B, H, W]
    if mask.dim() == 4:
        mask = mask.squeeze(1)  # [B, H, W]
    loss = loss * mask
    mask_sum = mask.sum()
    if mask_sum.item() == 0:
        return loss.sum() * 0.0
    return loss.sum() / mask_sum

# Helper to compute metrics
def compute_metrics(pred_logits, target_mask, loss_mask):
    """
    pred_logits: [B, 3, H, W]
    target_mask: [B, H, W] long
    loss_mask:   [B, 1, H, W] or [B, H, W]
    Returns: pixel_acc, mean_iou, recall_per_class (list of 3)
    """
    # Get predictions
    pred_prob = torch.softmax(pred_logits, dim=1)  # [B, 3, H, W]
    pred_class = torch.argmax(pred_prob, dim=1)    # [B, H, W]
    
    # Flatten valid pixels
    if loss_mask.dim() == 4:
        valid = loss_mask.squeeze(1) > 0.5   # [B, H, W]
    else:
        valid = loss_mask > 0.5
    
    pred_valid = pred_class[valid]
    target_valid = target_mask[valid]
    
    # Pixel accuracy
    correct = (pred_valid == target_valid).float().sum()
    total = valid.float().sum()
    pixel_acc = (correct / total * 100).item() if total > 0 else 0.0
    
    # IoU per class
    ious = []
    recalls = []
    for cls in range(3):
        pred_inds = (pred_valid == cls)
        target_inds = (target_valid == cls)
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        if union.item() == 0:
            iou = float('nan')
        else:
            iou = (intersection / union * 100).item()
        # Recall: TP / (TP + FN)
        tp = intersection
        fn = (~pred_inds & target_inds).float().sum()
        if (tp + fn).item() == 0:
            recall = float('nan')
        else:
            recall = (tp / (tp + fn) * 100).item()
        ious.append(iou)
        recalls.append(recall)
    
    mean_iou = np.nanmean(ious)
    return pixel_acc, mean_iou, recalls

class BusterNetWithAux(nn.Module):
    """
    BusterNet wrapper that also returns auxiliary predictions from each branch and logits.
    """
    def __init__(self, target_size=256):
        super().__init__()
        self.busternet = build_busternet_full(target_size=target_size)
        # We need access to the branches for aux outputs
        self.simidet = self.busternet.simidet
        self.manidet = self.busternet.manidet
        self.preprocess_layer = self.busternet.preprocess_layer
        self.fusion = self.busternet.fusion
        self.pred = self.busternet.pred
        self.target_size = self.busternet.target_size

    def preprocess(self, x):
        return self.preprocess_layer(x)

    def forward(self, x):
        # Save original size
        orig_h, orig_w = x.shape[2], x.shape[3]
        # Preprocess
        x = self.preprocess(x)
        # Extract features from both branches
        simi_out = self.simidet(x)
        mani_out = self.manidet(x)
        simi_feat = simi_out['d']   # [B, 6, target_size, target_size]
        mani_feat = mani_out['d']   # [B, 6, target_size, target_size]
        # Auxiliary predictions
        simi_m_pred = simi_out['m_pred']  # [B, 1, target_size, target_size]
        mani_m_pred = mani_out['m_pred']  # [B, 1, target_size, target_size]
        # Concatenate features: 6 + 6 = 12 channels
        fused = torch.cat([simi_feat, mani_feat], dim=1)
        # Fusion BN-Inception: 12 -> 9 channels
        fused = self.fusion(fused)
        # Prediction: 9 -> 3 channels (logits)
        pred_logits = self.pred(fused)
        # Softmax to get probabilities
        pred_prob = torch.softmax(pred_logits, dim=1)
        # Resize back to original resolution
        if orig_h != self.target_size or orig_w != self.target_size:
            pred_prob = F.interpolate(
                pred_prob,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )
            # Also resize auxiliary predictions and logits to original size for loss computation
            simi_m_pred = F.interpolate(
                simi_m_pred,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )
            mani_m_pred = F.interpolate(
                mani_m_pred,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )
            pred_logits = F.interpolate(
                pred_logits,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )
        return {
            'mc_pred': pred_prob,          # [B, 3, H, W] softmax probabilities
            'simi_m_pred': simi_m_pred,    # [B, 1, H, W] sigmoid probabilities
            'mani_m_pred': mani_m_pred,    # [B, 1, H, W] sigmoid probabilities
            'mc_logits': pred_logits,      # [B, 3, H, W] raw logits
            'simi_feat': simi_feat,
            'mani_feat': mani_feat
        }

def find_latest_checkpoint(save_dir, experiment):
    """Find the latest checkpoint file for resuming training."""
    pattern = os.path.join(save_dir, f'checkpoint_epoch*_exp{experiment}.pth')
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    # Extract epoch numbers and find the latest
    epochs = []
    for ckpt in checkpoints:
        try:
            epoch_str = ckpt.split(f'checkpoint_epoch')[1].split(f'_exp{experiment}.pth')[0]
            epochs.append((int(epoch_str), ckpt))
        except (IndexError, ValueError):
            continue
    if not epochs:
        return None, 0
    epochs.sort(key=lambda x: x[0])
    latest_epoch, latest_path = epochs[-1]
    return latest_path, latest_epoch

def parse_args():
    parser = argparse.ArgumentParser(description='Train BusterNet on synthetic dataset - comparison experiments')
    parser.add_argument('--experiment', type=str, choices=['A', 'B'], required=True,
                        help='Experiment A: fine-tune pretrained BusterNet; Experiment B: stage-wise training from scratch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (additional epochs from checkpoint)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--save-dir', type=str, default='outputs/comparison', help='Directory to save checkpoints and predictions')
    parser.add_argument('--pretrained-path', type=str, default='busternet_converted.pth',
                        help='Path to pretrained BusterNet .pth weights (for experiment A)')
    parser.add_argument('--manifest', type=str, default='coco_subset/forgery_manifest_subset_2000.csv',
                        help='Path to manifest CSV')
    parser.add_argument('--root', type=str, default='coco_subset', help='Root directory of dataset')
    parser.add_argument('--img-size', type=int, default=256, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help="Force CPU or GPU training; 'auto' (default) picks GPU if available, same as before")
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint in save-dir')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait before LR reduction (plateau)')
    parser.add_argument('--stop-patience', type=int, default=20, help='Epochs to wait after LR reduction before stopping')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], help='Run specific stage only (1=ManiDet, 2=SimiDet, 3=Fusion+Pred, 4=Full fine-tune)')
    parser.add_argument('--stage-epochs', type=int, help='Epochs to run for the specified stage (default: 10/10/5/50 for stages 1/2/3/4)')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_prediction(pred_tensor, image_tensor, mask_tensor, epoch, idx, prefix='', save_dir=''):
    # Save inputs and predictions for visual verification
    out_dir = os.path.join(save_dir, f'{prefix}epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)
    
    # pred_tensor: [3, H, W] softmax probabilities -> convert to class indices
    pred_class = torch.argmax(pred_tensor, dim=0).cpu().numpy().astype(np.uint8)
    # Map class indices to grayscale values for visualization (0->0, 1->127, 2->255)
    # np.vectorize(dict.get) returns int64 by default -- PIL's Image.fromarray()
    # doesn't support int64, only uint8/int32/float32, so it must be cast back down.
    pred_vis = np.vectorize({0:0, 1:127, 2:255}.get)(pred_class).astype(np.uint8)
    gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    # For mc mask, we need to map to 0,127,255 if it's multi-class
    if gt_mask.max() > 1:  # multi-class
        gt_vis = np.vectorize({0:0, 1:127, 2:255}.get)(gt_mask).astype(np.uint8)
    else:
        gt_vis = (gt_mask * 255).astype(np.uint8)
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [0,1] RGB
    
    Image.fromarray(img).save(os.path.join(out_dir, f'sample_{idx}_img.png'))
    Image.fromarray(gt_vis).save(os.path.join(out_dir, f'sample_{idx}_gt.png'))
    Image.fromarray(pred_vis).save(os.path.join(out_dir, f'sample_{idx}_pred.png'))

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False "
                               "on this machine (no GPU, or CUDA/driver not set up). "
                               "Use --device cpu or --device auto instead.")
        device = torch.device('cuda')
    else:  # 'auto' -- unchanged from original behavior
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dataset
    print('Loading dataset...')
    full_dataset = SyntheticCMFDataset(
        manifest_csv=args.manifest,
        root=args.root,
        img_size=args.img_size
    )
    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    print(f'Train size: {train_size}, Val size: {val_size}')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Fixed validation samples for visualization (first 4 samples)
    val_indices = list(range(min(4, len(val_dataset))))
    val_fixed_subset = Subset(val_dataset, val_indices)
    val_fixed_loader = DataLoader(val_fixed_subset, batch_size=1, shuffle=False)
    
    # Model
    print('Building model...')
    if args.experiment == 'A':
        # Experiment A
        model = BusterNetWithAux(target_size=args.img_size).to(device)
        # Load pretrained weights
        if os.path.exists(args.pretrained_path):
            print(f'Loading pretrained weights from {args.pretrained_path}')
            state_dict = torch.load(args.pretrained_path, map_location=device)
            busternet_state_dict = model.busternet.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in busternet_state_dict and busternet_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
            print(f'Loaded {len(filtered_state_dict)} matching parameters out of {len(state_dict)}')
            model.busternet.load_state_dict(filtered_state_dict, strict=False)
            print('Loaded pretrained weights into busternet submodule.')
        else:
            print(f'Pretrained weights not found at {args.pretrained_path}. Starting from scratch.')
    else:
        # Experiment B: start from scratch
        model = BusterNetWithAux(target_size=args.img_size).to(device)
        print('Starting from random initialization.')
    
    # Loss weights for auxiliary loss (can be adjusted)
    aux_weight = 0.5
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    optimizer = None
    stage = 1 if args.experiment == 'B' else 'fine-tune'
    lr_stage = args.lr * 10 if args.experiment == 'B' else args.lr
    plateau_counter = 0
    lr_reduced = False
    
    def create_optimizer(current_stage, current_lr):
        if args.experiment == 'B':
            for name, param in model.named_parameters():
                if current_stage == 1:
                    param.requires_grad = 'manidet' in name
                elif current_stage == 2:
                    param.requires_grad = 'simidet' in name
                elif current_stage == 3:
                    param.requires_grad = ('fusion' in name or 'pred' in name)
                else:
                    param.requires_grad = True
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
    
    if args.resume:
        ckpt_path, start_epoch = find_latest_checkpoint(args.save_dir, args.experiment)
        if ckpt_path:
            print(f'Resuming from checkpoint: {ckpt_path} (epoch {start_epoch})')
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Determine stage from epoch
            if args.experiment == 'B':
                stage1_epochs = 10
                stage2_epochs = 10
                stage3_epochs = 5
                if start_epoch < stage1_epochs:
                    stage = 1
                    lr_stage = args.lr * 10
                elif start_epoch < stage1_epochs + stage2_epochs:
                    stage = 2
                    lr_stage = args.lr * 10
                elif start_epoch < stage1_epochs + stage2_epochs + stage3_epochs:
                    stage = 3
                    lr_stage = args.lr * 10
                else:
                    stage = 4
                    lr_stage = args.lr
            optimizer = create_optimizer(stage, lr_stage)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('loss', float('inf'))
            print(f'Resumed at epoch {start_epoch}, stage {stage}, best_val_loss={best_val_loss:.4f}')
        else:
            print('No checkpoint found, starting from scratch.')
            optimizer = create_optimizer(stage, lr_stage)
    else:
        optimizer = create_optimizer(stage, lr_stage)
    
    print(f'Starting training for experiment {args.experiment} from epoch {start_epoch}...')
    
    # Stage epoch boundaries for Experiment B
    stage1_epochs = 10
    stage2_epochs = 10
    stage3_epochs = 5
    stage4_default_epochs = 50
    prev_stage = 0
    
    # If running specific stage, determine epoch range
    if args.experiment == 'B' and args.stage is not None:
        if args.stage == 1:
            epoch_start = 0
            epoch_end = args.stage_epochs if args.stage_epochs else stage1_epochs
        elif args.stage == 2:
            epoch_start = stage1_epochs
            epoch_end = stage1_epochs + (args.stage_epochs if args.stage_epochs else stage2_epochs)
        elif args.stage == 3:
            epoch_start = stage1_epochs + stage2_epochs
            epoch_end = epoch_start + (args.stage_epochs if args.stage_epochs else stage3_epochs)
        else:  # stage 4
            epoch_start = stage1_epochs + stage2_epochs + stage3_epochs
            epoch_end = epoch_start + (args.stage_epochs if args.stage_epochs else stage4_default_epochs)
        
        # Adjust start_epoch if resuming within this stage
        if start_epoch < epoch_start:
            start_epoch = epoch_start
        # Limit total epochs to run
        max_epoch = min(start_epoch + args.epochs, epoch_end)
    else:
        # Full training
        max_epoch = start_epoch + args.epochs
    
    for epoch in range(start_epoch, max_epoch):
        # Determine current stage for experiment B
        if args.experiment == 'B':
            if epoch < stage1_epochs:
                stage = 1
                lr_stage = args.lr * 10
            elif epoch < stage1_epochs + stage2_epochs:
                stage = 2
                lr_stage = args.lr * 10
            elif epoch < stage1_epochs + stage2_epochs + stage3_epochs:
                stage = 3
                lr_stage = args.lr * 10
            else:
                stage = 4
                lr_stage = args.lr
            
            # Recreate optimizer only when stage changes
            if stage != prev_stage:
                optimizer = create_optimizer(stage, lr_stage)
                prev_stage = stage
        else:
            stage = 'fine-tune'
        
        model.train()
        epoch_loss = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            mc_masks = batch['mc'].to(device).squeeze(1).long()  # [B, H, W]
            mm_masks = batch['mm'].to(device).float()           # [B, 1, H, W]
            loss_masks = batch['loss_mask'].to(device).float()  # [B, 1, H, W]
            
            optimizer.zero_grad()
            
            outputs = model(images)
            pred_logits = outputs['mc_logits']               # [B, 3, H, W] raw logits
            simi_m_pred = outputs['simi_m_pred']             # [B, 1, H, W]
            mani_m_pred = outputs['mani_m_pred']             # [B, 1, H, W]
            
            # Compute loss based on stage
            if args.experiment == 'B':
                if stage == 1:
                    loss = masked_bce_loss(mani_m_pred, mm_masks, loss_masks)
                    main_loss = 0.0
                    aux_loss = loss.item()
                elif stage == 2:
                    simi_target = batch['simi_binary'].to(device).float()
                    loss = masked_bce_loss(simi_m_pred, simi_target, loss_masks)
                    main_loss = 0.0
                    aux_loss = loss.item()
                elif stage == 3:
                    loss = masked_cross_entropy_loss(pred_logits, mc_masks, loss_masks)
                    main_loss = loss.item()
                    aux_loss = 0.0
                else:  # stage 4
                    main_loss_val = masked_cross_entropy_loss(pred_logits, mc_masks, loss_masks)
                    aux_loss_simi = masked_bce_loss(simi_m_pred, mm_masks, loss_masks)
                    aux_loss_mani = masked_bce_loss(mani_m_pred, mm_masks, loss_masks)
                    aux_loss_val = (aux_loss_simi + aux_loss_mani) / 2.0
                    loss = main_loss_val + aux_weight * aux_loss_val
                    main_loss = main_loss_val.item()
                    aux_loss = aux_loss_val.item()
            else:
                # Experiment A: combined loss
                main_loss_val = masked_cross_entropy_loss(pred_logits, mc_masks, loss_masks)
                aux_loss_simi = masked_bce_loss(simi_m_pred, mm_masks, loss_masks)
                aux_loss_mani = masked_bce_loss(mani_m_pred, mm_masks, loss_masks)
                aux_loss_val = (aux_loss_simi + aux_loss_mani) / 2.0
                loss = main_loss_val + aux_weight * aux_loss_val
                main_loss = main_loss_val.item()
                aux_loss = aux_loss_val.item()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_main_loss += main_loss
            epoch_aux_loss += aux_loss
            
            # Save predictions periodically
            if (epoch + 1) % 5 == 0 and i == 0:
                for b_idx in range(images.shape[0]):
                    save_prediction(
                        outputs['mc_pred'][b_idx],
                        images[b_idx],
                        batch['mc'][b_idx],
                        epoch + 1,
                        b_idx,
                        prefix='train_',
                        save_dir=args.save_dir
                    )
        
        avg_loss = epoch_loss / len(train_loader)
        avg_main_loss = epoch_main_loss / len(train_loader)
        avg_aux_loss = epoch_aux_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'pixel_acc': 0.0, 'mean_iou': 0.0, 'recall': [0.0, 0.0, 0.0]}
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                mc_masks = batch['mc'].to(device).squeeze(1).long()
                mm_masks = batch['mm'].to(device).float()
                loss_masks = batch['loss_mask'].to(device).float()
                
                outputs = model(images)
                pred_logits = outputs['mc_logits']
                simi_m_pred = outputs['simi_m_pred']
                mani_m_pred = outputs['mani_m_pred']
                
                # Compute validation loss (same as training loss for experiment A and stage 4 of B)
                if args.experiment == 'B' and stage in [1, 2, 3]:
                    if stage == 1:
                        loss = masked_bce_loss(mani_m_pred, mm_masks, loss_masks)
                    elif stage == 2:
                        simi_target = batch['simi_binary'].to(device).float()
                        loss = masked_bce_loss(simi_m_pred, simi_target, loss_masks)
                    else:  # stage 3
                        loss = masked_cross_entropy_loss(pred_logits, mc_masks, loss_masks)
                else:
                    # Experiment A or stage 4 of B: combined loss
                    main_loss_val = masked_cross_entropy_loss(pred_logits, mc_masks, loss_masks)
                    aux_loss_simi = masked_bce_loss(simi_m_pred, mm_masks, loss_masks)
                    aux_loss_mani = masked_bce_loss(mani_m_pred, mm_masks, loss_masks)
                    aux_loss_val = (aux_loss_simi + aux_loss_mani) / 2.0
                    loss = main_loss_val + aux_weight * aux_loss_val
                
                val_loss += loss.item()
                
                # Compute metrics
                pixel_acc, mean_iou, recalls = compute_metrics(pred_logits, mc_masks, loss_masks)
                val_metrics['pixel_acc'] += pixel_acc
                val_metrics['mean_iou'] += mean_iou
                for i in range(3):
                    val_metrics['recall'][i] += recalls[i] if not np.isnan(recalls[i]) else 0.0
                val_samples += 1
        
        avg_val_loss = val_loss / max(val_samples, 1)
        val_metrics['pixel_acc'] /= max(val_samples, 1)
        val_metrics['mean_iou'] /= max(val_samples, 1)
        for i in range(3):
            val_metrics['recall'][i] /= max(val_samples, 1)
        
        print(f'Epoch [{epoch+1}/{start_epoch + args.epochs}] Stage {stage} - '
              f'Train Loss: {avg_loss:.4f} (main: {avg_main_loss:.4f}, aux: {avg_aux_loss:.4f}) '
              f'Val Loss: {avg_val_loss:.4f} '
              f'Val Pixel Acc: {val_metrics["pixel_acc"]:.2f}% '
              f'Val mIoU: {val_metrics["mean_iou"]:.2f}% '
              f'Val Recall: [{val_metrics["recall"][0]:.2f}, {val_metrics["recall"][1]:.2f}, {val_metrics["recall"][2]:.2f}]')
        
        # Plateau-based LR reduction and early stopping (paper policy)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            plateau_counter = 0
            lr_reduced = False
            checkpoint_path = os.path.join(args.save_dir, f'best_model_exp{args.experiment}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            print(f'  -> Saved best model to {checkpoint_path}')
        else:
            plateau_counter += 1
            if plateau_counter >= args.patience and not lr_reduced:
                # Reduce LR by half
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f'  -> Validation loss plateaued for {args.patience} epochs. Reducing LR to {optimizer.param_groups[0]["lr"]:.2e}')
                lr_reduced = True
                plateau_counter = 0  # Reset counter after LR reduction
            elif plateau_counter >= args.stop_patience and lr_reduced:
                print(f'  -> Validation loss stopped improving for {args.stop_patience} epochs after LR reduction. Early stopping.')
                break
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}_exp{args.experiment}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    print('Training complete.')

if __name__ == '__main__':
    main()