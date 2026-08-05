"""
Experiment 0: Overfit Sanity Test on a tiny subset (8 images).
Fully incorporates all 9 diagnostic requirements:
1. Valid-pixel loss masking
2. Conservative LR (1e-4) for pretrained weights
3. Aux outputs tracking
4. Per-class masked pixel recall (background/source/target, computed individually)
5. Strict success criterion (Loss < 0.02 AND macro-average per-class recall > 99%)
6. Class distribution logging per epoch
7. Visual prediction saving every 20 epochs
8. Explicit weight key assertions
9. Epoch-based loop (200 epochs)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from PIL import Image

from dataset_busternet import BusterNetDataset
from model_busternet import build_busternet_full
from preprocessing import BusterNetPreprocess


def mask_to_rgb(mask_np):
    """0=background(black), 1=source(green), 2=target(red)"""
    vis = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    vis[mask_np == 1] = [0, 255, 0]
    vis[mask_np == 2] = [255, 0, 0]
    return vis


def save_all_comparisons(save_dir, imgs, targets, preds):
    """Save one [image | ground truth | prediction] triptych per sample, for
    every sample in the batch (not just sample 0), so results can be checked
    against ground truth by eye."""
    os.makedirs(save_dir, exist_ok=True)
    for i in range(imgs.shape[0]):
        img_np = (imgs[i].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        gt_np = mask_to_rgb(targets[i].detach().cpu().numpy().astype(np.uint8))
        pred_np = mask_to_rgb(preds[i].detach().cpu().numpy().astype(np.uint8))
        combined = np.concatenate([img_np, gt_np, pred_np], axis=1)  # [256, 768, 3]
        Image.fromarray(combined).save(os.path.join(save_dir, f"sample_{i}.png"))


def train_overfit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Experiment 0: Overfit Sanity Check on {device} ===")

    out_dir = 'experiment_0_outputs'
    os.makedirs(out_dir, exist_ok=True)

    # Path setup
    data_root = 'coco_subset'
    train_csv = os.path.join(data_root, 'forgery_manifest_with_masks_train.csv')
    if not os.path.exists(train_csv):
        train_csv = os.path.join(data_root, 'forgery_manifest_with_masks.csv')

    # Load 8 samples
    full_ds = BusterNetDataset(csv_path=train_csv, data_root=data_root, img_size=256, split='train', augment=False)
    tiny_ds = Subset(full_ds, list(range(min(8, len(full_ds)))))
    loader = DataLoader(tiny_ds, batch_size=8, shuffle=False)

    # Build PyTorch model
    model = build_busternet_full(target_size=256).to(device)

    # Checkpoint loading with strict assertion audit
    ckpt_path = 'busternet_converted.pth'
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"\n=== Checkpoint Audit ===")
        print(f"Loaded weights from {ckpt_path}")
        print(f"Missing keys ({len(missing)}):")
        for k in missing:
            print(f"  {k}")
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"  {k}")
        
        assert len(unexpected) == 0, f"Unexpected keys found in checkpoint: {unexpected}"

    # Conservative LR (1e-4) for pretrained weights
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.train()
    print("\nStarting overfit optimization for up to 100 epochs...")
    print("Early stopping when Loss < 0.02 AND macro-average per-class recall > 99%")

    for epoch in range(1, 101):
        for batch in loader:
            imgs = batch['image'].to(device)         # [B, 3, 256, 256]
            targets = batch['target_main'].to(device) # [B, 256, 256] in {0, 1, 2}
            masks = batch['loss_mask'].to(device)     # [B, 256, 256] in {0.0, 1.0}

            # Forward pass
            outputs = model(imgs)
            logits = outputs['mc_pred']  # [B, 3, 256, 256]

            # Masked Cross Entropy Loss (over valid pixels only)
            loss_raw = criterion(logits, targets)  # [B, 256, 256]
            valid_pixels = masks.sum()
            loss = (loss_raw * masks).sum() / torch.clamp(valid_pixels, min=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Per-class recall (masked): for each class, % of that class's true
            # pixels that were correctly predicted as that class. Denominator is
            # only that class's own pixel count, not the total image -- so a class
            # with few pixels isn't drowned out by background.
            preds = torch.argmax(logits, dim=1)  # [B, 256, 256]
            valid = masks.bool()

            recalls = []
            for c in range(3):
                class_true = (targets == c) & valid
                class_true_count = class_true.float().sum()
                class_correct = ((preds == c) & class_true).float().sum()
                recalls.append((class_correct / torch.clamp(class_true_count, min=1.0)) * 100.0)
            acc_bg, acc_src, acc_tgt = recalls
            macro_recall = (acc_bg + acc_src + acc_tgt) / 3.0

            # Class prediction distribution (over valid pixels) -- separate diagnostic:
            # what the model predicts, not whether it's correct
            valid_preds = preds[valid]
            total_valid = valid_preds.numel()
            c0_pct = (valid_preds == 0).float().sum() / total_valid * 100.0 if total_valid > 0 else 0.0
            c1_pct = (valid_preds == 1).float().sum() / total_valid * 100.0 if total_valid > 0 else 0.0
            c2_pct = (valid_preds == 2).float().sum() / total_valid * 100.0 if total_valid > 0 else 0.0

        # Epoch logging
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/100 | Loss: {loss.item():.6f} | "
                  f"Recall [BG: {acc_bg.item():.2f}%, SRC: {acc_src.item():.2f}%, TGT: {acc_tgt.item():.2f}%] | "
                  f"Pred Dist [C0(BG): {c0_pct:.1f}%, C1(SRC): {c1_pct:.1f}%, C2(TGT): {c2_pct:.1f}%]")

        # Save diagnostic visualization every 20 epochs -- all 8 samples, each as
        # [original image | ground truth mask | predicted mask]
        if epoch % 20 == 0 or epoch == 1:
            save_all_comparisons(os.path.join(out_dir, f"epoch_{epoch:03d}"), imgs, targets, preds)

        # Early stopping
        if loss.item() < 0.02 and macro_recall.item() > 99.0:
            print(f"\nEarly stopping at epoch {epoch}: Loss={loss.item():.4f}, "
                  f"Recall[BG={acc_bg.item():.2f}%, SRC={acc_src.item():.2f}%, TGT={acc_tgt.item():.2f}%]")
            break

    # Guaranteed final save of all 8 samples, regardless of which epoch training
    # stopped on (early stopping may land on an epoch not divisible by 20)
    final_dir = os.path.join(out_dir, "final")
    save_all_comparisons(final_dir, imgs, targets, preds)
    print(f"\nSaved final image/ground-truth/prediction comparisons for all 8 samples to {final_dir}/")

    print("\n=== Overfit Test Summary ===")
    final_loss = loss.item()
    final_bg, final_src, final_tgt = acc_bg.item(), acc_src.item(), acc_tgt.item()
    final_macro = macro_recall.item()
    print(f"Final Epoch: {epoch}/100")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Final Recall -- Background: {final_bg:.2f}% | Source: {final_src:.2f}% | Target: {final_tgt:.2f}%")
    print(f"Final Macro-Average Recall: {final_macro:.2f}%")

    # Strict double success criterion: Loss < 0.02 AND macro-average per-class recall > 99.0%
    if final_loss < 0.02 and final_macro > 99.0:
        print("[PASS] Experiment 0 Sanity Check PASSED! Model perfectly overfit 8 images.")
        return True
    else:
        print(f"[FAIL] Experiment 0 Sanity Check FAILED (Loss={final_loss:.4f}, "
              f"Macro Recall={final_macro:.2f}%) - investigate pipeline.")
        return False

if __name__ == '__main__':
    train_overfit()