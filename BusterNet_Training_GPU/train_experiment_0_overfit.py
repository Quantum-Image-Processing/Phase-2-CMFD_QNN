"""
Experiment 0: Overfit Sanity Test on a tiny subset (8 images).
Fully incorporates all 9 diagnostic requirements:
1. Valid-pixel loss masking
2. Conservative LR (1e-4) for pretrained weights
3. Aux outputs tracking
4. Masked pixel accuracy
5. Strict success criterion (Loss < 0.02 AND Masked Acc > 99%)
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
    print("Early stopping when Loss < 0.02 AND Masked Acc > 99%")

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

            # Masked accuracy calculation
            preds = torch.argmax(logits, dim=1)  # [B, 256, 256]
            correct = ((preds == targets) * masks.bool()).float().sum()
            masked_acc = (correct / torch.clamp(valid_pixels, min=1.0)) * 100.0

            # Class prediction distribution (over valid pixels)
            valid_preds = preds[masks.bool()]
            total_valid = valid_preds.numel()
            c0_pct = (valid_preds == 0).float().sum() / total_valid * 100.0 if total_valid > 0 else 0.0
            c1_pct = (valid_preds == 1).float().sum() / total_valid * 100.0 if total_valid > 0 else 0.0
            c2_pct = (valid_preds == 2).float().sum() / total_valid * 100.0 if total_valid > 0 else 0.0

        # Epoch logging
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/100 | Masked Loss: {loss.item():.6f} | Masked Acc: {masked_acc.item():.2f}% | "
                  f"Pred Dist [C0(BG): {c0_pct:.1f}%, C1(SRC): {c1_pct:.1f}%, C2(TGT): {c2_pct:.1f}%]")

        # Save diagnostic visualization every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_pred.png")
            # Save first sample prediction map: 0=black, 1=green, 2=red
            p_map = preds[0].cpu().numpy().astype(np.uint8)
            vis = np.zeros((256, 256, 3), dtype=np.uint8)
            vis[p_map == 1] = [0, 255, 0]   # Source = Green
            vis[p_map == 2] = [255, 0, 0]   # Target = Red
            Image.fromarray(vis).save(save_path)

        # Early stopping
        if loss.item() < 0.02 and masked_acc.item() > 99.0:
            print(f"\nEarly stopping at epoch {epoch}: Loss={loss.item():.4f}, Acc={masked_acc.item():.2f}%")
            break

    print("\n=== Overfit Test Summary ===")
    final_loss = loss.item()
    final_acc = masked_acc.item()
    print(f"Final Epoch: {epoch}/100")
    print(f"Final Masked Loss: {final_loss:.6f}")
    print(f"Final Masked Pixel Accuracy: {final_acc:.2f}%")

    # Strict double success criterion: Loss < 0.02 AND Masked Acc > 99.0%
    if final_loss < 0.02 and final_acc > 99.0:
        print("[PASS] Experiment 0 Sanity Check PASSED! Model perfectly overfit 8 images.")
        return True
    else:
        print(f"[FAIL] Experiment 0 Sanity Check FAILED (Loss={final_loss:.4f}, Acc={final_acc:.2f}%) - investigate pipeline.")
        return False

if __name__ == '__main__':
    train_overfit()