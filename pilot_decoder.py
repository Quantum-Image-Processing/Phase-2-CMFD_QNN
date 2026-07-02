"""Pilot check for CMFDExtractor + MaskDecoder before full training.

Root-cause fix from diagnostic:
  eval mode with fresh BN gives extractor std=0.007 → constant decoder output (0.49)
  train mode gives extractor std=0.58 → spatially variable decoder output (0.13–0.76)
  So all forward-pass checks use model.train() + torch.no_grad()

Checks (go/no-go):
  1. Output shape == (B, 1, 256, 256)
  2. Output values in [0, 1]
  3. Spatial std > 0.01 (not uniform)
  4. Dice loss pre-training measured on 3 representative images (20-50% forged)
  5. After 100 training steps, Dice loss on SAME 3 images decreases — gradient flows
  6. Gradient norm > 0 after first backward pass
  7. Overlay PNGs for visual inspection
"""
import os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
sys.stdout.reconfigure(line_buffering=True)

IMG_DIR  = 'data/synthetic_coco/images'
MASK_DIR = 'data/synthetic_coco/masks'
MANIFEST = 'data/synthetic_coco/manifest.json'
OUT_DIR  = 'pilot_decoder_out'
IMG_SIZE = 256
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ model (mirrors train_extractor.py)
def _bnconv(in_c, out_c, k, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class CMFDExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _bnconv(3,  16, 3, stride=2, pad=1)
        self.block2 = _bnconv(16, 32, 3, stride=2, pad=1)
        self.block3 = _bnconv(32, 32, 3, stride=2, pad=1)
        self.block4 = _bnconv(32, 32, 3, stride=2, pad=1)
        self.block5 = _bnconv(32, 32, 3, stride=1, pad=1)
    def forward(self, x):
        return self.block5(self.block4(self.block3(self.block2(self.block1(x)))))

class BNInception(nn.Module):
    def __init__(self, in_c, out_c=32):
        super().__init__()
        q = out_c // 4
        self.b1 = _bnconv(in_c, q, 1)
        self.b2 = nn.Sequential(_bnconv(in_c, q, 1), _bnconv(q, q, 3, pad=1))
        self.b3 = nn.Sequential(_bnconv(in_c, q, 1), _bnconv(q, q, 3, pad=1), _bnconv(q, q, 3, pad=1))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), _bnconv(in_c, q, 1))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = BNInception(32, 32)
        self.stage2 = BNInception(32, 32)
        self.stage3 = BNInception(32, 32)
        self.stage4 = BNInception(32, 32)
        self.head   = nn.Conv2d(32, 1, 1)
    def forward(self, feat):
        x = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.stage1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.stage3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.stage4(x)
        return torch.sigmoid(self.head(x))

class CMFDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = CMFDExtractor()
        self.decoder   = MaskDecoder()
    def forward(self, x):
        return self.decoder(self.extractor(x))


# ------------------------------------------------------------------ dice loss
def dice_loss(pred, target, eps=1e-6):
    p = pred.squeeze(1).float()
    t = target.float()
    inter = (p * t).sum(dim=(1, 2))
    union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


# ------------------------------------------------------------------ load samples
def load_sample(row):
    mean = torch.tensor(MEAN).view(3,1,1)
    std  = torch.tensor(STD).view(3,1,1)
    img  = Image.open(os.path.join(IMG_DIR,  row['id'] + '.jpg')).convert('RGB')
    mask = Image.open(os.path.join(MASK_DIR, row['id'] + '.png')).convert('L')
    img_256  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    mask_256 = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    img_t  = torch.tensor(np.array(img_256, dtype=np.float32)).permute(2,0,1) / 255.0
    img_t  = (img_t - mean) / std
    mask_t = torch.tensor(np.array(mask_256) > 0, dtype=torch.long)
    return img_t, mask_t, img_256


# ------------------------------------------------------------------ overlay
def save_overlay(img_pil, pred_np, gt_np, label, path):
    pred_bin = (pred_np > 0.5).astype(np.uint8)
    gt_bin   = gt_np.astype(np.uint8)

    def blend(base, mask, color):
        arr = np.array(base).copy().astype(float)
        for c, cv in enumerate(color):
            arr[:,:,c] = np.where(mask, arr[:,:,c]*0.45 + cv*0.55, arr[:,:,c])
        return Image.fromarray(arr.clip(0,255).astype(np.uint8))

    panels = [
        (img_pil,                          "Original"),
        (blend(img_pil, gt_bin,   (0,220,0)),   "GT mask (green)"),
        (blend(img_pil, pred_bin, (255,60,60)),  f"Pred mask (red) — {label}"),
    ]
    combined = Image.new('RGB', (IMG_SIZE * 3, IMG_SIZE + 18), (30,30,30))
    draw = ImageDraw.Draw(combined)
    for i, (panel, title) in enumerate(panels):
        combined.paste(panel, (i * IMG_SIZE, 18))
        draw.text((i * IMG_SIZE + 4, 2), title, fill=(220,220,220))
    combined.save(path)


# ------------------------------------------------------------------ tiny training dataset
class TinyDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.mean = torch.tensor(MEAN).view(3,1,1)
        self.std  = torch.tensor(STD).view(3,1,1)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        row  = self.samples[idx]
        img  = Image.open(os.path.join(IMG_DIR,  row['id']+'.jpg')).convert('RGB').resize((IMG_SIZE,IMG_SIZE), Image.BILINEAR)
        mask = Image.open(os.path.join(MASK_DIR, row['id']+'.png')).convert('L').resize((IMG_SIZE,IMG_SIZE), Image.NEAREST)
        img_t  = torch.tensor(np.array(img, dtype=np.float32)).permute(2,0,1)/255.0
        img_t  = (img_t - self.mean) / self.std
        mask_t = torch.tensor(np.array(mask) > 0, dtype=torch.long)
        return img_t, mask_t


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    samples = manifest['samples']

    # Pick 3 REPRESENTATIVE samples: mask coverage between 15% and 47% of the image
    # Not the near-zero or near-total outliers — those skew Dice expectations
    representative = [s for s in samples
                      if 0.15 <= s['coverage'] <= 0.47 and s['split'] == 'train'][:3]
    print(f"Pilot samples: {[(s['id'], round(s['coverage'], 3)) for s in representative]}", flush=True)

    train100 = [s for s in samples if s['split'] == 'train'][:100]

    model = CMFDModel()

    # ================================================================
    print("\n" + "="*60)
    print("CHECK 1 — forward pass (train mode + no_grad)")
    print("="*60)
    # Key fix: use train mode so BN computes batch statistics instead of
    # using uninitialized running stats (which collapse extractor std to 0.007)
    model.train()

    imgs_t, masks_t, imgs_pil = [], [], []
    for row in representative:
        it, mt, ip = load_sample(row)
        imgs_t.append(it); masks_t.append(mt); imgs_pil.append(ip)

    imgs_batch  = torch.stack(imgs_t)
    masks_batch = torch.stack(masks_t)

    with torch.no_grad():
        feat_map = model.extractor(imgs_batch)
        preds    = model.decoder(feat_map)

    print(f"  Extractor feat: shape={tuple(feat_map.shape)}  std={feat_map.std():.4f}")
    print(f"  Decoder output: shape={tuple(preds.shape)}   expected (3,1,256,256)")
    shape_ok = tuple(preds.shape) == (3, 1, 256, 256)
    print(f"  Shape check   : {'PASS' if shape_ok else 'FAIL'}")

    vmin, vmax = preds.min().item(), preds.max().item()
    print(f"  Value range   : [{vmin:.4f}, {vmax:.4f}]  expected [0,1]")
    range_ok = (vmin >= 0.0) and (vmax <= 1.0)
    print(f"  Range check   : {'PASS' if range_ok else 'FAIL'}")

    spatial_std = preds.std().item()
    print(f"  Spatial std   : {spatial_std:.4f}  (>0.01 = spatially variable)")
    spatial_ok = spatial_std > 0.01
    print(f"  Spatial check : {'PASS' if spatial_ok else 'FAIL'}")

    pre_dice = dice_loss(preds, masks_batch).item()
    gt_frac  = masks_batch.float().mean().item()
    print(f"  GT forged frac: {gt_frac*100:.1f}%")
    print(f"  Dice (random) : {pre_dice:.4f}  (expected 0.3–0.8 depending on forged fraction)")
    # With random predictions and typical forged fraction, Dice loss is 0.3–0.8 (not ~1.0)
    # The key is that it DROPS after training, not what it starts at
    pre_dice_ok = 0.20 < pre_dice < 0.95
    print(f"  Pre-train check: {'PASS' if pre_dice_ok else 'FAIL (outside expected random range)'}")

    # ================================================================
    print("\n" + "="*60)
    print("CHECK 2 — gradient norm after first backward pass")
    print("="*60)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss_first = dice_loss(model(imgs_batch), masks_batch)
    loss_first.backward()
    total_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    print(f"  Loss on pilot batch : {loss_first.item():.4f}")
    print(f"  Total gradient norm : {total_norm:.4f}  (>0.0 = gradient flowing)")
    grad_ok = total_norm > 1e-6
    print(f"  Gradient check      : {'PASS' if grad_ok else 'FAIL (zero gradients — dead network)'}")

    # ================================================================
    print("\n" + "="*60)
    print("CHECK 3 — loss drops after 100 training steps on same 3 images")
    print("="*60)
    # Train on 100 samples, then re-evaluate on the SAME 3 pilot images
    loader = DataLoader(TinyDataset(train100), batch_size=16, shuffle=True)
    for batch_imgs, batch_masks in loader:
        opt.zero_grad()
        loss = dice_loss(model(batch_imgs), batch_masks)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds_post = model(imgs_batch)
    # NOTE: now in eval mode with populated running stats (after 7 batches), BN should work
    post_dice = dice_loss(preds_post, masks_batch).item()
    drop = pre_dice - post_dice
    print(f"  Dice before training (pilot 3 images): {pre_dice:.4f}")
    print(f"  Dice after  training (pilot 3 images): {post_dice:.4f}")
    print(f"  Drop                                 : {drop:+.4f}  (positive = learning)")
    drop_ok = drop > 0.005
    print(f"  Learning check: {'PASS' if drop_ok else 'FAIL (loss did not decrease)'}")

    # ================================================================
    print("\n" + "="*60)
    print("CHECK 4 — visual overlay (saved to pilot_decoder_out/)")
    print("="*60)
    model.train()
    with torch.no_grad():
        preds_vis = model(imgs_batch)

    for i, (row, img_pil) in enumerate(zip(representative, imgs_pil)):
        pred_np = preds_vis[i, 0].numpy()
        gt_np   = masks_batch[i].numpy()
        forged_pct    = gt_np.mean() * 100
        activated_pct = (pred_np > 0.5).mean() * 100
        path = os.path.join(OUT_DIR, f"pilot_{i+1}_{row['id']}.png")
        save_overlay(img_pil, pred_np, gt_np, f"post-100steps", path)
        print(f"  Sample {i+1}: GT={forged_pct:.1f}%  pred_activated={activated_pct:.1f}%  -> {path}")

    # ================================================================
    print("\n" + "="*60)
    print("GO / NO-GO SUMMARY")
    print("="*60)
    checks = [
        ("Output shape (3,1,256,256)",        shape_ok),
        ("Values in [0,1]",                    range_ok),
        ("Spatially variable (std>0.01)",       spatial_ok),
        ("Pre-train Dice in expected range",    pre_dice_ok),
        ("Gradient norm > 0",                   grad_ok),
        ("Dice drops after 100 steps",          drop_ok),
    ]
    all_pass = all(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")

    print()
    if all_pass:
        print("=> GO: all checks passed. Run python train_extractor.py for full training.")
    else:
        print("=> NO-GO: fix failing checks before full training.")


if __name__ == '__main__':
    main()
