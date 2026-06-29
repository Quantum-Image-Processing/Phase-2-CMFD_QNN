"""Train copy-move segmentation extractor + BusterNet Mask Decoder on synthetic COCO.

Architecture
------------
CMFDExtractor  (kept after training)
  5 conv blocks, stride-2 × 4 then stride-1 × 1  →  (B, 32, 16, 16)
  channels: 3 → 16 → 32 → 32 → 32 → 32

MaskDecoder  (discarded after training)
  4 × [BilinearUpPool2D(2×) → BNInception] → Conv2d(32,1,1) + sigmoid
  16 → 32 → 64 → 128 → 256 (spatial), constant 32 channels throughout

Loss: Dice loss vs full-resolution 256×256 binary mask.
Saves: models/extractor_best.pth  (extractor state dict only)
"""
import os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)

IMG_DIR  = 'data/synthetic_coco/images'
MASK_DIR = 'data/synthetic_coco/masks'
MANIFEST = 'data/synthetic_coco/manifest.json'
OUT_PATH = 'models/extractor_best.pth'

IMG_SIZE = 256
EPOCHS   = 15
BATCH    = 16
LR       = 1e-3
SEED     = 42
WORKERS  = 0
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ building blocks
def _bnconv(in_c, out_c, k, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# ------------------------------------------------------------------ extractor (kept)
class CMFDExtractor(nn.Module):
    """5-block lightweight extractor → (B, 32, 16, 16) for 256×256 input."""
    def __init__(self):
        super().__init__()
        self.block1 = _bnconv(3,  16, 3, stride=2, pad=1)   # 256→128
        self.block2 = _bnconv(16, 32, 3, stride=2, pad=1)   # 128→64
        self.block3 = _bnconv(32, 32, 3, stride=2, pad=1)   #  64→32
        self.block4 = _bnconv(32, 32, 3, stride=2, pad=1)   #  32→16
        self.block5 = _bnconv(32, 32, 3, stride=1, pad=1)   #  16→16

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.block5(x)   # (B, 32, 16, 16)


# ------------------------------------------------------------------ BN-Inception (decoder only)
class BNInception(nn.Module):
    """BusterNet Fig. 3 BN-Inception: 4 parallel branches, each producing out_c//4 channels.

    Branch 1: 1×1
    Branch 2: 1×1 → 3×3
    Branch 3: 1×1 → 3×3 → 3×3  (factorized 5×5)
    Branch 4: MaxPool(3×3) → 1×1
    Output channels = out_c  (must be divisible by 4)
    """
    def __init__(self, in_c, out_c=32):
        super().__init__()
        q = out_c // 4
        self.b1 = _bnconv(in_c, q, 1)
        self.b2 = nn.Sequential(_bnconv(in_c, q, 1),
                                 _bnconv(q,    q, 3, pad=1))
        self.b3 = nn.Sequential(_bnconv(in_c, q, 1),
                                 _bnconv(q,    q, 3, pad=1),
                                 _bnconv(q,    q, 3, pad=1))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                 _bnconv(in_c, q, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


# ------------------------------------------------------------------ mask decoder (discarded)
class MaskDecoder(nn.Module):
    """BusterNet Fig. 3 decoder: 4× BilinearUpPool2D + BNInception → Conv + sigmoid.

    Input:  (B, 32, 16, 16)
    Output: (B,  1, 256, 256)
    """
    def __init__(self):
        super().__init__()
        # 4 upsample stages; spatial: 16→32→64→128→256, channels fixed at 32
        self.stage1 = BNInception(32, 32)   # 16→32 spatial after up
        self.stage2 = BNInception(32, 32)   # 32→64
        self.stage3 = BNInception(32, 32)   # 64→128
        self.stage4 = BNInception(32, 32)   # 128→256
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
        return torch.sigmoid(self.head(x))   # (B, 1, 256, 256)


# ------------------------------------------------------------------ full training model
class CMFDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = CMFDExtractor()
        self.decoder   = MaskDecoder()

    def forward(self, x):
        return self.decoder(self.extractor(x))   # (B, 1, 256, 256)


# ------------------------------------------------------------------ dice loss
def dice_loss(pred, target, eps=1e-6):
    """pred: (B,1,H,W) in [0,1]; target: (B,H,W) long binary."""
    p = pred.squeeze(1).float()
    t = target.float()
    inter = (p * t).sum(dim=(1, 2))
    union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


# ------------------------------------------------------------------ dataset
class SyntheticDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.mean    = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std     = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        img  = Image.open(os.path.join(IMG_DIR,  row['id'] + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(MASK_DIR, row['id'] + '.png')).convert('L')

        # Both resized to 256×256 for end-to-end pixel-level training
        img  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        img_t  = torch.tensor(np.array(img,  dtype=np.float32)).permute(2, 0, 1) / 255.0
        img_t  = (img_t - self.mean) / self.std
        mask_t = torch.tensor(np.array(mask) > 0, dtype=torch.long)   # (256, 256)
        return img_t, mask_t


# ------------------------------------------------------------------ epoch runner
def run_epoch(model, loader, opt, device):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)              # (B, 1, 256, 256)
            loss  = dice_loss(preds, masks)
            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(imgs)
            n += len(imgs)
    return total_loss / n


# ------------------------------------------------------------------ main
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs('models', exist_ok=True)

    print("Loading manifest ...", flush=True)
    with open(MANIFEST) as f:
        manifest = json.load(f)
    samples  = manifest['samples']
    train_s  = [s for s in samples if s['split'] == 'train']
    val_s    = [s for s in samples if s['split'] == 'test']
    print(f"  train={len(train_s)}  val={len(val_s)}", flush=True)

    train_dl = DataLoader(SyntheticDataset(train_s), batch_size=BATCH,
                          shuffle=True,  num_workers=WORKERS, pin_memory=False)
    val_dl   = DataLoader(SyntheticDataset(val_s),   batch_size=BATCH,
                          shuffle=False, num_workers=WORKERS, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    model = CMFDModel().to(device)
    ext_p  = sum(p.numel() for p in model.extractor.parameters())
    dec_p  = sum(p.numel() for p in model.decoder.parameters())
    print(f"Extractor params: {ext_p:,}  |  Decoder params: {dec_p:,}  |  Total: {ext_p+dec_p:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val   = float('inf')
    best_epoch = -1

    print(f"\n{'Epoch':>6}  {'Train Dice':>12}  {'Val Dice':>10}  {'Saved':>6}")
    print('-' * 46, flush=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_dl, opt,  device)
        va_loss = run_epoch(model, val_dl,   None, device)

        saved = ''
        if va_loss < best_val:
            best_val   = va_loss
            best_epoch = epoch
            # Save extractor weights only — decoder is discarded
            torch.save({'epoch': epoch,
                        'val_dice_loss': best_val,
                        'extractor_state': model.extractor.state_dict()},
                       OUT_PATH)
            saved = '*'

        print(f"{epoch:>6}  {tr_loss:>12.6f}  {va_loss:>10.6f}  {saved:>6}", flush=True)

    print(f"\nBest epoch: {best_epoch}  val Dice loss: {best_val:.6f}", flush=True)
    print(f"Extractor weights saved to {OUT_PATH}  (decoder discarded)", flush=True)


if __name__ == '__main__':
    main()
