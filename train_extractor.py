"""Train copy-move segmentation extractor using a pretrained ResNet18 backbone.

Replaces the from-scratch 32K-param CMFDExtractor with ResNet18 (truncated
after layer3, ImageNet-pretrained), fine-tuned on the same 15K synthetic
COCO dataset with the same Dice loss / same BusterNet-style decoder.

Rationale (see conversation record):
  - CMFDExtractor (32K params) was likely underpowered — Dice loss barely
    moved over 15 epochs (0.7083 -> 0.6719).
  - ResNet18 was already measured at 0.949 AUC on MICC in-dataset (earlier
    experiment). That failure mode was cross-dataset generalization
    (collapsed on CoMoFoD), which is explicitly out of scope now — only
    MICC-F2000 in-dataset performance matters for this deliverable.
  - Truncating ResNet18 after layer3 gives /16 spatial downsampling
    (256x256 -> 16x16), matching BusterNet's use of VGG16's first four
    blocks for the same reason (16x16 feature map before the decoder).

Architecture
------------
ResNetExtractor  (kept after training)
  torchvision resnet18(pretrained=True), truncated after layer3
  (conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3)
  256x256 input -> (B, 256, 16, 16)
  + 1x1 bottleneck conv: 256 -> 32 channels  (keeps decoder unchanged,
    keeps final feature vector at 32-dim like before)

MaskDecoder  (discarded after training, same as before)
  4 x [BilinearUpPool2D(2x) -> BNInception] -> Conv2d(32,1,1) + sigmoid

Loss: Dice loss vs full-resolution 256x256 binary mask.
Saves: models/resnet_extractor_best.pth  (extractor + bottleneck state dict)
"""
import os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)

IMG_DIR  = 'data/synthetic_coco/images'
MASK_DIR = 'data/synthetic_coco/masks'
MANIFEST = 'data/synthetic_coco/manifest.json'
OUT_PATH = 'models/resnet_extractor_best.pth'

IMG_SIZE   = 256
EPOCHS     = 15
BATCH      = 16
LR_HEAD    = 1e-3     # bottleneck + decoder (randomly init, needs higher LR)
LR_BACKBONE = 1e-4    # pretrained resnet layers (fine-tune gently)
SEED       = 42
WORKERS    = 0
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ building blocks
def _bnconv(in_c, out_c, k, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# ------------------------------------------------------------------ extractor (kept)
class ResNetExtractor(nn.Module):
    """ResNet18 truncated after layer3, + 1x1 bottleneck to 32 channels.

    256x256 input -> (B, 32, 16, 16), matching CMFDExtractor's output shape
    so the rest of the pipeline (decoder, feature pooling) needs no changes.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                     backbone.relu,  backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3          # -> (B, 256, 16, 16) for 256 input
        self.bottleneck = _bnconv(256, 32, 1)  # -> (B, 32, 16, 16)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.bottleneck(x)   # (B, 32, 16, 16)

    def backbone_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3):
            for p in m.parameters():
                yield p

    def head_parameters(self):
        return self.bottleneck.parameters()


# ------------------------------------------------------------------ BN-Inception (decoder only) — unchanged
class BNInception(nn.Module):
    """BusterNet Fig. 3 BN-Inception: 4 parallel branches, each out_c//4 channels."""
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


# ------------------------------------------------------------------ mask decoder (discarded) — unchanged
class MaskDecoder(nn.Module):
    """BusterNet Fig. 3 decoder: 4x BilinearUpPool2D + BNInception -> Conv + sigmoid.
    Input:  (B, 32, 16, 16)   Output: (B, 1, 256, 256)
    """
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


# ------------------------------------------------------------------ full training model
class CMFDModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.extractor = ResNetExtractor(pretrained=pretrained)
        self.decoder   = MaskDecoder()

    def forward(self, x):
        return self.decoder(self.extractor(x))


# ------------------------------------------------------------------ dice loss — unchanged
def dice_loss(pred, target, eps=1e-6):
    p = pred.squeeze(1).float()
    t = target.float()
    inter = (p * t).sum(dim=(1, 2))
    union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


# ------------------------------------------------------------------ dataset — unchanged
class SyntheticDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        img  = Image.open(os.path.join(IMG_DIR,  row['id'] + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(MASK_DIR, row['id'] + '.png')).convert('L')
        img  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img_t  = torch.tensor(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        img_t  = (img_t - self.mean) / self.std
        mask_t = torch.tensor(np.array(mask) > 0, dtype=torch.long)
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
            preds = model(imgs)
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
    samples = manifest['samples']
    train_s = [s for s in samples if s['split'] == 'train']
    val_s   = [s for s in samples if s['split'] == 'test']
    print(f"  train={len(train_s)}  val={len(val_s)}", flush=True)

    train_dl = DataLoader(SyntheticDataset(train_s), batch_size=BATCH,
                          shuffle=True,  num_workers=WORKERS, pin_memory=False)
    val_dl   = DataLoader(SyntheticDataset(val_s),   batch_size=BATCH,
                          shuffle=False, num_workers=WORKERS, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    model = CMFDModel(pretrained=True).to(device)
    backbone_p = sum(p.numel() for p in model.extractor.backbone_parameters())
    head_p     = sum(p.numel() for p in model.extractor.head_parameters())
    dec_p      = sum(p.numel() for p in model.decoder.parameters())
    print(f"Backbone (pretrained) params: {backbone_p:,}  |  "
          f"Bottleneck: {head_p:,}  |  Decoder: {dec_p:,}  |  "
          f"Total: {backbone_p+head_p+dec_p:,}", flush=True)

    # Discriminative LR: fine-tune pretrained backbone gently, train new
    # bottleneck + decoder faster (standard transfer-learning practice)
    opt = torch.optim.Adam([
        {'params': model.extractor.backbone_parameters(), 'lr': LR_BACKBONE},
        {'params': model.extractor.head_parameters(),      'lr': LR_HEAD},
        {'params': model.decoder.parameters(),             'lr': LR_HEAD},
    ])

    best_val, best_epoch = float('inf'), -1
    print(f"\n{'Epoch':>6}  {'Train Dice':>12}  {'Val Dice':>10}  {'Saved':>6}")
    print('-' * 46, flush=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_dl, opt,  device)
        va_loss = run_epoch(model, val_dl,   None, device)

        saved = ''
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
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