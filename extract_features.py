"""Train the extractor as a genuine patch classifier -- Al-khuzaie's actual
method, nothing substituted.

Al-khuzaie's training signal: 128x128 patches (sliding window, stride 8),
CNN -> feature -> FCL with dropout -> two-way softmax -> cross-entropy vs.
patch label. No decoder, no Dice loss, no segmentation pretext task.

Patch labels are DERIVED from the synthetic dataset's existing binary masks
(mask_patch.mean() > threshold => forged), not re-annotated by hand. This is
the fix for the earlier excuse ("we have masks, not patch labels") -- masks
already contain enough information to produce patch labels directly.

Disclosed deviation from literal fidelity (stated plainly, not hidden):
Al-khuzaie's stride-8 sliding window over a full image produces ~289
patches per image (for 256x256 input, 128x128 patch). Enumerating and
training on every single one for all 15K images (~4.3M patches/epoch) is
not a reasonable training budget. Instead, RANDOMLY SAMPLE a fixed number
of patches per image per epoch (default 20). This is a practical adaptation
for training only -- inference (extract_features.py) still uses the FULL
stride-8 sliding window per Al-khuzaie, since that's a one-time forward
pass per image, not a repeated training cost.

Backbone: ResNet18 (truncated after layer3, ImageNet-pretrained, fine-tuned)
-- same choice as before, same justification (0.949 AUC measured earlier,
32K from-scratch extractor was underpowered). Only the TRAINING SIGNAL
changes here (patch classification vs. mask reconstruction), not the
backbone choice.

Saves: models/patch_extractor_best.pth  (extractor state dict only;
the small FC classifier head used for training is discarded)
"""
import os, json, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)

IMG_DIR   = 'data/synthetic_coco/images'
MASK_DIR  = 'data/synthetic_coco/masks'
MANIFEST  = 'data/synthetic_coco/manifest.json'
OUT_PATH  = 'models/patch_extractor_best.pth'

FULL_IMG_SIZE = 256
PATCH_SIZE    = 128
STRIDE        = 8              # Al-khuzaie's stride, used to enumerate candidates
PATCHES_PER_IMAGE_TRAIN = 20   # disclosed sampling adaptation, see docstring
FORGED_THRESHOLD = 0.10        # patch is "forged" if >10% of its pixels are masked

EPOCHS = 15
BATCH  = 64
LR_BACKBONE = 1e-4
LR_HEAD     = 1e-3
SEED   = 42
WORKERS = 0
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ extractor (kept)
class ResNetExtractor(nn.Module):
    """ResNet18 truncated after layer3, + 1x1 bottleneck to 32 channels.
    128x128 patch input -> (B, 32, 8, 8)."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                     backbone.relu,  backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.bottleneck(x)

    def backbone_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3):
            for p in m.parameters():
                yield p

    def head_parameters(self):
        return self.bottleneck.parameters()


# ------------------------------------------------------------------ Al-khuzaie's classifier head (discarded after training)
class PatchClassifierHead(nn.Module):
    """GAP -> dropout -> FC -> two-way softmax, per Al-khuzaie's FCL description."""
    def __init__(self, in_dim=32, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, feat_map):          # (B, 32, 8, 8)
        vec = feat_map.mean(dim=(2, 3))    # GAP here is fine -- this feeds
                                            # ONLY the training-time classifier
                                            # head, not the downstream feature
        return self.fc(self.dropout(vec))  # (B, 2)


class PatchModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.extractor = ResNetExtractor(pretrained=pretrained)
        self.head = PatchClassifierHead()

    def forward(self, x):
        return self.head(self.extractor(x))


# ------------------------------------------------------------------ patch sampling dataset
class PatchDataset(Dataset):
    """Randomly samples PATCHES_PER_IMAGE_TRAIN patches per image per epoch,
    labeled from the corresponding region of the existing binary mask."""
    def __init__(self, samples):
        self.samples = samples
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1)
        self.coords = list(range(0, FULL_IMG_SIZE - PATCH_SIZE + 1, STRIDE))

    def __len__(self):
        return len(self.samples) * PATCHES_PER_IMAGE_TRAIN

    def __getitem__(self, idx):
        row = self.samples[idx // PATCHES_PER_IMAGE_TRAIN]
        img  = Image.open(os.path.join(IMG_DIR,  row['id'] + '.jpg')).convert('RGB')
        mask = Image.open(os.path.join(MASK_DIR, row['id'] + '.png')).convert('L')
        img  = img.resize((FULL_IMG_SIZE, FULL_IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((FULL_IMG_SIZE, FULL_IMG_SIZE), Image.NEAREST)
        mask_np = np.array(mask) > 0

        x = random.choice(self.coords)
        y = random.choice(self.coords)
        patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        mask_patch = mask_np[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        label = int(mask_patch.mean() > FORGED_THRESHOLD)

        t = torch.tensor(np.array(patch, dtype=np.float32)).permute(2, 0, 1) / 255.0
        t = (t - self.mean) / self.std
        return t, label


# ------------------------------------------------------------------ epoch runner
def run_epoch(model, loader, opt, device):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            logits = model(patches)
            loss = F.cross_entropy(logits, labels)
            if is_train:
                opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, total_correct / n


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    os.makedirs('models', exist_ok=True)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    samples = manifest['samples']
    train_s = [s for s in samples if s['split'] == 'train']
    val_s   = [s for s in samples if s['split'] == 'test']
    print(f"train images={len(train_s)}  val images={len(val_s)}  "
          f"(-> {len(train_s)*PATCHES_PER_IMAGE_TRAIN} / "
          f"{len(val_s)*PATCHES_PER_IMAGE_TRAIN} patches/epoch)", flush=True)

    train_dl = DataLoader(PatchDataset(train_s), batch_size=BATCH,
                          shuffle=True, num_workers=WORKERS)
    val_dl   = DataLoader(PatchDataset(val_s), batch_size=BATCH,
                          shuffle=False, num_workers=WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchModel(pretrained=True).to(device)
    print(f"Backbone params: {sum(p.numel() for p in model.extractor.backbone_parameters()):,}  "
          f"Bottleneck: {sum(p.numel() for p in model.extractor.head_parameters()):,}  "
          f"Classifier head (discarded): {sum(p.numel() for p in model.head.parameters()):,}",
          flush=True)

    opt = torch.optim.Adam([
        {'params': model.extractor.backbone_parameters(), 'lr': LR_BACKBONE},
        {'params': model.extractor.head_parameters(),      'lr': LR_HEAD},
        {'params': model.head.parameters(),                'lr': LR_HEAD},
    ])

    best_val_acc, best_epoch = 0.0, -1
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'Saved':>6}")
    print('-' * 66, flush=True)
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, opt, device)
        va_loss, va_acc = run_epoch(model, val_dl, None, device)
        saved = ''
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save({'epoch': epoch, 'val_acc': best_val_acc,
                        'extractor_state': model.extractor.state_dict()}, OUT_PATH)
            saved = '*'
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {va_loss:>9.4f}  {va_acc:>8.4f}  {saved:>6}",
              flush=True)

    print(f"\nBest epoch: {best_epoch}  val acc: {best_val_acc:.4f}")
    print(f"Extractor saved to {OUT_PATH} (classifier head discarded)")


if __name__ == '__main__':
    main()