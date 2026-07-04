"""Train the extractor as a genuine patch classifier -- Al-khuzaie's actual
method, FULL FIDELITY, no patch subsampling.

Every single stride-8, 128x128 patch from every training image is used,
every epoch. No shortcuts.

Patch labels derived from existing synthetic masks:
    label = int(mask_patch.mean() > FORGED_THRESHOLD)

Scale, stated plainly:
    256x256 image, 128x128 patch, stride 8 -> 17 positions/axis -> 289
    patches/image. 15,000 images -> ~4,340,000 patches/epoch.

This is loaded via an IterableDataset that opens each image ONCE and yields
all 289 of its patches before moving to the next image -- avoids re-decoding
the same JPEG 289 times, which would be the dominant cost otherwise. Patches
within an image are shuffled before yielding; image order is shuffled each
epoch. This is not identical to a global shuffle across all 4.3M patches,
but is the standard, disclosed compromise for iterable/streaming datasets
at this scale -- fully global shuffling would require holding the whole
patch set in memory or a shuffle buffer large enough to defeat the purpose.

Checkpointing: saves model+optimizer state after every N_BATCHES_PER_CKPT
batches (mid-epoch) AND after every epoch, so an interrupted run resumes
from the last checkpoint instead of restarting. Checkpoints under
checkpoints/, best-validation model under models/.

On epoch count: see the note printed at the end of training -- this script
does NOT hardcode a "right" number and picks it by early stopping on
validation accuracy instead of guessing.
"""
import os, json, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)

IMG_DIR   = 'data/synthetic_coco/images'
MASK_DIR  = 'data/synthetic_coco/masks'
MANIFEST  = 'data/synthetic_coco/manifest.json'
CKPT_DIR  = 'checkpoints'
OUT_PATH  = 'models/patch_extractor_best.pth'

FULL_IMG_SIZE = 256
PATCH_SIZE    = 128
STRIDE        = 8
FORGED_THRESHOLD = 0.10

MAX_EPOCHS   = 50          # upper bound; early stopping decides actual count
PATIENCE     = 5           # stop if val acc doesn't improve for this many epochs
BATCH        = 128
LR_BACKBONE  = 1e-4
LR_HEAD      = 1e-3
SEED         = 42
WORKERS      = 4
N_BATCHES_PER_CKPT = 2000  # mid-epoch checkpoint frequency

# Wall-clock session limit -- lets training run in bounded stretches (e.g. a
# Colab session) and resume cleanly next time instead of being killed
# mid-batch. Override via env var TRAIN_TIME_LIMIT_SEC; set to 0 to disable.
TIME_LIMIT_SEC = int(os.environ.get('TRAIN_TIME_LIMIT_SEC', 3 * 3600))

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class TimeLimitReached(Exception):
    pass


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


class PatchClassifierHead(nn.Module):
    """GAP -> dropout -> FC -> two-way softmax, per Al-khuzaie's FCL description."""
    def __init__(self, in_dim=32, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, feat_map):
        vec = feat_map.mean(dim=(2, 3))
        return self.fc(self.dropout(vec))


class PatchModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.extractor = ResNetExtractor(pretrained=pretrained)
        self.head = PatchClassifierHead()

    def forward(self, x):
        return self.head(self.extractor(x))


# ------------------------------------------------------------------ FULL patch enumeration, streaming
class FullPatchIterableDataset(IterableDataset):
    """Yields EVERY stride-8 patch from EVERY image, no subsampling.
    Opens each image once, yields all 289 of its patches (shuffled),
    then moves on -- avoids re-decoding the same image 289 times."""
    def __init__(self, samples, shuffle=True, seed=SEED):
        self.samples = samples
        self.shuffle = shuffle
        self.seed = seed
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1)
        self.coords = list(range(0, FULL_IMG_SIZE - PATCH_SIZE + 1, STRIDE))
        self.n_patches_per_img = len(self.coords) ** 2

    def __len__(self):
        return len(self.samples) * self.n_patches_per_img

    def _iter_images(self, epoch_seed):
        info = get_worker_info()
        samples = self.samples
        if self.shuffle:
            rng = random.Random(epoch_seed)
            samples = samples.copy()
            rng.shuffle(samples)
        if info is not None:  # split images across workers, not patches
            samples = samples[info.id::info.num_workers]
        return samples

    def __iter__(self):
        epoch_seed = self.seed + int(time.time()) if self.shuffle else self.seed
        for row in self._iter_images(epoch_seed):
            img  = Image.open(os.path.join(IMG_DIR,  row['id'] + '.jpg')).convert('RGB')
            mask = Image.open(os.path.join(MASK_DIR, row['id'] + '.png')).convert('L')
            img  = img.resize((FULL_IMG_SIZE, FULL_IMG_SIZE), Image.BILINEAR)
            mask = mask.resize((FULL_IMG_SIZE, FULL_IMG_SIZE), Image.NEAREST)
            mask_np = np.array(mask) > 0

            coord_pairs = [(x, y) for y in self.coords for x in self.coords]
            if self.shuffle:
                random.shuffle(coord_pairs)

            for x, y in coord_pairs:
                patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                mask_patch = mask_np[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                label = int(mask_patch.mean() > FORGED_THRESHOLD)
                t = torch.tensor(np.array(patch, dtype=np.float32)).permute(2, 0, 1) / 255.0
                t = (t - self.mean) / self.std
                yield t, label


# ------------------------------------------------------------------ checkpoint helpers
def save_checkpoint(path, model, opt, epoch, batch_idx, best_val_acc, epochs_no_improve=0):
    torch.save({
        'epoch': epoch, 'batch_idx': batch_idx, 'best_val_acc': best_val_acc,
        'epochs_no_improve': epochs_no_improve,
        'model_state': model.state_dict(), 'opt_state': opt.state_dict(),
    }, path)


def load_checkpoint(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    opt.load_state_dict(ckpt['opt_state'])
    return (ckpt['epoch'], ckpt['batch_idx'], ckpt['best_val_acc'],
            ckpt.get('epochs_no_improve', 0))


# ------------------------------------------------------------------ epoch runners
def train_one_epoch(model, loader, opt, device, epoch, n_total_batches,
                     session_start, best_val_acc, epochs_no_improve):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    t0 = time.time()
    for batch_idx, (patches, labels) in enumerate(loader, 1):
        patches, labels = patches.to(device), labels.to(device)
        logits = model(patches)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)

        if batch_idx % 200 == 0:
            elapsed = time.time() - t0
            pct = 100 * batch_idx / n_total_batches
            print(f"  epoch {epoch}  batch {batch_idx}/{n_total_batches} ({pct:.1f}%)  "
                  f"running loss={total_loss/n:.4f}  acc={total_correct/n:.4f}  "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

        if batch_idx % N_BATCHES_PER_CKPT == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            save_checkpoint(os.path.join(CKPT_DIR, 'midepoch.pth'),
                             model, opt, epoch, batch_idx, best_val_acc, epochs_no_improve)
            print(f"  [checkpoint saved at batch {batch_idx}]", flush=True)

        if TIME_LIMIT_SEC and (time.time() - session_start) > TIME_LIMIT_SEC:
            os.makedirs(CKPT_DIR, exist_ok=True)
            save_checkpoint(os.path.join(CKPT_DIR, 'midepoch.pth'),
                             model, opt, epoch, batch_idx, best_val_acc, epochs_no_improve)
            print(f"\n[time limit {TIME_LIMIT_SEC/3600:.1f}h reached -- checkpoint saved "
                  f"at epoch {epoch} batch {batch_idx}. Re-run the same command to continue.]",
                  flush=True)
            raise TimeLimitReached()

    return total_loss / n, total_correct / n


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    with torch.no_grad():
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            logits = model(patches)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, total_correct / n


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    os.makedirs('models', exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    samples = manifest['samples']
    train_s = [s for s in samples if s['split'] == 'train']
    val_s   = [s for s in samples if s['split'] == 'test']

    train_ds = FullPatchIterableDataset(train_s, shuffle=True)
    val_ds   = FullPatchIterableDataset(val_s,   shuffle=False)
    n_train_patches = len(train_ds)
    n_val_patches   = len(val_ds)
    n_batches_per_epoch = n_train_patches // BATCH

    print(f"Train images={len(train_s)}  ->  {n_train_patches:,} patches/epoch  "
          f"({n_batches_per_epoch:,} batches/epoch at batch size {BATCH})")
    print(f"Val images={len(val_s)}  ->  {n_val_patches:,} patches")
    print(f"Upper bound: {MAX_EPOCHS} epochs, early stopping patience={PATIENCE}\n", flush=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH, num_workers=WORKERS)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, num_workers=WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    model = PatchModel(pretrained=True).to(device)
    opt = torch.optim.Adam([
        {'params': model.extractor.backbone_parameters(), 'lr': LR_BACKBONE},
        {'params': model.extractor.head_parameters(),      'lr': LR_HEAD},
        {'params': model.head.parameters(),                'lr': LR_HEAD},
    ])

    # Resume: a completed epoch's checkpoint (checkpoints/epoch_N.pth) always
    # wins over a mid-epoch checkpoint from an OLDER epoch -- otherwise a
    # stale midepoch.pth left over from earlier in a since-completed epoch
    # would make the script needlessly re-run an already-finished epoch.
    start_epoch, best_val_acc, epochs_no_improve = 1, 0.0, 0
    completed = [int(f[len('epoch_'):-len('.pth')])
                 for f in os.listdir(CKPT_DIR) if f.startswith('epoch_') and f.endswith('.pth')] \
                if os.path.isdir(CKPT_DIR) else []
    last_completed = max(completed) if completed else 0

    midepoch_path = os.path.join(CKPT_DIR, 'midepoch.pth')
    if os.path.exists(midepoch_path):
        e, b, bva, eni = load_checkpoint(midepoch_path, model, opt, device)
        if e > last_completed:
            print(f"Resuming mid-epoch: epoch {e} was interrupted at batch {b} -- "
                  f"restarting epoch {e} from the beginning (standard streaming-dataset "
                  f"resume compromise; earlier completed epochs are untouched).")
            start_epoch, best_val_acc, epochs_no_improve = e, (bva or 0.0), eni
        else:
            print(f"Ignoring stale mid-epoch checkpoint (epoch {e} already completed); "
                  f"resuming from completed epoch {last_completed} instead.")
            os.remove(midepoch_path)
    if start_epoch == 1 and last_completed > 0:
        ckpt = torch.load(os.path.join(CKPT_DIR, f'epoch_{last_completed}.pth'), map_location=device)
        model.load_state_dict(ckpt['model_state']); opt.load_state_dict(ckpt['opt_state'])
        best_val_acc = ckpt['best_val_acc']
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        start_epoch = last_completed + 1
        print(f"Resuming from completed epoch {last_completed} "
              f"(best_val_acc so far {best_val_acc:.4f})")

    session_start = time.time()
    try:
        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, device, epoch,
                                               n_batches_per_epoch, session_start,
                                               best_val_acc, epochs_no_improve)
            va_loss, va_acc = evaluate(model, val_dl, device)
            print(f"[epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}", flush=True)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                epochs_no_improve = 0
                torch.save({'epoch': epoch, 'val_acc': best_val_acc,
                            'extractor_state': model.extractor.state_dict()}, OUT_PATH)
                print(f"  -> new best (val_acc={best_val_acc:.4f}), saved to {OUT_PATH}", flush=True)
            else:
                epochs_no_improve += 1
                print(f"  -> no improvement ({epochs_no_improve}/{PATIENCE})", flush=True)

            save_checkpoint(os.path.join(CKPT_DIR, f'epoch_{epoch}.pth'),
                             model, opt, epoch, 0, best_val_acc, epochs_no_improve)
            if os.path.exists(midepoch_path):
                os.remove(midepoch_path)  # this epoch is now fully checkpointed; drop the stale partial

            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best val_acc={best_val_acc:.4f} was at an earlier epoch.")
                break
    except TimeLimitReached:
        print("\nSession time limit reached -- exiting cleanly. "
              "Re-run this script to resume from the last checkpoint.")
        return

    print("\n--- On epoch count ---")
    print("No epoch count is asserted as 'correct' up front. This run used early")
    print(f"stopping (patience={PATIENCE}) on validation accuracy -- the actual")
    print("number of epochs trained is whatever's printed above before stopping.")
    print("Report that actual number in your FYP, not a planned/target number.")


if __name__ == '__main__':
    main()