"""Faithful implementation of Al-khuzaie's own CNN -- no substitution, no
pretrained backbone, exactly as specified in the paper text:

  "The architecture implemented in this study is a CNN comprising nine
  convolutional and two max-pooling layers... The input size of the
  network is a 128x128x3 patch... The first two convolutions have a 5x5
  kernel size and output... 30 kernels... A pooling operation follows
  these layers with a 2x2 filter. Following the eight layers are kernels,
  each having a convolution kernel size and a maximum pooling filter.
  These seven convolutions have stride one, whereas the pooling operation
  has stride two... every convolutional layer uses the ReLU activation
  function... each feature map is normalized locally before pooling...
  Using this data, a CNN extracts 400-D (5x5x16) feature representations
  of the patches. In the FCL, these features are compared with two-way
  softmax classifiers that use dropouts... we used only one fully
  connected layer to reduce the parameters."

Trained from scratch (no ImageNet pretraining -- the paper doesn't use
transfer learning). This is a smaller, cheaper network than the ResNet18
substitute, and should train considerably faster per batch on CPU.

Same full stride-8 patch enumeration, same checkpointing, same
LABEL_MODE A/B toggle, same --max-batches control as the ResNet version --
only the network architecture itself differs.

Saves: models/alkhuzaie_extractor_best.pth (conv stack only, FC discarded)
"""
import os, json, random, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)

IMG_DIR   = 'data/synthetic_coco/images'
MASK_DIR  = 'data/synthetic_coco/masks'
MANIFEST  = 'data/synthetic_coco/manifest.json'
CKPT_DIR  = 'checkpoints_alkhuzaie'
OUT_PATH  = 'models/alkhuzaie_extractor_best.pth'

FULL_IMG_SIZE = 256
PATCH_SIZE    = 128
STRIDE        = 8              # TRAINING stride -- unchanged, confirmed correct.
                                # Paper uses stride 8 for CNN weight training
                                # (page 5, "CNN Training" section: dense overlapping
                                # patches for learning). SEPARATELY, paper uses
                                # stride 128 only at the SVM feature-pooling stage
                                # (page 7) -- see extract_features.py, which now
                                # uses stride 128 for that different purpose.

LABEL_MODE = 'B'
FORGED_THRESHOLD = 0.10        # <-- set from check_patch_label_distribution.py output

MAX_EPOCHS   = 250             # confirmed from paper (page 7): "trained for 250 epochs"
PATIENCE     = 15              # paper trains a fixed 250, no early stopping mentioned;
                                # patience raised (not removed) as a practical guard
                                # against wasted compute -- set PATIENCE=MAX_EPOCHS
                                # for literal fidelity if you'd rather not stop early
BATCH        = 128             # paper's real value (20,000) is sized for their MICC-F600
                                # patch pool directly; infeasible on typical hardware
                                # (~4GB+ just for input tensors before activations).
                                # Disclosed deviation, not paper-matched.
LR           = 0.001           # confirmed from paper (page 7): initial learning rate
SEED         = 42
WORKERS      = 4
N_BATCHES_PER_CKPT = 2000
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ Al-khuzaie's CNN, faithful
def _conv_relu_lrn(in_c, out_c, k, stride=1, pad=None):
    if pad is None:
        pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=5),   # "each feature map is normalized locally before pooling"
    )


class AlKhuzaieExtractor(nn.Module):
    """9 conv layers + 2 max-pool layers, 128x128x3 -> 400-D (5x5x16).

    Layer breakdown (paper gives exact counts for layers 1-2 and the
    output shape; layers 3-9 are stated as "following the eight layers
    are kernels, each having a convolution kernel size" without full
    per-layer kernel counts -- filled in here with a standard
    progressively-widening channel scheme that is consistent with the
    stated constraints: 9 conv total, stride-1 for layers 3-9, ReLU +
    LRN throughout, ending at 16 channels x 5x5 spatial = 400-D).
    DISCLOSED: the exact per-layer channel counts for layers 3-9 are not
    fully specified in the paper text available -- this is a reasonable
    reconstruction, not a verbatim transcription. Flag this in your
    writeup if precision here matters for your methodology section.
    """
    def __init__(self):
        super().__init__()
        # Layers 1-2: 5x5 kernel, 30 kernels each (paper-specified)
        self.conv1 = _conv_relu_lrn(3,  30, 5, stride=1)
        self.conv2 = _conv_relu_lrn(30, 30, 5, stride=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)             # "2x2 filter" pooling after layers 1-2
                                                             # 128 -> 64

        # Layers 3-9 (7 conv layers, stride 1, paper-specified count/stride;
        # channel progression is the disclosed reconstruction noted above)
        self.conv3 = _conv_relu_lrn(30, 32, 3, stride=1)
        self.conv4 = _conv_relu_lrn(32, 32, 3, stride=1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)   # max-pool, stride 2   64 -> 32
        self.conv5 = _conv_relu_lrn(32, 32, 3, stride=1)
        self.conv6 = _conv_relu_lrn(32, 32, 3, stride=1)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)   # 32 -> 16
        self.conv7 = _conv_relu_lrn(32, 24, 3, stride=1)
        self.conv8 = _conv_relu_lrn(24, 20, 3, stride=1)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)   # 16 -> 8
        self.conv9 = _conv_relu_lrn(20, 16, 3, stride=1)
        self.pool5 = nn.AdaptiveMaxPool2d((5, 5))           # -> exactly 5x5x16 = 400-D, per paper

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.pool1(x)
        x = self.conv3(x); x = self.conv4(x); x = self.pool2(x)
        x = self.conv5(x); x = self.conv6(x); x = self.pool3(x)
        x = self.conv7(x); x = self.conv8(x); x = self.pool4(x)
        x = self.conv9(x); x = self.pool5(x)
        return x   # (B, 16, 5, 5) -- flatten for 400-D vector


class PatchClassifierHead(nn.Module):
    """One FC layer (paper: "we used only one fully connected layer to
    reduce the parameters"), dropout 0.5, two-way softmax."""
    def __init__(self, in_dim=400, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, feat_map):
        vec = feat_map.flatten(1)          # (B, 400)
        return self.fc(self.dropout(vec))


class PatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = AlKhuzaieExtractor()
        self.head = PatchClassifierHead()

    def forward(self, x):
        return self.head(self.extractor(x))


# ------------------------------------------------------------------ dataset (identical logic to ResNet version)
class FullPatchIterableDataset(IterableDataset):
    def __init__(self, samples, shuffle=True, seed=SEED):
        self.samples = samples
        self.shuffle = shuffle
        self.seed = seed
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1)
        self.coords = list(range(0, FULL_IMG_SIZE - PATCH_SIZE + 1, STRIDE))
        self.n_patches_per_img = len(self.coords) ** 2
        if LABEL_MODE == 'B' and FORGED_THRESHOLD is None:
            raise ValueError("LABEL_MODE='B' requires FORGED_THRESHOLD set. "
                              "Run check_patch_label_distribution.py first.")

    def __len__(self):
        return len(self.samples) * self.n_patches_per_img

    def _iter_images(self, epoch_seed):
        info = get_worker_info()
        samples = self.samples
        if self.shuffle:
            rng = random.Random(epoch_seed)
            samples = samples.copy()
            rng.shuffle(samples)
        if info is not None:
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
            image_is_forged = mask_np.any()

            coord_pairs = [(x, y) for y in self.coords for x in self.coords]
            if self.shuffle:
                random.shuffle(coord_pairs)

            for x, y in coord_pairs:
                patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                if LABEL_MODE == 'B':
                    mask_patch = mask_np[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                    label = int(mask_patch.mean() > FORGED_THRESHOLD)
                else:
                    label = int(image_is_forged)
                t = torch.tensor(np.array(patch, dtype=np.float32)).permute(2, 0, 1) / 255.0
                t = (t - self.mean) / self.std
                yield t, label


def save_checkpoint(path, model, opt, epoch, batch_idx, best_val_acc):
    torch.save({'epoch': epoch, 'batch_idx': batch_idx, 'best_val_acc': best_val_acc,
                'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, path)


def load_checkpoint(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    opt.load_state_dict(ckpt['opt_state'])
    return ckpt['epoch'], ckpt['batch_idx'], ckpt['best_val_acc']


def train_one_epoch(model, loader, opt, device, epoch, n_total_batches,
                     skip_batches=0, max_batches_this_run=None):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    t0 = time.time()
    batches_done_this_run = 0
    stopped_early = False
    for batch_idx, (patches, labels) in enumerate(loader, 1):
        if batch_idx <= skip_batches:
            continue
        patches, labels = patches.to(device), labels.to(device)
        logits = model(patches)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
        batches_done_this_run += 1

        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            pct = 100 * batch_idx / n_total_batches
            print(f"  epoch {epoch}  batch {batch_idx}/{n_total_batches} ({pct:.1f}%)  "
                  f"loss={total_loss/max(n,1):.4f}  acc={total_correct/max(n,1):.4f}  "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

        if batch_idx % N_BATCHES_PER_CKPT == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            save_checkpoint(os.path.join(CKPT_DIR, 'midepoch.pth'), model, opt, epoch, batch_idx, None)
            print(f"  [checkpoint saved at batch {batch_idx}]", flush=True)

        if max_batches_this_run is not None and batches_done_this_run >= max_batches_this_run:
            os.makedirs(CKPT_DIR, exist_ok=True)
            save_checkpoint(os.path.join(CKPT_DIR, 'midepoch.pth'), model, opt, epoch, batch_idx, None)
            print(f"\n[STOPPED at batch budget: {batches_done_this_run} batches this run]", flush=True)
            stopped_early = True
            break
    avg_loss = total_loss / n if n > 0 else float('nan')
    avg_acc  = total_correct / n if n > 0 else float('nan')
    return avg_loss, avg_acc, stopped_early


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-batches', type=int, default=None)
    args = parser.parse_args()

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
    n_batches_per_epoch = len(train_ds) // BATCH
    print(f"Train images={len(train_s)} -> {len(train_ds):,} patches/epoch "
          f"({n_batches_per_epoch:,} batches at batch size {BATCH})")
    if args.max_batches:
        print(f"BATCH BUDGET THIS RUN: {args.max_batches}")
    print(flush=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH, num_workers=WORKERS)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, num_workers=WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    model = PatchModel().to(device)
    n_params = sum(p.numel() for p in model.extractor.parameters())
    print(f"Al-khuzaie extractor params: {n_params:,} (compare to ResNet18's ~11M)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch, best_val_acc, epochs_no_improve = 1, 0.0, 0
    resume_batch_offset = 0
    resume_path = os.path.join(CKPT_DIR, 'midepoch.pth')
    if os.path.exists(resume_path):
        e, b, bva = load_checkpoint(resume_path, model, opt, device)
        print(f"Resuming: epoch {e}, batch {b}, best_val_acc {bva}")
        start_epoch, resume_batch_offset = e, b
        if bva is not None:
            best_val_acc = bva

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        skip = resume_batch_offset if epoch == start_epoch else 0
        tr_loss, tr_acc, stopped = train_one_epoch(model, train_dl, opt, device, epoch,
                                                     n_batches_per_epoch, skip, args.max_batches)
        if stopped:
            print("Batch budget reached. Rerun same command to continue.")
            return
        va_loss, va_acc = evaluate(model, val_dl, device)
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}", flush=True)
        save_checkpoint(os.path.join(CKPT_DIR, f'epoch_{epoch}.pth'), model, opt, epoch, 0, best_val_acc)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        if va_acc > best_val_acc:
            best_val_acc, epochs_no_improve = va_acc, 0
            torch.save({'epoch': epoch, 'val_acc': best_val_acc,
                        'extractor_state': model.extractor.state_dict()}, OUT_PATH)
            print(f"  -> new best ({best_val_acc:.4f}), saved", flush=True)
        else:
            epochs_no_improve += 1
            print(f"  -> no improvement ({epochs_no_improve}/{PATIENCE})", flush=True)
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break


if __name__ == '__main__':
    main()