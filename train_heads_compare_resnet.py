"""Timing test ONLY -- does not train anything meaningfully, does not save
a usable model. Run this first to find out how long the real run will take
before committing to it.

Runs TEST_BATCHES batches of the actual training loop (real data, real
model, real backward pass) and extrapolates to full epoch / full run time.
"""
import time
import torch
from train_extractor_patch_classifier import (
    PatchModel, FullPatchIterableDataset, BATCH, WORKERS,
    LR_BACKBONE, LR_HEAD, MAX_EPOCHS, IMG_DIR, MASK_DIR, MANIFEST
)
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json

TEST_BATCHES = 50   # small enough to run in a couple minutes, large enough to average out noise


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    train_s = [s for s in manifest['samples'] if s['split'] == 'train']

    train_ds = FullPatchIterableDataset(train_s, shuffle=True)
    n_train_patches = len(train_ds)
    n_batches_per_epoch = n_train_patches // BATCH

    train_dl = DataLoader(train_ds, batch_size=BATCH, num_workers=WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("WARNING: no GPU detected. Full training on CPU at this scale "
              "is very likely impractical -- pay close attention to the "
              "extrapolated time below.")

    model = PatchModel(pretrained=True).to(device)
    opt = torch.optim.Adam([
        {'params': model.extractor.backbone_parameters(), 'lr': LR_BACKBONE},
        {'params': model.extractor.head_parameters(),      'lr': LR_HEAD},
        {'params': model.head.parameters(),                'lr': LR_HEAD},
    ])

    print(f"Dataset: {n_train_patches:,} patches/epoch, "
          f"{n_batches_per_epoch:,} batches/epoch at batch size {BATCH}")
    print(f"Running {TEST_BATCHES} timed batches (first few excluded as warmup)...\n")

    model.train()
    times = []
    it = iter(train_dl)
    for i in range(TEST_BATCHES):
        patches, labels = next(it)
        patches, labels = patches.to(device), labels.to(device)

        t0 = time.time()
        logits = model(patches)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0

        if i >= 5:   # skip first 5 as warmup (cudnn autotune, first-batch overhead)
            times.append(dt)
        print(f"  batch {i+1}/{TEST_BATCHES}  {dt*1000:.1f} ms"
              f"{'  (warmup, excluded)' if i < 5 else ''}")

    avg_batch_time = sum(times) / len(times)
    epoch_time_sec = avg_batch_time * n_batches_per_epoch
    epoch_time_hr  = epoch_time_sec / 3600

    print(f"\nAverage batch time (post-warmup): {avg_batch_time*1000:.1f} ms")
    print(f"Extrapolated time per epoch: {epoch_time_hr:.2f} hours "
          f"({n_batches_per_epoch:,} batches)")
    print(f"Extrapolated time for {MAX_EPOCHS} epochs (upper bound, unlikely to "
          f"run this long due to early stopping): {epoch_time_hr*MAX_EPOCHS:.1f} hours")
    print(f"\nNote: this does NOT include val-set evaluation time or checkpoint")
    print(f"I/O overhead -- actual per-epoch time will be somewhat higher.")
    print(f"Note: data-loading (image decode) may be a bottleneck depending on")
    print(f"WORKERS={WORKERS} and disk speed -- if GPU utilization is low during")
    print(f"a real run, that's the likely cause, not the model itself.")


if __name__ == '__main__':
    main()