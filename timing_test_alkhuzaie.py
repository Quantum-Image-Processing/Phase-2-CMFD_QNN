"""Timing test for the Al-khuzaie-native extractor ONLY -- does not train
anything meaningfully, does not save a usable model. Run this first to
find out how long the real run will take on THIS machine before committing
to it, and to pick a sensible --max-batches value for controlled/resumable
runs (see train_extractor_alkhuzaie_native.py).

This is the small, paper-faithful CNN (9 conv layers, ~thousands of
params) -- expected to be much cheaper per batch than the ResNet18
version, which is why this one is the CPU candidate while ResNet18 runs
on GPU separately. But that expectation should still be checked here, not
assumed.

Runs TEST_BATCHES batches of the actual training loop (real data, real
model, real backward pass) and extrapolates to full epoch / full run time.
"""
import time
import torch
from Train_extractor_alkhuzaie_native import (
    PatchModel, FullPatchIterableDataset, BATCH, WORKERS,
    LR, MAX_EPOCHS, IMG_DIR, MASK_DIR, MANIFEST
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
        print("Running on CPU -- expected, this is the CPU-side model. "
              "The Al-khuzaie-native CNN is much smaller than ResNet18, "
              "so CPU should be far more practical here than it was for "
              "the ResNet18 timing test -- but verify below, don't assume.")

    model = PatchModel().to(device)
    n_params = sum(p.numel() for p in model.extractor.parameters())
    print(f"Al-khuzaie extractor params: {n_params:,} (compare to ResNet18's ~11M)")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Dataset: {n_train_patches:,} patches/epoch, "
          f"{n_batches_per_epoch:,} batches/epoch at batch size {BATCH}")
    print(f"Running {TEST_BATCHES} timed batches (first 5 excluded as warmup)...\n")

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

        if i >= 5:
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
    print(f"\nUse this to pick a --max-batches value for "
          f"train_extractor_alkhuzaie_native.py that fits a convenient session "
          f"length on this machine (e.g. if 1000 batches ~= 20 min, and you want "
          f"~2hr sessions, use --max-batches 6000).")
    print(f"\nNote: this does NOT include val-set evaluation time or checkpoint I/O.")
    print(f"Note: data-loading (image decode) may bottleneck depending on "
          f"WORKERS={WORKERS} and disk speed on this machine.")


if __name__ == '__main__':
    main()