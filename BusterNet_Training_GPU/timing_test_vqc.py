"""
timing_test_vqc.py

Standalone benchmark for VQCPred (both arms), mirroring this repo's existing
timing_test.py / timing_test_alkhuzaie.py / timing_test_resnet.py style.

Run this FIRST, before touching model_busternet.py or train_comparison.py.
Imports nothing from those files -- cannot interfere with the lab-PC
Experiment A/B job.

Usage:
    python timing_test_vqc.py
"""
import time
import torch

from vqc_fusion import VQCPred

N_TRAIN_IMAGES = 11603   # coco_subset/forgery_manifest_with_masks_train.csv


def bench(arm, pool_size, batch_size=8, n_batches=5, resolution=256, device="cpu"):
    print(f"\n=== arm={arm} pool_size={pool_size} resolution={resolution} device={device} ===")
    model = VQCPred(arm=arm, pool_size=pool_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(batch_size, 9, resolution, resolution, device=device)
    target = torch.randn(batch_size, 3, resolution, resolution, device=device)

    # warmup -- PennyLane/TorchLayer has first-call overhead, don't count it
    out = model(x)
    torch.nn.functional.mse_loss(out, target).backward()
    opt.step()
    opt.zero_grad()

    times = []
    for i in range(n_batches):
        t0 = time.time()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  batch {i + 1}/{n_batches}: {elapsed * 1000:.1f} ms")

    avg = sum(times) / len(times)
    batches_per_epoch = N_TRAIN_IMAGES // batch_size
    est_epoch = avg * batches_per_epoch
    print(f"avg/batch: {avg * 1000:.1f} ms  ->  extrapolated full epoch: "
          f"{est_epoch:.1f}s (~{est_epoch / 60:.1f} min) ON TOP OF the "
          f"~24 min/epoch classical baseline")
    return avg


if __name__ == "__main__":
    device = "cpu"   # forced -- GPU buys nothing here (VQC itself is CPU-bound
                      # regardless of device), and this avoids any GPU risk

    # Cheap sanity check first, small resolution, both arms.
    for arm in ("amplitude", "reupload"):
        bench(arm, pool_size=16, batch_size=8, n_batches=3, resolution=64, device=device)

    # The real question: full 256x256, both arms. NOT benchmarked yet as of
    # this design -- start with n_batches=1 if unsure how slow it'll be.
    for arm in ("amplitude", "reupload"):
        bench(arm, pool_size=None, batch_size=8, n_batches=1, resolution=256, device=device)
