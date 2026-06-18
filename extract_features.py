"""Step 1: frozen ResNet18 -> 512-D features, extracted ONCE, then PCA+scaler fit
on train-split-only, then the whole cache is finalized read-only. Refuses to
overwrite an existing cache unless --force."""
import argparse, os
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

from cache_contract import (make_split, write_cache, finalize_read_only,
                            sha256_bytes)
from reduction import fit_and_save_reduction
from data import load_all


def build_extractor():
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()                 # -> 512-D global-avg-pooled feature
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)                  # FROZEN
    preprocess = weights.transforms()            # canonical resize/crop/normalize
    weights_tag = sha256_bytes(b"resnet18-IMAGENET1K_V1")
    return net, preprocess, weights_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cache_dir = cfg["cache_dir"]

    if os.path.exists(os.path.join(cache_dir, "manifest.json")) and not args.force:
        raise SystemExit("Cache already exists (write-once). Use --force to rebuild.")

    items = load_all(cfg["datasets"])            # [(id, path, label, source_id)]
    image_ids = [x[0] for x in items]
    paths     = [x[1] for x in items]
    labels    = [x[2] for x in items]
    sources   = [x[3] for x in items]

    net, preprocess, weights_tag = build_extractor()
    feats = np.zeros((len(items), 512), dtype=np.float32)
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            feats[i] = net(preprocess(img).unsqueeze(0)).squeeze(0).numpy()
            if (i + 1) % 100 == 0:
                print(f"  extracted {i + 1}/{len(items)}")

    split = make_split(image_ids, labels, sources, cfg["seed"], cfg["test_frac"])
    write_cache(cache_dir, feats, image_ids, labels, split,
                "resnet18", weights_tag, cfg)
    fit_and_save_reduction(cache_dir, feats, image_ids, split,
                           cfg["k_max"], cfg["seed"])
    finalize_read_only(cache_dir)
    n_tr = sum(v == "train" for v in split.values())
    print(f"Cache built: {len(items)} images ({n_tr} train / {len(items)-n_tr} test), "
          f"512-D, PCA({cfg['k_max']}) fit on train only. Finalized read-only.")


if __name__ == "__main__":
    main()
