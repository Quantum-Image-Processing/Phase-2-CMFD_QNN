"""Write-once, hashed, read-only feature cache — the physical enforcement of
Tier-1 matched conditions.

Contract:
  * features.npy   : (N, 512) frozen-backbone features, extracted ONCE.
  * pca.pkl        : PCA(n_components=k_max) fit on the TRAIN split only.
  * scaler.pkl     : MinMaxScaler([-1,1]) fit on TRAIN-split projections only.
  * manifest.json  : versions, hashes, per-image label + split, config identity.
After finalize_read_only() all four files are chmod 0o444. Every head reads the
same files; nothing in a training run may re-fit or overwrite them. verify_cache()
recomputes hashes and refuses to proceed if anything was touched.
"""
import hashlib, json, os, stat, time
import numpy as np

CACHE_VERSION = "1.0.0"
# Only these keys define cache identity. Changing K_sweep/L_sweep/train_* is fine.
EXTRACT_KEYS = ("extractor", "image_size", "k_max", "test_frac", "datasets", "seed")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def config_hash(d: dict) -> str:
    return sha256_bytes(json.dumps(d, sort_keys=True).encode())


def extract_subset(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in EXTRACT_KEYS}


def extract_config_hash(cfg: dict) -> str:
    return config_hash(extract_subset(cfg))


def make_split(image_ids, labels, source_ids, seed, test_frac):
    """Leakage-safe split: group by source_id (so CoMoFoD variants of one base
    image never straddle train/test), then stratify groups by majority label."""
    import collections, random
    rng = random.Random(seed)
    lab = dict(zip(image_ids, labels))
    groups = collections.defaultdict(list)
    for iid, src in zip(image_ids, source_ids):
        groups[src].append(iid)
    grp_label = {src: int(round(sum(lab[i] for i in ids) / len(ids)))
                 for src, ids in groups.items()}
    pos = [s for s in groups if grp_label[s] == 1]
    neg = [s for s in groups if grp_label[s] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    take = lambda lst: set(lst[:int(round(len(lst) * test_frac))])
    test_src = take(pos) | take(neg)
    split = {}
    for src, ids in groups.items():
        s = "test" if src in test_src else "train"
        for i in ids:
            split[i] = s
    return split


def _manifest_path(cache_dir):
    return os.path.join(cache_dir, "manifest.json")


def _dump_manifest(cache_dir, manifest):
    mpath = _manifest_path(cache_dir)
    if os.path.exists(mpath):
        os.chmod(mpath, stat.S_IWUSR | stat.S_IRUSR)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_cache(cache_dir, features, image_ids, labels, split,
                extractor_id, extractor_weights_hash, cfg):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "features.npy"), features)
    manifest = dict(
        cache_version=CACHE_VERSION,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config_hash=extract_config_hash(cfg),
        extractor_id=extractor_id,
        extractor_weights_hash=extractor_weights_hash,
        n_images=len(image_ids),
        feature_dim=int(features.shape[1]),
        image_ids=list(image_ids),
        labels={i: int(l) for i, l in zip(image_ids, labels)},
        split=split,
        features_sha256=sha256_array(features),
        pca_sha256=None, scaler_sha256=None,
        reduction_train_ids_sha256=None, k_max=None,
    )
    _dump_manifest(cache_dir, manifest)
    return manifest


def load_manifest(cache_dir):
    with open(_manifest_path(cache_dir)) as f:
        return json.load(f)


def finalize_read_only(cache_dir):
    ro = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 0o444
    for name in ("features.npy", "pca.pkl", "scaler.pkl", "manifest.json"):
        p = os.path.join(cache_dir, name)
        if os.path.exists(p):
            os.chmod(p, ro)


def verify_cache(cache_dir, expect_config_hash=None):
    """Fail loudly if the cache was modified, is incomplete, or was built from a
    different extract-config. Returns (manifest, features)."""
    m = load_manifest(cache_dir)
    feats = np.load(os.path.join(cache_dir, "features.npy"))
    if sha256_array(feats) != m["features_sha256"]:
        raise RuntimeError("CACHE VIOLATION: features.npy hash != manifest "
                           "(features were modified after caching).")
    for k in ("pca_sha256", "scaler_sha256"):
        if not m.get(k):
            raise RuntimeError(f"CACHE INCOMPLETE: {k} missing — run extract_features "
                               "so the reduction is fit on train-split-only and saved.")
    if sha256_file(os.path.join(cache_dir, "pca.pkl")) != m["pca_sha256"]:
        raise RuntimeError("CACHE VIOLATION: pca.pkl changed since manifest was written.")
    if sha256_file(os.path.join(cache_dir, "scaler.pkl")) != m["scaler_sha256"]:
        raise RuntimeError("CACHE VIOLATION: scaler.pkl changed since manifest was written.")
    if expect_config_hash and m["config_hash"] != expect_config_hash:
        raise RuntimeError("CONFIG MISMATCH: run config differs from cached extract-config. "
                           "Rebuild the cache with --force or restore the original config.")
    return m, feats
