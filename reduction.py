"""Fixed, shared 512->K reduction. Fit ONCE on the train split with n_components
= k_max; because PCA components are nested (ordered by variance), the first K of a
k_max fit equals PCA(K). MinMaxScaler is per-dimension, so slicing the first K dims
of a k_max scaler equals fitting on K dims. => one reduction, identical across both
heads AND across the whole K sweep. Heads may only load_reduction(); never fit."""
import os, pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from cache_contract import sha256_file, sha256_bytes, load_manifest, _dump_manifest


def fit_and_save_reduction(cache_dir, features, image_ids, split, k_max, seed):
    train_idx = [i for i, iid in enumerate(image_ids) if split[iid] == "train"]
    Xtr = features[train_idx]
    pca = PCA(n_components=k_max, random_state=seed).fit(Xtr)        # TRAIN ONLY
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(pca.transform(Xtr))  # TRAIN ONLY
    with open(os.path.join(cache_dir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(cache_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    m = load_manifest(cache_dir)
    m["pca_sha256"] = sha256_file(os.path.join(cache_dir, "pca.pkl"))
    m["scaler_sha256"] = sha256_file(os.path.join(cache_dir, "scaler.pkl"))
    train_ids_sorted = sorted(iid for iid in image_ids if split[iid] == "train")
    m["reduction_train_ids_sha256"] = sha256_bytes(repr(train_ids_sorted).encode())
    m["k_max"] = int(k_max)
    _dump_manifest(cache_dir, m)


def load_reduction(cache_dir):
    with open(os.path.join(cache_dir, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(cache_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return pca, scaler


def reduce_to_k(features, pca, scaler, k):
    """Project, scale on train stats, take first k dims, clip test points that fall
    outside the train range so Ry(x*pi/2) stays in [-pi/2, pi/2]."""
    P = scaler.transform(pca.transform(features))
    return np.clip(P[:, :k], -1.0, 1.0).astype(np.float32)
