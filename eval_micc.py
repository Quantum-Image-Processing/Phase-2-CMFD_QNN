"""Evaluate CMFDExtractor features on MICC-F2000 with SVM / MLP / VQC heads.

Pipeline
--------
1. Load MICC-F2000 via adapter_micc_f2000()
2. Extract 32-dim image features: CMFDExtractor -> global-avg-pool -> (N, 32)
3. 80/20 leakage-safe split grouped by source_id
4. PCA(8) + MinMaxScaler fit on train split only, cached to cache_micc/
5. Sweep K in {4,6,8}, L in {1,2,4}, N_seeds=5
6. Report balanced accuracy, F1, AUC for SVM / MLP / VQC
7. Save results/micc_heads.json
"""
import os, json, sys
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

sys.stdout.reconfigure(line_buffering=True)

from data import adapter_micc_f2000
from heads import (make_svm, MLPHead, DressedVQCHead,
                   hidden_for_param_match, vqc_param_count)
from reduction import fit_and_save_reduction, load_reduction, reduce_to_k
from cache_contract import make_split, write_cache, load_manifest, sha256_file

# ── paths ─────────────────────────────────────────────────────────────────────
MICC_ROOT   = 'data/MICC-F2000'
CACHE_DIR   = 'cache_micc'
RESULTS_DIR = 'results'
MODEL_PATH  = 'models/extractor_best.pth'

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 256
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
K_MAX      = 8
K_LIST     = [4, 6, 8]
L_LIST     = [1, 2, 4]
N_SEEDS    = 5
SEED       = 42
TEST_FRAC  = 0.20
MLP_EPOCHS = 300
MLP_LR     = 1e-3
VQC_EPOCHS = 80
VQC_LR     = 1e-2
PATIENCE   = 15


# ── CMFDExtractor (matches train_extractor.py exactly) ───────────────────────
def _bnconv(in_c, out_c, k, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class CMFDExtractor(nn.Module):
    """5-block extractor -> (B, 32, 16, 16) for 256x256 input."""
    def __init__(self):
        super().__init__()
        self.block1 = _bnconv(3,  16, 3, stride=2, pad=1)
        self.block2 = _bnconv(16, 32, 3, stride=2, pad=1)
        self.block3 = _bnconv(32, 32, 3, stride=2, pad=1)
        self.block4 = _bnconv(32, 32, 3, stride=2, pad=1)
        self.block5 = _bnconv(32, 32, 3, stride=1, pad=1)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.block5(x)


# ── training helper (silent per-epoch, prints only per-run summary) ───────────
def _fit_head(head, Xtr, ytr, Xte, epochs, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    n     = len(Xtr_t)
    val_n = max(1, int(0.15 * n))
    perm  = np.random.RandomState(seed).permutation(n)
    vi, ti = perm[:val_n], perm[val_n:]

    opt   = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best_val, best_state, wait = float('inf'), None, 0

    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        lossf(head(Xtr_t[ti]), ytr_t[ti]).backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            vl = lossf(head(Xtr_t[vi]), ytr_t[vi]).item()
        if vl < best_val - 1e-6:
            best_val, best_state, wait = vl, deepcopy(head.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(Xte_t)).numpy()


# ── metrics ───────────────────────────────────────────────────────────────────
def _metrics(scores, labels):
    preds = (scores >= 0.5).astype(int)
    ba  = balanced_accuracy_score(labels, preds)
    f1  = float(f1_score(labels, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float('nan')
    return {'balanced_acc': ba, 'f1': f1, 'auc': auc}


def _mean_std(runs):
    keys = runs[0].keys()
    mean = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    std  = {k: float(np.std( [r[k] for r in runs])) for k in keys}
    return mean, std


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,   exist_ok=True)

    # ── 1. load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading MICC-F2000 from {MICC_ROOT} ...")
    records = adapter_micc_f2000(MICC_ROOT)
    image_ids  = [r[0] for r in records]
    paths      = [r[1] for r in records]
    labels     = [r[2] for r in records]
    source_ids = [r[3] for r in records]
    forged = sum(labels); auth = len(labels) - forged
    print(f"  Total: {len(records)}  Forged: {forged}  Authentic: {auth}")

    # ── 2. extract or load cached features ───────────────────────────────────
    feat_path = os.path.join(CACHE_DIR, 'features.npy')
    meta_path = os.path.join(CACHE_DIR, 'manifest.json')

    if os.path.exists(feat_path) and os.path.exists(meta_path):
        print(f"\nLoading cached features from {CACHE_DIR}/ ...")
        features = np.load(feat_path)
        manifest = load_manifest(CACHE_DIR)
        split    = manifest['split']
        print(f"  {features.shape[0]} vectors, dim={features.shape[1]}")
    else:
        print(f"\nLoading extractor from {MODEL_PATH} ...")
        ckpt = torch.load(MODEL_PATH, map_location=device)
        ext  = CMFDExtractor().to(device)
        ext.load_state_dict(ckpt['extractor_state'])
        ext.eval()
        for p in ext.parameters():
            p.requires_grad_(False)

        mean_t = torch.tensor(MEAN, device=device).view(1,3,1,1)
        std_t  = torch.tensor(STD,  device=device).view(1,3,1,1)
        pool   = nn.AdaptiveAvgPool2d(1).to(device)

        print(f"Extracting features from {len(records)} images ...")
        feats = []
        for i, path in enumerate(paths):
            img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            t   = torch.tensor(np.array(img, np.float32)).permute(2,0,1).unsqueeze(0).to(device) / 255.0
            t   = (t - mean_t) / std_t
            with torch.no_grad():
                f = pool(ext(t)).squeeze().cpu().numpy()   # (32,)
            feats.append(f)
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(records)}]")
        features = np.array(feats, dtype=np.float32)
        print(f"  Done — shape {features.shape}")

        split = make_split(image_ids, labels, source_ids, seed=SEED, test_frac=TEST_FRAC)
        wh    = sha256_file(MODEL_PATH)
        write_cache(CACHE_DIR, features, image_ids, labels, split,
                    extractor_id='cmfd_extractor', extractor_weights_hash=wh,
                    cfg={'extractor': 'cmfd_extractor', 'image_size': IMG_SIZE,
                         'k_max': K_MAX, 'test_frac': TEST_FRAC,
                         'datasets': [{'name': 'micc_f2000', 'root': MICC_ROOT}],
                         'seed': SEED})
        fit_and_save_reduction(CACHE_DIR, features, image_ids, split, K_MAX, SEED)
        print(f"  Cached to {CACHE_DIR}/")

    # ── 3. build index arrays ─────────────────────────────────────────────────
    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    train_idx = [id_to_idx[iid] for iid in image_ids if split[iid] == 'train']
    test_idx  = [id_to_idx[iid] for iid in image_ids if split[iid] == 'test']
    y_train   = np.array([labels[i] for i in train_idx])
    y_test    = np.array([labels[i] for i in test_idx])
    print(f"\nSplit  train={len(train_idx)} (forged={y_train.sum()})"
          f"  test={len(test_idx)} (forged={y_test.sum()})")

    # ── 4. load reduction ─────────────────────────────────────────────────────
    pca, scaler = load_reduction(CACHE_DIR)

    # ── 5. sweep heads ────────────────────────────────────────────────────────
    results = {}

    for K in K_LIST:
        Xtr = reduce_to_k(features[train_idx], pca, scaler, K)
        Xte = reduce_to_k(features[test_idx],  pca, scaler, K)

        # SVM-RBF ─────────────────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"[K={K}] SVM-RBF ...")
        svm  = make_svm()
        svm.fit(Xtr, y_train)
        raw  = svm.decision_function(Xte)
        lo, hi = raw.min(), raw.max()
        svm_scores = (raw - lo) / (hi - lo + 1e-9)
        svm_m = _metrics(svm_scores, y_test)
        print(f"  bAcc={svm_m['balanced_acc']:.4f}  F1={svm_m['f1']:.4f}  AUC={svm_m['auc']:.4f}")
        results[f"svm_K{K}"] = {'mean': svm_m, 'std': {}, 'runs': [svm_m]}

        # MLP ─────────────────────────────────────────────────────────────────
        target_p = vqc_param_count(K, 1)
        hidden   = hidden_for_param_match(K, target_p)
        print(f"[K={K}] MLP hidden={hidden}  ({N_SEEDS} seeds) ...")
        mlp_runs = []
        for s in range(N_SEEDS):
            probs = _fit_head(MLPHead(K, hidden), Xtr, y_train, Xte,
                              MLP_EPOCHS, MLP_LR, seed=s)
            m = _metrics(probs, y_test)
            mlp_runs.append(m)
            print(f"  seed {s}: bAcc={m['balanced_acc']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
        mn, sd = _mean_std(mlp_runs)
        print(f"  MLP mean: bAcc={mn['balanced_acc']:.4f}±{sd['balanced_acc']:.4f}"
              f"  F1={mn['f1']:.4f}  AUC={mn['auc']:.4f}")
        results[f"mlp_K{K}"] = {'mean': mn, 'std': sd, 'runs': mlp_runs}

        # VQC ─────────────────────────────────────────────────────────────────
        for L in L_LIST:
            print(f"[K={K} L={L}] VQC ({N_SEEDS} seeds) ...")
            vqc_runs = []
            for s in range(N_SEEDS):
                probs = _fit_head(DressedVQCHead(K, L), Xtr, y_train, Xte,
                                  VQC_EPOCHS, VQC_LR, seed=s)
                m = _metrics(probs, y_test)
                vqc_runs.append(m)
                print(f"  seed {s}: bAcc={m['balanced_acc']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
            mn, sd = _mean_std(vqc_runs)
            print(f"  VQC mean: bAcc={mn['balanced_acc']:.4f}±{sd['balanced_acc']:.4f}"
                  f"  F1={mn['f1']:.4f}  AUC={mn['auc']:.4f}")
            results[f"vqc_K{K}_L{L}"] = {'mean': mn, 'std': sd, 'runs': vqc_runs}

    # ── 6. final summary table ────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"FINAL RESULTS — MICC-F2000")
    print(f"{'Config':<20} {'bAcc':>8} {'±':>6} {'F1':>8} {'AUC':>8}")
    print("-"*65)
    for K in K_LIST:
        r = results[f"svm_K{K}"]
        print(f"{'SVM-RBF K='+str(K):<20} {r['mean']['balanced_acc']:>8.4f} {'—':>6} "
              f"{r['mean']['f1']:>8.4f} {r['mean']['auc']:>8.4f}")
    for K in K_LIST:
        r = results[f"mlp_K{K}"]
        print(f"{'MLP K='+str(K):<20} {r['mean']['balanced_acc']:>8.4f} "
              f"{r['std']['balanced_acc']:>6.4f} "
              f"{r['mean']['f1']:>8.4f} {r['mean']['auc']:>8.4f}")
    for K in K_LIST:
        for L in L_LIST:
            r = results[f"vqc_K{K}_L{L}"]
            print(f"{'VQC K='+str(K)+' L='+str(L):<20} {r['mean']['balanced_acc']:>8.4f} "
                  f"{r['std']['balanced_acc']:>6.4f} "
                  f"{r['mean']['f1']:>8.4f} {r['mean']['auc']:>8.4f}")
    print("="*65)

    out = os.path.join(RESULTS_DIR, 'micc_heads.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
