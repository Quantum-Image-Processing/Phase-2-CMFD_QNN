"""Train SVM / MLP / VQC heads on cached DEFACTO patch features.

Cache:     cache_defacto/  (patches.npy, labels.npy, image_ids.npy, manifest.json)
Reduction: PCA(k_max=8) + MinMaxScaler[-1,1], fit on train split only
Results:   results/defacto_heads.json

Subsampling for tractability:
  SVM : 20K balanced patches (10K per class) — RBF grid search on full 270K is intractable
  MLP : 50K balanced patches per seed       — mini-batch, fast
  VQC :  4K balanced patches per seed       — PennyLane simulation is slow per sample
  Test: full 69K test set for all heads (no subsampling at eval time)
"""
import os, json, random, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
sys.stdout.reconfigure(line_buffering=True)

from heads import (DressedVQCHead, MLPHead, make_svm,
                   vqc_param_count, hidden_for_param_match, count_params)
from reduction import fit_and_save_reduction, load_reduction, reduce_to_k

CACHE_DIR   = 'cache_defacto'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

K_LIST      = [4, 6, 8]
L_LIST      = [1, 2, 4, 6]
N_SEEDS     = 5
K_MAX       = 8
SEED        = 42

SVM_N       = 20_000   # 10K per class
MLP_N       = 50_000   # 25K per class
VQC_N       =  4_000   # 2K per class

MLP_BATCH   = 1024
MLP_EPOCHS  = 100
VQC_BATCH   = 64
VQC_EPOCHS  = 30
MLP_LR      = 1e-3
VQC_LR      = 1e-2
PATIENCE    = 10


# ----------------------------------------------------------- load + split
def load_cache():
    print("Loading cache ...", flush=True)
    features = np.load(os.path.join(CACHE_DIR, 'patches.npy'))
    labels   = np.load(os.path.join(CACHE_DIR, 'labels.npy'))
    img_ids  = np.load(os.path.join(CACHE_DIR, 'image_ids.npy'))
    with open(os.path.join(CACHE_DIR, 'manifest.json')) as f:
        manifest = json.load(f)
    print(f"  features: {features.shape}  labels: {labels.shape}", flush=True)
    return features, labels, img_ids, manifest


def reconstruct_split(img_ids_arr, seed=42, test_frac=0.20):
    """Reproduce the exact per-image split from extraction (same RNG, same order)."""
    seen, ordered = set(), []
    for iid in img_ids_arr:
        if iid not in seen:
            seen.add(iid); ordered.append(iid)
    rng = random.Random(seed)
    ids = list(ordered)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_set = set(ids[:n_test])
    return {iid: ('test' if iid in test_set else 'train') for iid in ids}


# ----------------------------------------------------------- helpers
def balanced_subsample(X, y, n_total, seed):
    rng = np.random.RandomState(seed)
    n_each = n_total // 2
    i0 = np.where(y == 0)[0]; i1 = np.where(y == 1)[0]
    s0 = rng.choice(i0, min(n_each, len(i0)), replace=False)
    s1 = rng.choice(i1, min(n_each, len(i1)), replace=False)
    idx = np.concatenate([s0, s1]); rng.shuffle(idx)
    return X[idx], y[idx]


def metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')
    return dict(balanced_acc=round(float(bacc), 4),
                f1=round(float(f1), 4),
                auc=round(float(auc), 4))


def image_level_metrics(y_score_patches, patch_img_ids):
    """Any-patch rule: image label = 1 if max patch score >= 0.5."""
    img_score = {}
    for score, iid in zip(y_score_patches, patch_img_ids):
        img_score[iid] = max(img_score.get(iid, -1e9), float(score))
    iids = list(img_score.keys())
    y_score_img = np.array([img_score[i] for i in iids])
    y_true_img  = np.ones(len(iids), dtype=int)   # all DEFACTO test images are forged
    return metrics(y_true_img, y_score_img)


def mean_std(vals):
    a = np.array(vals)
    return round(float(a.mean()), 4), round(float(a.std()), 4)


# ----------------------------------------------------------- mini-batch trainer
def fit_minibatch(head, Xtr, ytr, Xte, epochs, lr, seed, batch_size, patience):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    n = len(Xtr_t)
    val_n = max(1, int(0.15 * n))
    perm  = np.random.RandomState(seed).permutation(n)
    vi, ti = perm[:val_n], perm[val_n:]
    Xv, yv = Xtr_t[vi], ytr_t[vi]
    Xt, yt = Xtr_t[ti], ytr_t[ti]

    opt   = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best_val, best_state, wait = np.inf, None, 0

    for epoch in range(epochs):
        head.train()
        perm2 = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm2[i:i+batch_size]
            opt.zero_grad()
            lossf(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = lossf(head(Xv), yv).item()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  val_loss={vl:.4f}", flush=True)
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    early stop @ epoch {epoch+1}", flush=True)
                break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probs = torch.sigmoid(head(Xte_t)).numpy()
    return probs


# ----------------------------------------------------------- print table
def print_table(rows):
    hdr = f"{'Head':<22} {'K':>3} {'L':>3}  {'P-bAcc':>8} {'P-F1':>7} {'P-AUC':>7}  {'I-bAcc':>8} {'I-F1':>7} {'I-AUC':>7}"
    print("\n" + hdr)
    print('-' * len(hdr))
    for r in rows:
        L  = str(r.get('L', '-'))
        pm = r['patch']; im = r['image']
        def fmt(d):
            v = d['balanced_acc']; return f"{v:.4f}" if isinstance(v, float) else f"{v}"
        print(f"{r['head']:<22} {r['K']:>3} {L:>3}  "
              f"{pm['balanced_acc']:>8.4f} {pm['f1']:>7.4f} {pm['auc']:>7.4f}  "
              f"{im['balanced_acc']:>8.4f} {im['f1']:>7.4f} {im['auc']:>7.4f}",
              flush=True)


# ============================================================ MAIN
def main():
    features, labels, img_ids, manifest = load_cache()

    # --- reconstruct split
    split = reconstruct_split(img_ids, seed=manifest['seed'],
                              test_frac=manifest['test_frac'])
    tr_mask = np.array([split[i] == 'train' for i in img_ids])
    te_mask = ~tr_mask

    Xtr, ytr = features[tr_mask], labels[tr_mask]
    Xte, yte = features[te_mask], labels[te_mask]
    te_ids   = img_ids[te_mask]

    print(f"Train: {Xtr.shape[0]} patches  Test: {Xte.shape[0]} patches", flush=True)

    # --- fit reduction (PCA + scaler) on train only
    pca_path = os.path.join(CACHE_DIR, 'pca.pkl')
    if not os.path.exists(pca_path):
        print(f"Fitting PCA(k_max={K_MAX}) + scaler on train split ...", flush=True)
        fit_and_save_reduction(CACHE_DIR, features, img_ids, split, K_MAX, SEED)
        print("Reduction saved.", flush=True)
    else:
        print("Reduction already fitted — loading.", flush=True)
    pca, scaler = load_reduction(CACHE_DIR)

    all_results = []

    # ======================================================== SVM
    print("\n" + "="*60)
    print("SVM-RBF")
    print("="*60, flush=True)

    for K in K_LIST:
        Xtr_k = reduce_to_k(Xtr, pca, scaler, K)
        Xte_k = reduce_to_k(Xte, pca, scaler, K)

        Xs, ys = balanced_subsample(Xtr_k, ytr, SVM_N, SEED)
        print(f"\n  K={K}: training SVM on {len(ys)} patches ...", flush=True)
        t0 = time.time()
        clf = make_svm()
        clf.fit(Xs, ys)
        print(f"  Best: {clf.best_params_}  ({time.time()-t0:.0f}s)", flush=True)

        df = clf.decision_function(Xte_k)
        pm = metrics(yte, df, threshold=0.0)
        im = image_level_metrics(df, te_ids)
        print(f"  Patch  bAcc={pm['balanced_acc']:.4f}  F1={pm['f1']:.4f}  AUC={pm['auc']:.4f}")
        print(f"  Image  bAcc={im['balanced_acc']:.4f}  F1={im['f1']:.4f}  AUC={im['auc']:.4f}", flush=True)

        row = dict(head='SVM-RBF', K=K, patch=pm, image=im,
                   best_params=clf.best_params_)
        all_results.append(row)

    print_table([r for r in all_results if r['head'] == 'SVM-RBF'])

    # ======================================================== MLP
    print("\n" + "="*60)
    print("MLP")
    print("="*60, flush=True)

    for K in K_LIST:
        Xtr_k = reduce_to_k(Xtr, pca, scaler, K)
        Xte_k = reduce_to_k(Xte, pca, scaler, K)

        for L in L_LIST:
            target_p = vqc_param_count(K, L)
            h = hidden_for_param_match(K, target_p)
            print(f"\n  K={K} L={L}: MLP hidden={h}  "
                  f"params={target_p} (VQC) vs {(K+1)*h+(h+1)} (MLP)", flush=True)

            seed_pm, seed_im = [], []
            for s in range(N_SEEDS):
                Xs, ys = balanced_subsample(Xtr_k, ytr, MLP_N, seed=SEED + s)
                head = MLPHead(K, h)
                probs = fit_minibatch(head, Xs, ys, Xte_k,
                                      MLP_EPOCHS, MLP_LR, seed=SEED+s,
                                      batch_size=MLP_BATCH, patience=PATIENCE)
                seed_pm.append(metrics(yte, probs))
                seed_im.append(image_level_metrics(probs, te_ids))
                print(f"  seed {s}: patch bAcc={seed_pm[-1]['balanced_acc']:.4f}  "
                      f"AUC={seed_pm[-1]['auc']:.4f}", flush=True)

            def agg(lst, key):
                m, sd = mean_std([x[key] for x in lst])
                return f"{m:.4f}±{sd:.4f}"

            pm_agg = {k: mean_std([x[k] for x in seed_pm]) for k in ['balanced_acc','f1','auc']}
            im_agg = {k: mean_std([x[k] for x in seed_im]) for k in ['balanced_acc','f1','auc']}
            pm_mean = {k: pm_agg[k][0] for k in pm_agg}
            im_mean = {k: im_agg[k][0] for k in im_agg}

            print(f"  K={K} L={L} MLP  patch bAcc={agg(seed_pm,'balanced_acc')}  "
                  f"AUC={agg(seed_pm,'auc')}", flush=True)

            row = dict(head='MLP', K=K, L=L,
                       patch=pm_mean, image=im_mean,
                       patch_std={k: pm_agg[k][1] for k in pm_agg},
                       image_std={k: im_agg[k][1] for k in im_agg},
                       seed_patch=[{k: round(x[k],4) for k in x} for x in seed_pm],
                       seed_image=[{k: round(x[k],4) for k in x} for x in seed_im])
            all_results.append(row)

    print_table([r for r in all_results if r['head'] == 'MLP'])

    # save intermediate
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'defacto_heads.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nIntermediate results saved to {out_path}", flush=True)

    # ======================================================== VQC
    print("\n" + "="*60)
    print("VQC")
    print("="*60, flush=True)

    for K in K_LIST:
        Xtr_k = reduce_to_k(Xtr, pca, scaler, K)
        Xte_k = reduce_to_k(Xte, pca, scaler, K)

        for L in L_LIST:
            n_params = vqc_param_count(K, L)
            print(f"\n  K={K} L={L}: VQC params={n_params}", flush=True)

            seed_pm, seed_im = [], []
            for s in range(N_SEEDS):
                Xs, ys = balanced_subsample(Xtr_k, ytr, VQC_N, seed=SEED + s)
                head = DressedVQCHead(K, L)
                probs = fit_minibatch(head, Xs, ys, Xte_k,
                                      VQC_EPOCHS, VQC_LR, seed=SEED+s,
                                      batch_size=VQC_BATCH, patience=PATIENCE)
                seed_pm.append(metrics(yte, probs))
                seed_im.append(image_level_metrics(probs, te_ids))
                print(f"  seed {s}: patch bAcc={seed_pm[-1]['balanced_acc']:.4f}  "
                      f"AUC={seed_pm[-1]['auc']:.4f}", flush=True)

            pm_agg = {k: mean_std([x[k] for x in seed_pm]) for k in ['balanced_acc','f1','auc']}
            im_agg = {k: mean_std([x[k] for x in seed_im]) for k in ['balanced_acc','f1','auc']}
            pm_mean = {k: pm_agg[k][0] for k in pm_agg}
            im_mean = {k: im_agg[k][0] for k in im_agg}

            def agg(lst, key):
                m, sd = mean_std([x[key] for x in lst])
                return f"{m:.4f}±{sd:.4f}"
            print(f"  K={K} L={L} VQC  patch bAcc={agg(seed_pm,'balanced_acc')}  "
                  f"AUC={agg(seed_pm,'auc')}", flush=True)

            row = dict(head='VQC', K=K, L=L,
                       patch=pm_mean, image=im_mean,
                       patch_std={k: pm_agg[k][1] for k in pm_agg},
                       image_std={k: im_agg[k][1] for k in im_agg},
                       seed_patch=[{k: round(x[k],4) for k in x} for x in seed_pm],
                       seed_image=[{k: round(x[k],4) for k in x} for x in seed_im])
            all_results.append(row)

        # save after each K so we don't lose partial VQC results
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved (K={K} done).", flush=True)

    print_table([r for r in all_results if r['head'] == 'VQC'])

    # ======================================================== final table
    print("\n" + "="*60)
    print("FULL SUMMARY")
    print("="*60)
    print_table(all_results)

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
