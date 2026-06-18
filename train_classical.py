"""Step 2 (build FIRST): classical control heads on the matched Tier-1 features.
SVM-RBF (one per K) is the Al-khuzaie tie. The MLP is param-matched to the VQC it
will be compared against, so it is swept over (K, L) too."""
import argparse, json, os
import numpy as np
import yaml

from cache_contract import verify_cache, extract_config_hash
from reduction import load_reduction, reduce_to_k
from heads import (make_svm, MLPHead, fit_torch_head, vqc_param_count,
                   hidden_for_param_match, count_params)
from metrics import compute_metrics, summarize


def split_arrays(feats, m, pca, scaler, k):
    ids = m["image_ids"]; lab = m["labels"]; sp = m["split"]
    X = reduce_to_k(feats, pca, scaler, k)
    tr = [i for i, iid in enumerate(ids) if sp[iid] == "train"]
    te = [i for i, iid in enumerate(ids) if sp[iid] == "test"]
    return (X[tr], [lab[ids[i]] for i in tr],
            X[te], [lab[ids[i]] for i in te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    m, feats = verify_cache(cfg["cache_dir"], extract_config_hash(cfg))
    pca, scaler = load_reduction(cfg["cache_dir"])

    results = {}
    for K in cfg["K_sweep"]:
        Xtr, ytr, Xte, yte = split_arrays(feats, m, pca, scaler, K)
        res_k = {}

        # --- SVM-RBF: deterministic; decision_function centered at 0 ---
        svm = make_svm().fit(Xtr, ytr)
        scores = svm.decision_function(Xte)
        res_k["svm"] = dict(metrics=compute_metrics(yte, scores, threshold=0.0),
                            params=int(svm.best_estimator_.support_vectors_.size),
                            note="param count = SVM support-vector elements (not capacity-matched)")

        # --- MLP: param-matched to each VQC depth L, mean+/-std over seeds ---
        res_k["mlp"] = {}
        for L in cfg["L_sweep"]:
            target = vqc_param_count(K, L)
            hidden = hidden_for_param_match(K, target)
            runs, pcount = [], None
            for s in range(cfg["n_seeds"]):
                head = MLPHead(K, hidden)
                pcount = count_params(head)
                sc, _, _ = fit_torch_head(head, Xtr, ytr, Xte,
                                         cfg["train_epochs"], cfg["train_lr"], seed=s)
                runs.append(compute_metrics(yte, sc, threshold=0.5))
            res_k["mlp"][str(L)] = dict(summary=summarize(runs), params=pcount,
                                        matched_to_vqc_params=target, hidden=hidden)
        results[str(K)] = res_k

    os.makedirs(cfg["results_dir"], exist_ok=True)
    out = os.path.join(cfg["results_dir"], "classical.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Classical baselines written to {out}")


if __name__ == "__main__":
    main()
