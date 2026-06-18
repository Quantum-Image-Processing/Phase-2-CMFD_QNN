"""Step 3: the dressed VQC head on the SAME Tier-1 features/PCA the classical heads
used. Sweeps K and L; mean+/-std over seeds."""
import argparse, json, os
import yaml

from cache_contract import verify_cache, extract_config_hash
from reduction import load_reduction, reduce_to_k
from heads import DressedVQCHead, fit_torch_head, vqc_param_count, count_params
from metrics import compute_metrics, summarize
from train_classical import split_arrays


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
        results[str(K)] = {}
        for L in cfg["L_sweep"]:
            runs, pcount = [], None
            for s in range(cfg["n_seeds"]):
                head = DressedVQCHead(K, L, device=cfg["device"])
                pcount = count_params(head)
                sc, _, _ = fit_torch_head(head, Xtr, ytr, Xte,
                                         cfg["train_epochs"], cfg["train_lr"], seed=s)
                runs.append(compute_metrics(yte, sc, threshold=0.5))
            results[str(K)][str(L)] = dict(summary=summarize(runs), params=pcount,
                                           analytic_params=vqc_param_count(K, L))
            print(f"  K={K} L={L} done (params={pcount})")

    os.makedirs(cfg["results_dir"], exist_ok=True)
    out = os.path.join(cfg["results_dir"], "vqc.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"VQC results written to {out}")


if __name__ == "__main__":
    main()
