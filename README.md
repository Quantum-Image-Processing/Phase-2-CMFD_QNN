## QNN (Quantum Neural Network) Implementation for Project Phase-2, Sem-6

# qhybrid-cmfd

First quantum-hybrid copy-move forgery detector with a **confound-controlled**
VQC-vs-classical head comparison. Image-level binary (forged vs authentic),
PennyLane simulator only.

## The one idea that makes the comparison valid

One **frozen** ImageNet ResNet18 (fc removed -> 512-D) feeds **every** head. A single
PCA (fit on the train split only) reduces 512->K. The K-D vectors are written to a
**write-once, hashed, read-only cache**; the SVM, the param-matched MLP, and the
dressed VQC all read the *same* file. Nothing in a training run can re-fit or alter
the reduction. So any performance difference is attributable to the **head**, not the
backbone or the compression. (PCA components are nested, so the first K of the
k_max-component fit equals PCA(K) — the reduction is identical across the whole K
sweep as well as across heads.)

This is Tier 1 (strict match). A trainable 512->K reduction is a deliberately
*separate, labelled* Tier-2 "Mari-faithful" variant — not in this skeleton, because a
jointly-trained reduction differs per head and reintroduces the Röseler confound.

## Build order

1. `python src/extract_features.py`            # frozen extractor -> read-only cache
2. `python src/train_classical.py`             # SVM + param-matched MLP (run FIRST)
3. `python src/train_vqc.py`                    # dressed VQC, sweep K and L

Run 2 before 3 on purpose: the classical numbers decide K, depth, and the
patch-vs-whole-image question. See `HYPOTHESES.md` — Baseline 1 is expected near
chance, and that is the planned trigger for the patch + self-similarity revision.

## Layout

    config.yaml            shared config (cache identity vs sweep separated)
    HYPOTHESES.md          pre-registered predictions (read before running)
    src/cache_contract.py  write-once / hashed / read-only enforcement + leakage-safe split
    src/reduction.py       fixed shared PCA + scaler (train-only), nested-K slicing
    src/heads.py           SVM, MLP, gate-level dressed VQC (Mari Example 2), param matching
    src/extract_features.py frozen ResNet18 -> 512-D -> cache -> finalize read-only
    src/train_classical.py classical controls on Tier-1 features
    src/train_vqc.py       VQC head on the SAME Tier-1 features
    src/data.py            dataset adapters (TODO: local paths/layouts)
    src/metrics.py         accuracy / P / R / F1 / AUC, mean +/- std
    cache/  results/       generated

## VQC (gate level, Mari Example 2)

Embedding `H` then `Ry(x_k * pi/2)` per qubit; L layers of per-qubit `Ry` + linear
CNOT chain; `<Z>` readout per qubit; linear K->1 to a logit. Sweep `K in {4,6,8}`,
`L in {1,2,4,6}`. Analytic expectations in simulation (no shots) for speed/stability.

## Datasets

MICC-F600 / F2000, CoMoFoD, CASIA (copy-move subset). `src/data.py` adapters return
`(image_id, path, label, source_id)`; `source_id` groups CoMoFoD variants so they
never straddle the split. Fill the TODOs for your local copies.

## Citations (related work, not baselines to reproduce)

Al-khuzaie (head-swap template; from-scratch CNN deliberately not reproduced),
Mari (the dressed-circuit method, implemented here), Röseler (compression-confound
warning motivating Tier 1), D2PRL / QDL-CMFD / Cozzolino Dense-Field (pixel-level
ceiling, deferred). Note: D2PRL leads the best prior *deep* model by ~16 percentage
points pixel-level F1 on CASIA (0.806 vs 0.643), with a keypoint method (SSG, 0.796)
nearly level — it does not beat all prior work.
