# qhybrid-cmfd
### QNN Phase 1 — Quantum-Hybrid Copy-Move Forgery Detector

A confound-controlled comparison of VQC vs classical heads (SVM-RBF, MLP) for copy-move forgery detection, following the D2PRL / BusterNet pipeline with a quantum head attached.

---

## Architecture

```
Image (256×256)
  ↓
CMFDExtractor  — 5 conv blocks (Conv+BN+ReLU), stride-2 × 4 + stride-1 × 1
  → (B, 32, 16, 16)  lightweight, 32K params
  ↓ training only
MaskDecoder  — BusterNet Fig. 3: 4× BilinearUpPool2D + BNInception + Conv+sigmoid
  → (B, 1, 256, 256) pixel mask, Dice loss vs ground truth
  ↓ inference (decoder discarded after training)
Self-correlation (BusterNet Eq. 1–5) on 16×16×32 feature map
  → K-dim similarity profile per patch
  ↓
VQC / MLP / SVM head
  → 16×16 patch prediction grid → upsample → localization output
```

The MaskDecoder trains the extractor to produce spatially meaningful features. Once frozen, the decoder is discarded and the quantum/classical heads attach directly to the extractor output — exactly the transfer-learning pattern from Mari et al.

---

## What makes the comparison valid (Tier-1 confound control)

- One **frozen** extractor feeds every head — backbone is not a variable
- One **scaler** fit on the train split only — no leakage
- Identical feature vectors for SVM, MLP, and VQC — any perf difference is attributable to the **head**, not the representation

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Synthetic data | `generate_synthetic_coco.py` | `data/synthetic_coco/` — 15K images, 80/20 split |
| 2. Train extractor | `train_extractor.py` | `models/extractor_best.pth` — extractor weights only |
| 3. DEFACTO features | `extract_defacto_patches.py` | `cache_defacto/` — 339K patches, 512-D |
| 4. Head comparison | `train_defacto.py` | `results/defacto_heads.json` — SVM / MLP / VQC |

```bash
python generate_synthetic_coco.py --n 15000
python train_extractor.py
python extract_defacto_patches.py
python train_defacto.py
```

Run `pilot_decoder.py` before full training to validate gradient flow and decoder output (catches structural problems on 3 images in seconds).

---

## Key design decisions

**Synthetic data (D2PRL Section V-B)**
- Source: 80% random polygon, 20% COCO instance mask
- Affine: rotation ±180°, scale 0.5–2.0×
- Placement: source and target centroids must map to **different cells in the 16×16 grid**
- Post-process: JPEG Q∈[85,95] + Gaussian noise σ∈[1,3]
- 15,000 images, 6 skipped (0.04% skip rate)

**Extractor (lightweight, D2PRL-style)**
- 5 blocks: channels 3→16→32→32→32→32, output (B, 32, 16, 16) for 256×256 input
- 32,976 parameters — small enough to train on CPU

**MaskDecoder (BusterNet Fig. 3, training only)**
- 4× BilinearUpPool2D(2×) alternating with BNInception blocks
- BNInception: 4 branches (1×1 | 1×1→3×3 | 1×1→3×3→3×3 | MaxPool→1×1), 32 channels out
- Final Conv2d(32,1,1) + sigmoid → 256×256 mask
- 11,489 parameters — discarded after training

**VQC (Mari Example 2)**
- Embedding: H then Ry(x_k × π/2) per qubit
- L variational layers: per-qubit Ry + linear CNOT chain
- ⟨Z⟩ readout per qubit → linear K→1 logit
- Sweep K ∈ {4, 6, 8}, L ∈ {1, 2, 4, 6}

---

## Results (DEFACTO, VGG16 blocks 1-4 features, patch-level)

| Head | K | L | bAcc | AUC |
|------|---|---|------|-----|
| SVM-RBF | 8 | — | 0.667 | 0.721 |
| MLP | 8 | 1 | 0.642 | 0.710 |
| VQC | 8 | 6 | 0.653 | 0.711 |

All heads on identical frozen features. Results in `results/defacto_heads.json`.

---

## File layout

```
generate_synthetic_coco.py   D2PRL synthetic copy-move generation
train_extractor.py           CMFDExtractor + MaskDecoder end-to-end training
pilot_decoder.py             Validation gate — run before full training
extract_defacto_patches.py   VGG16 feature extraction on DEFACTO dataset
train_defacto.py             SVM / MLP / VQC head training on DEFACTO cache
heads.py                     DressedVQCHead, MLP, SVM definitions
reduction.py                 PCA + MinMaxScaler pipeline (train-only fit)
metrics.py                   Shared evaluation metrics
data.py                      Dataset adapters
cache_contract.py            Cache schema and leakage-safe split contract
config.yaml                  Shared config
```

---

## Citations

- Mari et al. — dressed-circuit VQC method implemented here
- D2PRL — synthetic generation protocol (Section V-B) and pixel-level ceiling
- BusterNet — MaskDecoder architecture (Fig. 3) and self-correlation features (Eq. 1–5)
- Röseler — compression-confound warning motivating Tier-1 control
