# Pre-registered hypotheses

> Written **before** running anything. The point is that a near-chance Baseline-1
> reads as planned evidence, not failure — so the prediction is committed in
> advance, not rationalized afterward.

## H1 — Baseline 1 (frozen whole-image features) will land near chance (~50%)

**Prediction (pre-committed).** Experiment 1 — frozen ImageNet ResNet18, whole
image, global-average-pooled to 512-D, reduced to K, classical head — will score
**near chance (~50% balanced accuracy, F1 ~0.5)** on image-level copy-move.

**Why this is expected, not a bug.** Copy-move is a *relational* property: a region
is duplicated. A frozen ImageNet backbone encodes object semantics ("what is in the
image"), and global pooling collapses all spatial layout. The duplicated object
appears in the pooled vector exactly once, indistinguishable from a single instance.
The features are structurally blind to duplication.

**What the result triggers (decided in advance):**
- Near-chance (≈50%) → **not a debugging spiral.** It is the evidence that motivates
  the patch + self-similarity revision (per-patch features + top-k pairwise
  similarity, so "a region repeats" becomes representable). Proceed to that revision.
- Clearly above chance (say >65%) → investigate *why* the global feature carries
  copy-move signal (dataset artifact? compression cue? class imbalance leak?) before
  trusting it.

## H2 — The head comparison is meaningful only at matched conditions

Any SVM/MLP-vs-VQC difference is interpretable as a *head* effect **only** under
Tier-1: one frozen backbone, one fixed PCA fit on train-split-only, identical K-D
inputs to every head (enforced by the read-only cache contract). A jointly-trained
reduction would reintroduce the Röseler confound and is excluded from the headline.

## H3 — No accuracy-advantage claim

We do not predict the VQC beats the classical heads on accuracy. The contribution is
novelty + a confound-controlled comparison at matched K and matched parameter count.
The only known quantum accuracy win in the cited literature (Röseler, MNIST) uses 49
qubits on real hardware — outside this project's simulator / 4–8-qubit constraint.
