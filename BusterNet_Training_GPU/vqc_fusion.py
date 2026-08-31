"""
vqc_fusion.py

VQC replacement for BusterNet Fusion's final Conv2d(9->3) step.

DESIGN (decided across a full design conversation -- NOT the earlier "additive
pooled gate on raw 12 values" idea from a superseded, pre-Mari-paper pass):
  - model_busternet.py's `self.fusion` (BnInception 12->9) is KEPT UNCHANGED.
    Per Mari et al. 2020's "dressed quantum circuit" (Eq. 9,
    Q~ = L(nq->nout) o Q o L(nin->nq)), it already IS the trainable
    pre-processing layer that template wants -- no new reduction layer needed.
  - Only `self.pred` (Conv2d(9->3)) is replaced. This module produces the
    replacement, operating at every pixel (same resolution self.pred already
    runs at): 9 values in -> VQC -> small classical post-layer -> 3 logits out.
  - Two arms, a genuine ablation (not primary+fallback):
      * Amplitude, 4 qubits (2^4=16 >= 9 values)
      * Angle + data re-uploading, 3 qubits (9 = 3 rounds x 3 values, exact fit)
  - Amplitude's forced L2-normalization discards the real magnitude signal
    measured on busternet_converted.pth (L2 norm spread ~0.97-6.25 across real
    pixels) -- fixed via an extra classical side-channel carrying the
    pre-normalization norm.
  - Whether this needs to run pooled or at full 256x256 (65,536 circuit
    calls/image) is NOT YET DECIDED -- no real benchmark exists for this exact
    circuit at that scale yet. pool_size=None (default) = full resolution;
    pass an int to pool as a fallback if timing_test_vqc.py shows full-res is
    too slow. Decide AFTER running that script.
  - Off by default everywhere -- does not touch the currently-running lab-PC
    Experiment A/B job unless explicitly wired in by hand (see the
    INTEGRATION DIFFS comment block at the bottom of this file).
  - Uses `busternet_converted.pth` (the original authors' pretrained
    checkpoint) as the only source of real feature statistics available so
    far -- your own Experiment A/B checkpoints don't exist yet. Expect to
    re-tune scaling/qubit counts once a real trained checkpoint exists.

Mirrors heads.py's PennyLane/TorchLayer/backprop/angle-encoding conventions
(the established, working precedent already in this repo).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

N_VALUES = 9  # BnInception(12->9)'s output width -- the thing being replaced


# =============================================================================
# Arm 1: Amplitude encoding, 4 qubits
# =============================================================================
AMP_QUBITS = 4       # 2**4 = 16 >= 9 values, padded with zeros
AMP_LAYERS = 4        # matches the depth used in this session's benchmarks

_amp_dev = qml.device("default.qubit", wires=AMP_QUBITS)


@qml.qnode(_amp_dev, interface="torch", diff_method="backprop")
def _amplitude_circuit(inputs, weights):
    """inputs: [..., 9] (scaled vector -- AmplitudeEmbedding L2-normalizes
    internally). weights: [AMP_LAYERS, AMP_QUBITS] (shared, not batched)."""
    qml.AmplitudeEmbedding(inputs, wires=range(AMP_QUBITS), normalize=True, pad_with=0.0)
    for l in range(AMP_LAYERS):
        for q in range(AMP_QUBITS):
            qml.RY(weights[l, q], wires=q)
        for q in range(AMP_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
    return [qml.expval(qml.PauliZ(q)) for q in range(AMP_QUBITS)]


class AmplitudeVQC(nn.Module):
    """9 values -> 4-qubit VQC -> [N, AMP_QUBITS + 1]. The +1 is the
    pre-normalization L2 norm side-channel, recovering the magnitude info
    AmplitudeEmbedding's normalize=True otherwise discards."""

    # AmplitudeEmbedding(normalize=True) divides by the input vector's L2
    # norm. A pixel whose 9 fused-feature values are all exactly (or near-
    # exactly) zero -- plausible here, BusterNet's fused features come out
    # of ReLU, so flat/pristine background pixels can legitimately hit this
    # -- is a 0/0 division -> NaN. Confirmed by direct test, not hypothetical.
    # Fixed by nudging away from the origin by a fixed, tiny epsilon before
    # embedding; negligible relative to any real (non-degenerate) signal.
    _ZERO_VEC_EPS = 1e-6

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(N_VALUES, affine=True)
        self.qlayer = qml.qnn.TorchLayer(_amplitude_circuit, {"weights": (AMP_LAYERS, AMP_QUBITS)})

    def forward(self, x):
        x = self.bn(x) + self._ZERO_VEC_EPS       # [N, 9] -- epsilon avoids the all-zero/NaN case
        norm = x.norm(dim=1, keepdim=True)         # [N, 1] captured BEFORE AmplitudeEmbedding normalizes
        q_out = self.qlayer(x)                      # [N, AMP_QUBITS]
        return torch.cat([q_out, norm], dim=1)      # [N, AMP_QUBITS + 1]

    @property
    def out_features(self):
        return AMP_QUBITS + 1


# =============================================================================
# Arm 2: Angle encoding with data re-uploading, 3 qubits
# =============================================================================
REUP_QUBITS = 3        # 9 = 3 rounds x 3 values -- exact fit, no padding
REUP_ROUNDS = 3
REUP_LAYERS_PER_ROUND = 1

_reup_dev = qml.device("default.qubit", wires=REUP_QUBITS)


@qml.qnode(_reup_dev, interface="torch", diff_method="backprop")
def _reupload_circuit(inputs, weights):
    """inputs: [..., 9], split into 3 rounds of 3 values.
    weights: [REUP_ROUNDS * REUP_LAYERS_PER_ROUND, REUP_QUBITS] (shared).

    NOTE: uses `...` (ellipsis) indexing throughout, not a plain leading-dim
    index -- PennyLane's TorchLayer passes the whole batch through at once,
    so `inputs[start:end]` would silently slice the BATCH dimension instead
    of the feature dimension. `inputs[..., start:end]` is the correct form.
    """
    for q in range(REUP_QUBITS):
        qml.Hadamard(wires=q)   # once, establishes superposition

    layer_idx = 0
    for r in range(REUP_ROUNDS):
        chunk = inputs[..., r * REUP_QUBITS:(r + 1) * REUP_QUBITS]
        for q in range(REUP_QUBITS):
            qml.RY(chunk[..., q] * (torch.pi / 2), wires=q)
        for _ in range(REUP_LAYERS_PER_ROUND):
            for q in range(REUP_QUBITS):
                qml.RY(weights[layer_idx, q], wires=q)
            for q in range(REUP_QUBITS - 1):
                qml.CNOT(wires=[q, q + 1])
            layer_idx += 1

    return [qml.expval(qml.PauliZ(q)) for q in range(REUP_QUBITS)]


class ReuploadVQC(nn.Module):
    """9 values -> 3-qubit VQC -> [N, REUP_QUBITS]. No forced normalization --
    magnitude preserved (up to RY's 2*pi periodicity)."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(N_VALUES, affine=True)
        n_layers = REUP_ROUNDS * REUP_LAYERS_PER_ROUND
        self.qlayer = qml.qnn.TorchLayer(_reupload_circuit, {"weights": (n_layers, REUP_QUBITS)})

    def forward(self, x):
        x = torch.tanh(self.bn(x))   # bound to [-1,1] so RY(x*pi/2) stays in [-pi/2, pi/2]
        return self.qlayer(x)          # [N, REUP_QUBITS]

    @property
    def out_features(self):
        return REUP_QUBITS


# =============================================================================
# Wrapper: replaces model_busternet.py's `self.pred` (Conv2d(9,3,...))
# =============================================================================
class VQCPred(nn.Module):
    """Drop-in replacement for `nn.Conv2d(9, 3, kernel_size=3, padding=1)`.

    arm: "amplitude" or "reupload" -- pick ONE per model instance; run both as
    separate experiments for the ablation, don't blend them at runtime.

    pool_size: None = full 256x256 (NOT YET BENCHMARKED at this config --
    run timing_test_vqc.py first). An int (e.g. 16) pools first as a fallback.
    """

    def __init__(self, arm="amplitude", pool_size=None):
        super().__init__()
        assert arm in ("amplitude", "reupload"), f"unknown arm: {arm}"
        self.arm = arm
        self.pool_size = pool_size
        self.vqc = AmplitudeVQC() if arm == "amplitude" else ReuploadVQC()
        self.post = nn.Conv2d(self.vqc.out_features, 3, kernel_size=1)

    def forward(self, x):
        # x: [B, 9, H, W] -- BnInception(12->9)'s output, unchanged upstream
        B, C, H, W = x.shape
        assert C == N_VALUES, f"expected {N_VALUES} input channels, got {C}"

        if self.pool_size is not None:
            x_in = F.adaptive_avg_pool2d(x, self.pool_size)
            h, w = self.pool_size, self.pool_size
        else:
            x_in = x
            h, w = H, W

        flat = x_in.permute(0, 2, 3, 1).reshape(B * h * w, C)     # [N, 9]
        q_out = self.vqc(flat)                                     # [N, vqc.out_features]
        q_out = q_out.reshape(B, h, w, -1).permute(0, 3, 1, 2)      # [B, out_features, h, w]

        if self.pool_size is not None:
            q_out = F.interpolate(q_out, size=(H, W), mode="bilinear", align_corners=True)

        return self.post(q_out)   # [B, 3, H, W] -- matches self.pred's original output shape


# =============================================================================
# INTEGRATION DIFFS (apply by hand when ready -- NOT auto-applied, off by
# default. train_comparison.py's BusterNetWithAux reimplements forward()
# inline and does NOT call model_busternet.py's forward -- needs its own,
# separate, deliberate edit. Do not touch either file while Experiment A/B
# is running on the lab PC.)
# =============================================================================
#
# --- model_busternet.py, BusterNet.__init__ (replacing the self.pred line) ---
#   from vqc_fusion import VQCPred
#   if use_vqc:
#       self.pred = VQCPred(arm=vqc_arm, pool_size=vqc_pool_size)   # new __init__ args
#   else:
#       self.pred = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, padding=1)
#
# --- model_busternet.py, BusterNet.forward -- NO CHANGE NEEDED.
#     `self.pred(fused)` already called generically; VQCPred's forward() has
#     the same [B,9,H,W] -> [B,3,H,W] signature as the Conv2d it replaces.
#
# --- train_comparison.py, BusterNetWithAux -- same swap, applied SEPARATELY,
#     at its own `self.pred = self.busternet.pred` line (~line 94) and
#     wherever it constructs build_busternet_full(...).
