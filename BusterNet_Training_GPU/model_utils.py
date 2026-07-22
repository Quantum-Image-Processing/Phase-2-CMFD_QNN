import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# BnInception class and other decoder classes are defined below


def get_vgg_first4(pretrained=True):
    """
    Loads VGG16 features and returns nn.Sequential up to and including the 4th MaxPool.
    Output channels = 512, spatial downsampling = /16 (for input H,W -> H/16,W/16).
    """
    vgg = models.vgg16(pretrained=pretrained)
    features = vgg.features
    layers = []
    maxpool_count = 0
    for m in features:
        layers.append(m)
        if isinstance(m, nn.MaxPool2d):
            maxpool_count += 1
            if maxpool_count == 4:
                break
    seq = nn.Sequential(*layers)
    return seq


class BnInception(nn.Module):
    def __init__(self, in_ch, nb_inc, kernels):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Conv2d(
                in_ch,
                nb_inc,
                kernel_size=k,
                padding=(k[0] // 2, k[1] // 2),
                bias=True
            )
            for k in kernels
        ])

        self.bn = nn.BatchNorm2d(nb_inc * len(kernels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class SimilarityDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        patch = [(1, 1), (3, 3), (5, 5)]

        self.mpf = BnInception(256, 8, patch)
        self.dx2 = BnInception(24, 6, patch)
        self.dx4 = BnInception(42, 4, patch)
        self.dx8 = BnInception(36, 2, patch)
        self.dx16 = BnInception(30, 2, patch)

        self.dxF = BnInception(
            30,
            2,
            [(5, 5), (7, 7), (11, 11)]
        )

        self.pred = nn.Conv2d(
            6,
            1,
            kernel_size=3,
            padding=1
        )

    def up(self, x):
        return F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

    def forward(self, x):

        # 16×16×256 → 16×16×24
        f16 = self.mpf(x)

        # 16→32
        f32 = self.up(f16)
        dx32 = self.dx2(f32)

        # 32→64
        f64a = self.up(f32)
        f64b = self.up(dx32)
        dx64 = self.dx4(torch.cat([f64a, f64b], dim=1))

        # 64→128
        f128a = self.up(f64a)
        f128b = self.up(dx64)
        dx128 = self.dx8(torch.cat([f128a, f128b], dim=1))

        # 128→256
        f256a = self.up(f128a)
        f256b = self.up(dx128)
        dx256 = self.dx16(torch.cat([f256a, f256b], dim=1))

        # Final fusion
        fm256 = torch.cat([f256a, dx256], dim=1)
        masks = self.dxF(fm256)

        pred = torch.sigmoid(self.pred(masks))
        return masks, pred

class ManiDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        patch = [(1, 1), (3, 3), (5, 5)]

        # 16×16×512 → 16×16×24
        self.mpf = BnInception(512, 8, patch)

        # 16→32
        self.dx2 = BnInception(24, 6, patch)

        # 32→64
        self.dx4 = BnInception(18, 4, patch)

        # 64→128
        self.dx8 = BnInception(12, 2, patch)

        # 128→256
        self.dx16 = BnInception(
            6,
            2,
            [(5, 5), (7, 7), (11, 11)]
        )

    def up(self, x):
        return F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

    def forward(self, x):

        # 16×16×512 → 16×16×24
        x = self.mpf(x)

        # 16→32
        x = self.up(x)
        x = self.dx2(x)

        # 32→64
        x = self.up(x)
        x = self.dx4(x)

        # 64→128
        x = self.up(x)
        x = self.dx8(x)

        # 128→256
        x = self.up(x)
        x = self.dx16(x)

        return x

def std_norm_along_chs(x):
    """
    Per-pixel channel-wise standard normalization.
    Input: [B, C, H, W]
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)  # Match TensorFlow's biased estimator
    std = torch.clamp(std, min=1e-4)
    return (x - mean) / std

def self_corr_and_percentile(x, K=256):
    """
    Self-correlation and percentile pooling operation.

    Args:
      x: Tensor [B, C, H, W] (expected C=512, H=W=16 in this pipeline)
      K: number of pooled positions to keep (default 256)

    Returns:
      S_raw: Tensor raw self-correlation reshaped to [B, H, W, L] where L=H*W
      P: Tensor pooled top-K along last axis with shape [B, H, W, K]

    Notes:
      - This implementation matches TensorFlow's SelfCorrelationPercPooling:
        * self-correlation matrix S = (x·xᵀ) / C
        * sort descending along last dim
        * generate K ranks with linspace(1, L-1, K) (1-based, skipping rank 0)
        * gather those ranks from the sorted tensor
      - If K == L, TensorFlow still performs the linspace/gather (producing duplicates),
        so we do NOT shortcut-return the full sorted list.
    """
    B, C, H, W = x.shape
    L = H * W
    # flatten spatial positions into vectors: [B, L, C]
    x_flat = x.permute(0, 2, 3, 1).reshape(B, L, C)
    # L2 normalize each vector
    S = torch.matmul(x_flat, x_flat.transpose(-1, -2)) / C
    # reshape to [B, H, W, L]
    S_reshaped = S.reshape(B, H, W, L)
    # sort descending along last dim
    S_sorted, _ = torch.sort(S_reshaped, dim=-1, descending=True)
    # generate K ranks: 1, 2, ..., L-1 (TensorFlow uses 1-based ranks, skips rank 0)
    idx = torch.linspace(1, L - 1, steps=K, device=S_sorted.device).round().long()
    P = S_sorted[..., idx]
    # S_reshaped and P are float32 tensors
    return S_reshaped, P

def masked_bce_loss(pred, target, loss_mask):
    """
    Binary Cross Entropy computed only over valid (non-padded) pixels.

    Args:
        pred:      [B,1,H,W] after sigmoid
        target:    [B,1,H,W] values in {0,1}
        loss_mask: [B,1,H,W] values in {0,1}
    """

    # Compute per-pixel BCE (do not reduce yet)
    pixel_loss = F.binary_cross_entropy(
        pred,
        target,
        reduction='none'
    )

    # Ignore padded pixels
    pixel_loss = pixel_loss * loss_mask

    # Number of valid pixels
    valid_pixels = loss_mask.sum()

    # Safety check (should never happen)
    if valid_pixels.item() == 0:
        return pixel_loss.sum() * 0.0

    # Average only over valid pixels
    return pixel_loss.sum() / valid_pixels