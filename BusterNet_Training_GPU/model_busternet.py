import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import BnInception
from preprocessing import BusterNetPreprocess

class BusterNet(nn.Module):
    def __init__(self, simidet, manidet, target_size: int = 256):
        super().__init__()
        self.simidet = simidet
        self.manidet = manidet
        self.target_size = target_size

        # Preprocessing layer (matches TF Preprocess layer)
        self.preprocess_layer = BusterNetPreprocess(target_size)

        # Fusion block: BnInception with nb_inc=3
        # Our branches output 6 channels each (6 + 6 = 12 total)
        self.fusion = BnInception(
            in_ch=12,
            nb_inc=3,
            kernels=[(1, 1), (3, 3), (5, 5)]
        )

        # Prediction layer: 9 -> 3 channels
        self.pred = nn.Conv2d(
            in_channels=9,
            out_channels=3,
            kernel_size=3,
            padding=1
        )

    def preprocess(self, x):
        """Preprocess input using BusterNetPreprocess layer."""
        return self.preprocess_layer(x)

    def forward(self, x):
        # Save original size
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Preprocess
        x = self.preprocess(x)

        # Extract features from both branches
        simi_out = self.simidet(x)
        mani_out = self.manidet(x)

        simi_feat = simi_out['d']  # [B, 6, 256, 256]
        mani_feat = mani_out['d']  # [B, 6, 256, 256]

        # Concatenate: 6 + 6 = 12 channels
        fused = torch.cat([simi_feat, mani_feat], dim=1)

        # Fusion BN-Inception: 12 -> 9 channels
        fused = self.fusion(fused)

        # Prediction: 9 -> 3 channels
        pred = self.pred(fused)

        # Softmax (not sigmoid)
        pred = F.softmax(pred, dim=1)

        # Resize back to original resolution
        if orig_h != 256 or orig_w != 256:
            pred = F.interpolate(
                pred,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )

        return {
            'mc_pred': pred,
            'simi_feat': simi_feat,
            'mani_feat': mani_feat
        }


def build_busternet(simidet, manimet):
    return BusterNet(simidet, manimet)


def build_busternet_full(target_size=256):
    from model_simidet import build_simidet
    from model_manidet import build_manidet
    return BusterNet(build_simidet(), build_manidet(), target_size=target_size)