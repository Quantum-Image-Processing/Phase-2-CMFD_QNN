import torch
import torch.nn as nn
from model_utils import (
    get_vgg_first4,
    SimilarityDecoder,
    self_corr_and_percentile,
    std_norm_along_chs,
)

class SimiDetBranch(nn.Module):
    def __init__(self, K=256, target_size=256):
        super().__init__()
        # load pretrained ImageNet weights into simi branch backbone by default
        self.backbone = get_vgg_first4(pretrained=True)
        self.K = K
        self.target_size = target_size
        self.corr_bn = nn.BatchNorm2d(256, eps=1e-3)
        self.decoder = SimilarityDecoder()

    def forward(self, x):
        # x: [B,3,target_size,target_size] - already preprocessed (BGR, mean-subtracted)
        f = self.backbone(x)  # [B,512,16,16]
        f = std_norm_along_chs(f)
        S_raw, P = self_corr_and_percentile(f, K=self.K)

        # convert P to channels-first for decoder
        P_ch = P.permute(0, 3, 1, 2).contiguous()

        P_ch = self.corr_bn(P_ch)

        d, m_pred = self.decoder(P_ch)

        return {
            'f': f,
            'S': S_raw,
            'P': P,
            'd': d,
            'm_pred': m_pred
        }


def build_simidet(target_size=256):
    return SimiDetBranch(target_size=target_size)
