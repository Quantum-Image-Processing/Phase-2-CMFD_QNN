import torch
import torch.nn as nn
from model_utils import get_vgg_first4, ManiDecoder


class ManiDetBranch(nn.Module):
    def __init__(self, target_size=256):
        super().__init__()
        self.target_size = target_size
        # paper: ManiDet backbone is VGG first4 but we do not load pretrained here
        self.backbone = get_vgg_first4(pretrained=False)
        # in_ch = 512 from VGG first4
        self.decoder = ManiDecoder()
        self.aux_head = nn.Conv2d(6, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B,3,target_size,target_size], BGR mean-subtracted
        f = self.backbone(x)  # expected [B,512,16,16]
        d = self.decoder(f)   # expected [B,6,target_size,target_size]
        m_logits = self.aux_head(d)
        m_pred = torch.sigmoid(m_logits)
        return {'f': f, 'd': d, 'm_pred': m_pred}


def build_manidet(target_size=256):
    return ManiDetBranch(target_size=target_size)
