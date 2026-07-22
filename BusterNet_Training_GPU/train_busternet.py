import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SyntheticCMFDataset
from model_busternet import BusterNet, build_busternet_full
from model_simidet import SimiDetBranch
from model_manidet import ManiDetBranch
from model_utils import masked_bce_loss

# Helper function for masked cross-entropy loss using logits
def masked_cross_entropy_loss(logits, target, mask):
    """
    logits: [B, C, H, W] raw logits
    target: [B, H, W] long tensor with class indices
    mask:   [B, 1, H, W] or [B, H, W] float mask with 1 for valid pixels, 0 for ignored
    """
    # Compute per-pixel cross entropy loss without reduction
    loss = F.cross_entropy(logits, target, reduction='none')  # [B, H, W]
    # Ensure mask has same shape as loss
    if mask.dim() == 4:
        mask = mask.squeeze(1)  # [B, H, W]
    loss = loss * mask
    # Avoid division by zero
    mask_sum = mask.sum()
    if mask_sum.item() == 0:
        return loss.sum() * 0.0
    return loss.sum() / mask_sum