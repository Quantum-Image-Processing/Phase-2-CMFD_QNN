"""
Unified preprocessing module for BusterNet.
Matches TensorFlow Preprocess layer exactly (BusterNetCore.py lines 156-190).

Both TF and PyTorch should use this logic for preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np


# ImageNet BGR mean (from keras.applications.vgg16.preprocess_input)
IMAGENET_MEAN_BGR = torch.tensor([103.939, 116.779, 123.68], dtype=torch.float32)


def letterbox_resize(image: Image.Image, target_size: int = 256, fill: int = 0) -> Image.Image:
    """
    Resize image preserving aspect ratio with padding (letterbox).
    
    Args:
        image: PIL Image (RGB)
        target_size: Target size (square)
        fill: Padding color (0 for black)
    
    Returns:
        Letterboxed PIL Image of size (target_size, target_size)
    """
    w0, h0 = image.size
    scale = min(target_size / w0, target_size / h0)
    nw = int(round(w0 * scale))
    nh = int(round(h0 * scale))
    img_resized = image.resize((nw, nh), resample=Image.BILINEAR)
    new_img = Image.new(image.mode, (target_size, target_size), color=fill)
    left = (target_size - nw) // 2
    top = (target_size - nh) // 2
    new_img.paste(img_resized, (left, top))
    return new_img


def letterbox_resize_mask(mask: Image.Image, target_size: int = 256, fill: int = 0) -> Image.Image:
    """
    Resize mask preserving aspect ratio with padding (letterbox), using nearest neighbor.
    """
    w0, h0 = mask.size
    scale = min(target_size / w0, target_size / h0)
    nw = int(round(w0 * scale))
    nh = int(round(h0 * scale))
    mask_resized = mask.resize((nw, nh), resample=Image.NEAREST)
    new_mask = Image.new(mask.mode, (target_size, target_size), color=fill)
    left = (target_size - nw) // 2
    top = (target_size - nh) // 2
    new_mask.paste(mask_resized, (left, top))
    return new_mask


def get_letterbox_params(image: Image.Image, target_size: int = 256):
    """Get letterbox parameters for loss mask creation."""
    w0, h0 = image.size
    scale = min(target_size / w0, target_size / h0)
    nw = int(round(w0 * scale))
    nh = int(round(h0 * scale))
    left = (target_size - nw) // 2
    top = (target_size - nh) // 2
    return left, top, nw, nh, (w0, h0)


def create_loss_mask(target_size: int, left: int, top: int, nw: int, nh: int) -> torch.Tensor:
    """Create loss mask: 1 for valid pixels, 0 for padded."""
    loss_mask = torch.zeros((target_size, target_size), dtype=torch.float32)
    loss_mask[top:top+nh, left:left+nw] = 1.0
    return loss_mask


def preprocess_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply VGG preprocessing to image tensor.
    
    Args:
        image_tensor: [3, H, W] or [B, 3, H, W] in RGB, range [0, 255], dtype float32
    
    Returns:
        Preprocessed tensor in BGR, mean-subtracted, range [-123, 136], dtype float32
    """
    # Handle both [3, H, W] and [B, 3, H, W]
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # RGB -> BGR
    image_tensor = image_tensor[:, [2, 1, 0], :, :]
    
    # Subtract ImageNet BGR mean
    mean_bgr = IMAGENET_MEAN_BGR.view(1, 3, 1, 1).to(image_tensor.device)
    image_tensor = image_tensor - mean_bgr
    
    if squeeze:
        image_tensor = image_tensor.squeeze(0)
    
    return image_tensor


def preprocess_image_pil(image: Image.Image, target_size: int = 256) -> torch.Tensor:
    """
    Full preprocessing pipeline for a PIL image (matches TF Preprocess layer).
    
    Args:
        image: PIL Image (RGB)
        target_size: Target size
    
    Returns:
        Tensor [3, target_size, target_size] in BGR, mean-subtracted, float32
    """
    # Letterbox resize
    image_lb = letterbox_resize(image, target_size)
    
    # Convert to tensor [0, 1] -> scale to [0, 255]
    image_tensor = TF.to_tensor(image_lb).type(torch.float32) * 255.0
    
    # VGG preprocessing: RGB -> BGR, subtract mean
    image_tensor = preprocess_image_tensor(image_tensor)
    
    return image_tensor


class BusterNetPreprocess(nn.Module):
    """
    PyTorch module matching TensorFlow Preprocess layer exactly.
    
    TF Preprocess layer (BusterNetCore.py lines 156-190):
    1. If input not 256x256: resize to 256x256 (bilinear, align_corners=True)
    2. If dtype == float32: assume already normalized, return as-is
    3. If dtype == uint8 (RGB): cast to float32, RGB->BGR, subtract BGR mean
    
    This module expects input as:
    - uint8 [0,255] RGB (from PIL) -> applies full preprocessing
    - float32 [0,1] RGB (from dataset) -> applies full preprocessing (×255, RGB->BGR, subtract mean)
    - float32 already preprocessed (BGR, mean-subtracted) -> only resizes if needed
    """
    
    def __init__(self, target_size: int = 256):
        super().__init__()
        self.target_size = target_size
        self.register_buffer('imagenet_mean_bgr', IMAGENET_MEAN_BGR.view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] or [3, H, W] - uint8 or float32
               If uint8: assumed RGB [0,255]
               If float32 and max <= 1.0: assumed RGB [0,1] -> applies full preprocessing
               If float32 and max > 1.0: assumed already preprocessed (BGR, mean-subtracted)
        
        Returns:
            [B, 3, 256, 256] preprocessed tensor
        """
        # Handle [3, H, W] -> [1, 3, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        b, c, h, w = x.shape
        
        # Resize if needed (bilinear, align_corners=True matches TF)
        if h != self.target_size or w != self.target_size:
            x = F.interpolate(
                x, 
                size=(self.target_size, self.target_size), 
                mode='bilinear', 
                align_corners=True
            )
        
        # Apply VGG preprocessing based on dtype and value range
        if x.dtype == torch.uint8:
            # uint8 RGB [0,255] -> float32, RGB->BGR, subtract mean
            x = x.to(torch.float32)
            x = x[:, [2, 1, 0], :, :]
            x = x - self.imagenet_mean_bgr.to(x.device)
        elif x.dtype == torch.float32:
            # Check if already preprocessed (values typically outside [0,1] after mean subtraction)
            # or if it's [0,1] RGB that needs preprocessing
            if x.max() <= 1.0:
                # float32 RGB [0,1] -> scale to [0,255], RGB->BGR, subtract mean
                x = x * 255.0
                x = x[:, [2, 1, 0], :, :]
                x = x - self.imagenet_mean_bgr.to(x.device)
            # else: float32 already preprocessed (BGR, mean-subtracted) -> pass through
        
        if squeeze:
            x = x.squeeze(0)
        
        return x


def preprocess_batch(images: torch.Tensor, target_size: int = 256) -> torch.Tensor:
    """
    Preprocess a batch of images (for inference scripts).
    
    Args:
        images: [B, 3, H, W] or [B, H, W, 3] - uint8 RGB [0,255] or float32 preprocessed
        target_size: Target size
    
    Returns:
        [B, 3, target_size, target_size] preprocessed
    """
    # Handle NHWC -> NCHW
    if images.dim() == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2).contiguous()
    
    preprocess = BusterNetPreprocess(target_size)
    return preprocess(images)


# Test function
if __name__ == "__main__":
    # Test with random image
    test_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8), 'RGB')
    
    print("Testing preprocessing module...")
    
    # Test letterbox
    lb_img = letterbox_resize(test_img, 256)
    print(f"Letterbox: {test_img.size} -> {lb_img.size}")
    
    # Test full preprocessing
    processed = preprocess_image_pil(test_img, 256)
    print(f"Processed tensor: shape={processed.shape}, dtype={processed.dtype}")
    print(f"Range: [{processed.min():.2f}, {processed.max():.2f}]")
    print(f"Mean per channel (BGR): {processed.mean(dim=(1,2))}")
    
    # Test module
    preprocess_module = BusterNetPreprocess(256)
    
    # Test with uint8 input
    uint8_tensor = torch.from_numpy(np.array(test_img)).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    out_uint8 = preprocess_module(uint8_tensor)
    print(f"\nModule (uint8 input): shape={out_uint8.shape}, range=[{out_uint8.min():.2f}, {out_uint8.max():.2f}]")
    
    # Test with float32 input (already preprocessed)
    float32_tensor = processed.unsqueeze(0)
    out_float32 = preprocess_module(float32_tensor)
    print(f"Module (float32 input): shape={out_float32.shape}, range=[{out_float32.min():.2f}, {out_float32.max():.2f}]")
    print(f"Match: {torch.allclose(out_uint8, out_float32, atol=1e-4)}")