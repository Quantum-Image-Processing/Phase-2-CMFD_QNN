import os
import glob
import numpy as np
import torch
from PIL import Image
from model_busternet import build_busternet_full
import torchvision.transforms as T

# ----------------------------------------------------------------------
# 1. Setup model
# ----------------------------------------------------------------------
device = torch.device('cpu')
model = build_busternet_full().to(device)
# Load checkpoint allowing missing keys (aux heads)
state = torch.load('busternet_converted.pth', map_location=device)
ret = model.load_state_dict(state, strict=False)
print("Missing keys:")
for k in ret.missing_keys:
    print(k)
print("Unexpected keys:")
for k in ret.unexpected_keys:
    print(k)
model.eval()

# ----------------------------------------------------------------------
# 2. Find one image and its ground‑truth mask
# ----------------------------------------------------------------------
image_dir   = 'coco_subset/images_forged'
mask_dir    = 'coco_subset/masks_3class'

# Collect image paths (png, jpg, jpeg, bmp)
image_paths = (glob.glob(os.path.join(image_dir, '*.png')) +
               glob.glob(os.path.join(image_dir, '*.jpg')) +
               glob.glob(os.path.join(image_dir, '*.jpeg')) +
               glob.glob(os.path.join(image_dir, '*.bmp')))
if not image_paths:
    raise FileNotFoundError('No images found in ' + image_dir)
img_path = image_paths[0]
print('Using image:', img_path)

# Infer mask filename: look for a file in masks_3class that contains the same stem
img_stem = os.path.splitext(os.path.basename(img_path))[0]
mask_candidates = glob.glob(os.path.join(mask_dir, '*' + img_stem + '*'))
if not mask_candidates:
    raise FileNotFoundError('No matching mask found for ' + img_path)
mask_path = mask_candidates[0]
print('Using mask:', mask_path)

# ----------------------------------------------------------------------
# 3. Load and preprocess image
# ----------------------------------------------------------------------
pil_img = Image.open(img_path).convert('RGB')
pil_img = pil_img.resize((256, 256))

# Keras VGG16 preprocessing: RGB -> BGR, subtract ImageNet mean, keep 0-255 range
img_np = np.array(pil_img, dtype=np.float32)          # RGB, 0-255
img_np = img_np[:, :, ::-1].copy()                           # RGB -> BGR
img_np[..., 0] -= 103.939
img_np[..., 1] -= 116.779
img_np[..., 2] -= 123.68
# Convert to tensor: [H,W,3] -> [1,3,256,256]
input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

# ----------------------------------------------------------------------
# 4. Inference
# ----------------------------------------------------------------------
with torch.no_grad():
    output = model(input_tensor)

# The probability map we care about
mc_pred = output['mc_pred']          # [1,3,256,256]
# Convert to class index map (0,1,2) – the mask
pred_mask = mc_pred.argmax(dim=1, keepdim=True).squeeze(0).squeeze(0)  # [256,256]

# ---- Debug prints ----------------------------------------------------
print('torch.unique(predicted_classes):', torch.unique(pred_mask))
print('predicted_classes.min():', pred_mask.min())
print('predicted_classes.max():', pred_mask.max())
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 5. Load ground‑truth mask for reference
# ----------------------------------------------------------------------
gt_pil = Image.open(mask_path).convert('L').resize((256, 256))
gt_mask_np = np.array(gt_pil, dtype=np.uint8)
gt_mask = torch.from_numpy(gt_mask_np)

# ----------------------------------------------------------------------
# 5b. Compute accuracy metrics
# ----------------------------------------------------------------------
# Pixel accuracy
correct = (pred_mask == gt_mask).sum().item()
total = gt_mask.numel()
pixel_acc = correct / total
print(f'Pixel accuracy: {pixel_acc:.4f}')

# Per-class IoU
for cls in range(3):
    pred_cls = (pred_mask == cls)
    gt_cls = (gt_mask == cls)
    intersection = (pred_cls & gt_cls).sum().item()
    union = (pred_cls | gt_cls).sum().item()
    iou = intersection / union if union > 0 else 0.0
    print(f'Class {cls} IoU: {iou:.4f}')

# ----------------------------------------------------------------------
# 6. Save results
# ----------------------------------------------------------------------
out_dir = 'single_image_output'
os.makedirs(out_dir, exist_ok=True)

# Original image
pil_img.save(os.path.join(out_dir, 'original.png'))

# Save raw prediction (class IDs 0,1,2) for metrics
pred_mask_np = pred_mask.cpu().numpy().astype('uint8')
Image.fromarray(pred_mask_np).save(os.path.join(out_dir, 'predicted_mask.png'))

# Create visualization mapping: 0→0, 1→128, 2→255
vis = np.zeros_like(pred_mask_np, dtype=np.uint8)
vis[pred_mask_np == 1] = 128
vis[pred_mask_np == 2] = 255
Image.fromarray(vis).save(os.path.join(out_dir, 'predicted_mask_vis.png'))

# Verify mapping by printing unique values
print('Unique values in raw pred_mask_np:', np.unique(pred_mask_np))
print('Unique values in vis mask:', np.unique(vis))

# Save ground‑truth mask
Image.fromarray(gt_mask.numpy().astype('uint8')).save(os.path.join(out_dir, 'ground_truth_mask.png'))

# ----------------------------------------------------------------------
# 7. Print summary statistics
# ----------------------------------------------------------------------
print('--- Summary ------------------------------------------------')
print('Image filename          :', os.path.basename(img_path))
print('Image size (W×H)        :', pil_img.size)                     # (width, height)
print('Input tensor shape      :', input_tensor.shape)
print('Output dict keys       :', list(output.keys()))
print('Output mc_pred shape   :', mc_pred.shape)

# Class histogram
unique_vals, counts = pred_mask.unique(return_counts=True)
hist = dict(zip(unique_vals.tolist(), counts.tolist()))
print('Predicted class histogram:', hist)

# Confidence metrics
flat_probs = mc_pred.view(-1)
max_conf = flat_probs.max().item()
mean_conf = flat_probs.mean().item()
print('Maximum confidence      :', max_conf)
print('Mean confidence         :', mean_conf)

# Verify that more than one class appears and that IDs are valid (0‑2)
print('Unique predicted classes :', unique_vals.tolist())
print('More than one class?    :', len(unique_vals) > 1)
print('All class IDs valid (0‑2)?:', torch.all(pred_mask < 3) and torch.all(pred_mask >= 0))
print('--- End ---------------------------------------------------')