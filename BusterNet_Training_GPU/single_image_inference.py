import os
import glob
import torch
import numpy as np
from PIL import Image
from model_busternet import build_busternet_full
from preprocessing import letterbox_resize, letterbox_resize_mask

# ----------------------------------------------------------------------
# 1. Setup
# ----------------------------------------------------------------------
device = torch.device('cpu')
model = build_busternet_full().to(device)
model.load_state_dict(torch.load('busternet_converted.pth', map_location=device), strict=False)
model.eval()

# ----------------------------------------------------------------------
# 2. Locate a single image and its ground-truth mask
# ----------------------------------------------------------------------
image_dir   = 'coco_subset/images_forged'
mask_dir    = 'coco_subset/masks_3class'

png_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
jpg_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
jpeg_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpeg')))
bmp_paths = sorted(glob.glob(os.path.join(image_dir, '*.bmp')))
image_paths = png_paths + jpg_paths + jpeg_paths + bmp_paths

if not image_paths:
    raise FileNotFoundError('No image found in ' + image_dir)

img_path = image_paths[0]
print('Using image:', img_path)

mask_path = os.path.join(mask_dir, os.path.basename(img_path))
if not os.path.isfile(mask_path):
    raise FileNotFoundError('Mask not found for ' + img_path)

# ----------------------------------------------------------------------
# 3. Load and preprocess image: letterbox + convert to float32 [0,1] RGB
#    Model's BusterNetPreprocess layer handles: x255, RGB->BGR, VGG mean subtraction
# ----------------------------------------------------------------------
pil_img = Image.open(img_path).convert('RGB')
orig_w, orig_h = pil_img.size

# Letterbox resize
pil_img_lb = letterbox_resize(pil_img, 256)

# Convert to float32 tensor in [0,1] RGB (model preprocess layer expects this)
input_tensor = torch.from_numpy(np.array(pil_img_lb, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,256,256] float32 [0,1]

# ----------------------------------------------------------------------
# 4. Inference
# ----------------------------------------------------------------------
with torch.no_grad():
    output = model(input_tensor)

# Probability map of interest
mc_pred = output['mc_pred']          # shape [1,3,256,256]

# Predicted class indices (0,1,2) - the mask
pred_mask = mc_pred.argmax(dim=1, keepdim=True).squeeze(0).squeeze(0)  # [256,256]

# ----------------------------------------------------------------------
# 5. Load ground-truth mask (for reference) - apply same letterbox
# ----------------------------------------------------------------------
gt_mask_pil = Image.open(mask_path).convert('L')
gt_mask_lb = letterbox_resize_mask(gt_mask_pil, 256, fill=0)
gt_mask_np = torch.from_numpy(np.array(gt_mask_lb, dtype=np.int64)).long()

# Map GT values if needed (0, 127, 255 -> 0, 1, 2)
gt_unique = torch.unique(gt_mask_np)
if set(gt_unique.tolist()).issubset({0, 127, 255}):
    mapping = {0: 0, 127: 1, 255: 2}
    gt_mask_np = torch.tensor([mapping.get(int(v), 0) for v in gt_mask_np.flatten()]).reshape(gt_mask_np.shape)

# ----------------------------------------------------------------------
# 6. Save results
# ----------------------------------------------------------------------
out_dir = 'single_image_output'
os.makedirs(out_dir, exist_ok=True)

# Save original image (letterboxed)
pil_img_lb.save(os.path.join(out_dir, 'original.png'))

# Save predicted mask (apply simple palette for visibility)
palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # red, green, blue
pred_pil = Image.fromarray(pred_mask.byte().cpu().numpy()).convert('RGB')
# Simple palette mapping: each class gets a distinct color
palette_flat = []
for i in range(3):
    palette_flat.extend(palette[i])
pred_pil.putpalette(palette_flat)
pred_pil.save(os.path.join(out_dir, 'predicted_mask.png'))

# Save ground-truth mask
Image.fromarray(gt_mask_np.byte().numpy()).save(os.path.join(out_dir, 'ground_truth_mask.png'))

# ----------------------------------------------------------------------
# 7. Print summary statistics
# ----------------------------------------------------------------------
print('--- Summary ------------------------------------------------')
print('Image filename          :', os.path.basename(img_path))
print('Image size (W×H)        :', (orig_w, orig_h))
print('Input tensor shape      :', input_tensor.shape)
print('Output dict keys       :', output.keys())
print('Output mc_pred shape   :', mc_pred.shape)

# Class histogram
unique_vals, counts = pred_mask.unique(return_counts=True)
hist = dict(zip(unique_vals.tolist(), counts.tolist()))
print('Predicted class histogram:', hist)

# Confidence metrics
prob_flat = mc_pred.view(-1)
max_conf = prob_flat.max().item()
mean_conf = prob_flat.mean().item()
print('Maximum confidence      :', max_conf)
print('Mean confidence         :', mean_conf)

# Verify that not all pixels belong to a single class
print('Unique predicted classes :', unique_vals.tolist())
print('More than one class?    :', len(unique_vals) > 1)
print('All class IDs valid (0-2)?:', torch.all(pred_mask < 3) and torch.all(pred_mask >= 0))

# Compare with GT
if gt_mask_np.shape == pred_mask.shape:
    acc = (pred_mask.cpu() == gt_mask_np).float().mean().item()
    print('Pixel accuracy vs GT    :', acc)
print('--- End ---------------------------------------------------')