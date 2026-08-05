import os
import csv
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as TF


class SyntheticCMFDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset reader for CMF-style manifests.

    Returns dict per sample:
      image: FloatTensor [3, img_size, img_size] in [0,1] RGB (NOT preprocessed for VGG)
      mc: LongTensor [1, img_size, img_size] values {0,1,2}
      mm: UInt8/FloatTensor [1, img_size, img_size] values {0,1}
      simi_binary: FloatTensor [1, img_size, img_size]
      loss_mask: FloatTensor [1, img_size, img_size], all ones (plain resize, no padding)
      meta: dict(original_size=(W,H), manifest_row)

    Image/masks are plain-resized (stretched) to img_size x img_size, not
    letterboxed -- measured better accuracy this way than aspect-ratio-
    preserving resize + padding.
    """

    def __init__(self, manifest_csv='coco_subset/forgery_manifest_with_masks.csv', root='coco_subset', img_size=256):
        self.manifest_csv = manifest_csv
        self.root = root
        self.img_size = img_size

        if not os.path.exists(self.manifest_csv):
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv}")

        with open(self.manifest_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.rows = [r for r in reader]

        # common candidate columns for image filenames
        self.candidate_cols = ['forged_file_name', 'forged_image_path', 'image_id', 'file_name', 'image']
        self.mask_candidate_cols = ['mc_file_name', 'mm_file_name', 'mask_file_name', 'mask_path', 'mask_name']

    def __len__(self):
        return len(self.rows)

    def _find_image_path(self, row):
        # Prefer images_forged folder when available
        for col in self.candidate_cols:
            if col in row and row[col].strip():
                val = row[col].strip()
                base = os.path.basename(val)
                # try images_forged then images
                p1 = os.path.join(self.root, 'images_forged', base)
                p2 = os.path.join(self.root, 'images', base)
                if os.path.exists(p1):
                    return p1
                if os.path.exists(p2):
                    return p2
                # if the csv already contains a path relative to root
                p3 = os.path.join(self.root, val)
                if os.path.exists(p3):
                    return p3
        # fallback: try to scan images_forged/images for first matching basename in row values
        for v in row.values():
            if not v:
                continue
            base = os.path.basename(v)
            for folder in ('images_forged', 'images'):
                p = os.path.join(self.root, folder, base)
                if os.path.exists(p):
                    return p
        raise FileNotFoundError(f"Could not find image for manifest row: {row}")

    def _find_mask_path(self, row, folder, primary_col=None, fallback_name=None):
        # Gather column values to check, prioritizing primary_col
        vals_to_check = []
        if primary_col and primary_col in row and row[primary_col] and str(row[primary_col]).strip():
            vals_to_check.append(str(row[primary_col]).strip())

        for col in self.mask_candidate_cols:
            if col != primary_col and col in row and row[col] and str(row[col]).strip():
                vals_to_check.append(str(row[col]).strip())

        for val in vals_to_check:
            candidates = []
            # If the manifest already includes a relative path from the dataset root
            candidates.append(os.path.join(self.root, val))
            # If it includes a path under a different folder, also try the expected folder
            if folder:
                candidates.append(os.path.join(self.root, folder, os.path.basename(val)))
            candidates.append(os.path.join(self.root, os.path.basename(val)))
            for p in candidates:
                if os.path.exists(p):
                    return p

        # Fallback: try the image basename directly under the expected folder
        if fallback_name:
            p = os.path.join(self.root, folder, fallback_name)
            if os.path.exists(p):
                return p
            p = os.path.join(self.root, fallback_name)
            if os.path.exists(p):
                return p

        raise FileNotFoundError(f"Mask not found for folder {folder} in manifest row: {row}")

    def _load_mask(self, p, convert_mode='L'):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Mask not found: {p}")
        m = Image.open(p).convert(convert_mode)
        return m

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = self._find_image_path(row)
        basename = os.path.basename(img_path)

        # load image
        img = Image.open(img_path).convert('RGB')

        # load masks
        # mc (3-class) in masks_3class, mm (binary) in masks_binary
        mc_path = self._find_mask_path(row, 'masks_3class', primary_col='mc_file_name', fallback_name=basename)
        mm_path = self._find_mask_path(row, 'masks_binary', primary_col='mm_file_name', fallback_name=basename)
        mc_img = self._load_mask(mc_path, convert_mode='L')
        mm_img = self._load_mask(mm_path, convert_mode='L')

        # Plain resize (stretch) for image and both masks -- no letterbox/padding
        original_size = img.size  # (W, H)
        img_r = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mc_r = mc_img.resize((self.img_size, self.img_size), Image.NEAREST)
        mm_r = mm_img.resize((self.img_size, self.img_size), Image.NEAREST)

        # No padding -- every pixel is real content, nothing to exclude from loss
        loss_mask = torch.ones((self.img_size, self.img_size), dtype=torch.float32)

        # image to tensor [C,H,W] float32 in [0,1] (NOT preprocessed for VGG - model handles that)
        image_tensor = TF.to_tensor(img_r).type(torch.float32)  # [0,1] RGB

        # mc: to integer labels {0,1,2}
        mc_np = np.array(mc_r, dtype=np.int64)
        # if mask uses palette or other values, attempt to map common patterns
        mc_unique = np.unique(mc_np)
        if set(mc_unique).issubset({0, 127, 255}):
            mapping = {0: 0, 127: 1, 255: 2}
            mc_np = np.vectorize(lambda v: mapping.get(int(v), 0))(mc_np)
        elif set(mc_unique).issubset({0, 1, 2}):
            pass
        else:
            raise ValueError(f"Unexpected mask values: {mc_unique}")

        mc_tensor = torch.from_numpy(mc_np.astype(np.int64))[None, ...]  # [1,H,W]

        # mm: binary {0,1}
        mm_np = np.array(mm_r, dtype=np.uint8)
        mm_np = (mm_np > 0).astype(np.uint8)
        mm_tensor = torch.from_numpy(mm_np.astype(np.uint8))[None, ...]

        simi_binary = (mc_tensor != 0).float()

        meta = {
            'original_size': original_size,  # (W,H)
            'manifest_row': row,
        }

        return {
            'image': image_tensor,  # [3,H,W] in [0,1] RGB
            'mc': mc_tensor,        # [1,H,W]
            'mm': mm_tensor.float(), # [1,H,W]
            'simi_binary': simi_binary, # [1,H,W]
            'loss_mask': loss_mask[None, ...].float(), # [1,H,W]
            'meta': meta,
        }
