"""
BusterNet Dataset using predefined COCO subset CSV splits.
Matches the original dataset.py logic but uses predefined train/val/test splits.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


class BusterNetDataset(Dataset):
    """
    BusterNet Dataset using predefined COCO subset CSV splits.
    
    CSV columns:
    - sample_id, image_id, file_name, mask_file_name, forged_file_name, mt_file_name
    - clone_type, rotation, scale, translation_x, translation_y, overlap_ratio
    - mc_file_name (3-class mask: 0=authentic, 1=source, 2=target)
    - mm_file_name (binary mask: 0=authentic, 1=forged)
    """
    
    def __init__(self, csv_path, data_root, img_size=256, split='train', 
                 use_binary_mask=False, augment=True):
        """
        Args:
            csv_path: Path to CSV manifest (train/val/test)
            data_root: Root directory containing images and masks
            img_size: Target image size (default 256)
            split: 'train', 'val', or 'test'
            use_binary_mask: If True, use binary mask (mm) as target; else use 3-class mask (mc)
            augment: Whether to apply data augmentation (only for train split)
        """
        self.csv_path = csv_path
        self.data_root = data_root
        self.img_size = img_size
        self.split = split
        self.use_binary_mask = use_binary_mask
        self.augment = augment and split == 'train'
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {split} split: {len(self.df)} samples from {csv_path}")
        
        # Image/masks are plain-resized (stretched) to img_size x img_size in
        # __getitem__ -- no letterbox/aspect-ratio-preserving padding. Measured
        # better accuracy this way; loss_mask is therefore all-ones (every pixel
        # is real content, nothing to exclude).

        # Augmentation for training
        if self.augment:
            self.augment_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.augment_transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load forged image
        forged_rel = row['forged_file_name']
        forged_path = os.path.join(self.data_root, forged_rel)
        if not os.path.exists(forged_path):
            forged_path = os.path.join(self.data_root, 'images_forged', forged_rel)
        image_raw = Image.open(forged_path).convert('RGB')
        loss_mask = torch.ones((self.img_size, self.img_size), dtype=torch.float32)

        # Load 3-class mask (mc) - 0=authentic, 1=source, 2=target
        mc_rel = row['mc_file_name']
        mc_path = os.path.join(self.data_root, mc_rel)
        if not os.path.exists(mc_path):
            mc_path = os.path.join(self.data_root, 'masks_3class', os.path.basename(mc_rel))
        mask_3class = Image.open(mc_path)
        
        # Load binary mask (mm) - 0=authentic, 1=forged
        mm_rel = row['mm_file_name']
        mm_path = os.path.join(self.data_root, mm_rel)
        if not os.path.exists(mm_path):
            mm_path = os.path.join(self.data_root, 'masks_binary', os.path.basename(mm_rel))
        mask_binary = Image.open(mm_path)
        
        # Plain resize (stretch to img_size x img_size, no aspect-ratio
        # preservation / no padding).
        image_r = image_raw.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = T.functional.to_tensor(image_r)  # [3, H, W], [0, 1]

        mask_3class = mask_3class.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_binary = mask_binary.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Convert masks to numpy and map {0,127,255} -> {0,1,2}
        m3_np = np.array(mask_3class, dtype=np.int64)
        if set(np.unique(m3_np)).issubset({0, 127, 255}):
            mapping = {0: 0, 127: 1, 255: 2}
            m3_mapped = np.vectorize(lambda v: mapping.get(int(v), 0))(m3_np).astype(np.int64)
        else:
            m3_mapped = m3_np
        mask_3class = torch.from_numpy(m3_mapped)  # [H, W] in {0, 1, 2}

        # Binary mask mapping
        mb_np = np.array(mask_binary, dtype=np.int64)
        mb_mapped = (mb_np > 0).astype(np.int64)
        mask_binary = torch.from_numpy(mb_mapped)  # [H, W] in {0, 1}
        
        # Apply augmentation (same geometric transform for image, masks, AND
        # loss_mask). loss_mask starts all-ones, but rotation can still fill
        # rotated-out corners with 0 -- keeping it in the same combined
        # transform means those corners get correctly excluded from loss too.
        if self.augment_transform is not None:
            combined = torch.cat([
                image,
                mask_3class.unsqueeze(0).float(),
                mask_binary.unsqueeze(0).float(),
                loss_mask.unsqueeze(0),
            ], dim=0)
            # Note: ColorJitter only applies to the image; geometric ops apply to all
            if torch.rand(1) < 0.5:
                combined = T.functional.hflip(combined)
            if torch.rand(1) < 0.5:
                combined = T.functional.vflip(combined)
            if torch.rand(1) < 0.3:
                angle = torch.randint(-15, 16, (1,)).item()
                combined = T.functional.rotate(combined, angle, interpolation=T.InterpolationMode.NEAREST)

            image = combined[0:3]
            mask_3class = combined[3].long()
            mask_binary = combined[4].long()
            loss_mask = combined[5]
        
        # Prepare targets
        # Main target: 3-class (0=authentic, 1=source, 2=target)
        target_main = mask_3class  # [H, W]
        
        # Auxiliary targets (for Stages 1-2): binary masks for each branch
        # ManiDet aux: binary forged mask (target only, per paper)
        target_manidet = (mask_3class == 2).long()  # 1 for target only
        
        # SimiDet aux: similarity detection uses source U target
        target_simidet = (mask_3class > 0).long()
        
        aux_targets = {
            'manidet': target_manidet,
            'simidet': target_simidet
        }
        
        # Main target for training (3-class)
        if self.use_binary_mask:
            # Use binary mask as main target (0=authentic, 1=forged)
            # But model expects 3 classes, so we need to adapt
            target_main = mask_binary
        
        return {
            'image': image,           # [3, H, W], float32 [0, 1]
            'target_main': target_main,      # [H, W], int64 (0,1,2)
            'loss_mask': loss_mask,          # [H, W], float32 1.0/0.0
            'target_aux': aux_targets,       # dict with 'manidet', 'simidet' [H, W]
            'sample_id': row['sample_id'],
            'image_id': row['image_id'],
            'clone_type': row['clone_type'],
        }


def get_dataloaders(data_root, csv_dir, batch_size=8, img_size=256, num_workers=4,
                    use_binary_mask=False, pin_memory=True):
    """
    Create train/val/test dataloaders using predefined CSV splits.
    
    Args:
        data_root: Root directory containing images and masks
        csv_dir: Directory containing the CSV split files
        batch_size: Batch size
        img_size: Image size
        num_workers: DataLoader workers
        use_binary_mask: Use binary mask as main target
        pin_memory: Pin memory for GPU
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_csv = os.path.join(csv_dir, 'forgery_manifest_with_masks_train.csv')
    val_csv = os.path.join(csv_dir, 'forgery_manifest_with_masks_val.csv')
    test_csv = os.path.join(csv_dir, 'forgery_manifest_with_masks_test.csv')
    
    train_dataset = BusterNetDataset(
        train_csv, data_root, img_size=img_size, split='train',
        use_binary_mask=use_binary_mask, augment=True
    )
    val_dataset = BusterNetDataset(
        val_csv, data_root, img_size=img_size, split='val',
        use_binary_mask=use_binary_mask, augment=False
    )
    test_dataset = BusterNetDataset(
        test_csv, data_root, img_size=img_size, split='test',
        use_binary_mask=use_binary_mask, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


class FixedValidationSet:
    """
    Fixed validation set for consistent evaluation across epochs.
    Selects N fixed samples from validation set for prediction saving.
    """
    def __init__(self, val_loader, num_samples=8, device='cuda'):
        self.num_samples = num_samples
        self.device = device
        self.samples = []
        
        # Collect fixed samples
        for batch in val_loader:
            for i in range(min(batch['image'].size(0), num_samples - len(self.samples))):
                sample = {
                    'image': batch['image'][i:i+1].to(device),
                    'target_main': batch['target_main'][i:i+1].to(device),
                    'target_aux': {k: v[i:i+1].to(device) for k, v in batch['target_aux'].items()},
                    'sample_id': batch['sample_id'][i] if isinstance(batch['sample_id'], list) else batch['sample_id'][i].item(),
                    'clone_type': batch['clone_type'][i] if isinstance(batch['clone_type'], list) else batch['clone_type'][i],
                }
                self.samples.append(sample)
            if len(self.samples) >= num_samples:
                break
        
        print(f"Fixed validation set: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)


def compute_f1_score(pred, target, num_classes=3, ignore_index=255):
    """
    Compute per-class F1 and macro F1 score.
    
    Args:
        pred: [B, H, W] predicted class indices
        target: [B, H, W] ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in target
    
    Returns:
        dict with per-class F1 and macro F1
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Mask out ignore_index
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    f1_scores = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        tp = (pred_c & target_c).sum().float()
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()
        
        if tp + fp + fn == 0:
            f1 = torch.tensor(float('nan'))
        else:
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        f1_scores.append(f1)
    
    valid_f1s = [f for f in f1_scores if not torch.isnan(f)]
    macro_f1 = torch.stack(valid_f1s).mean() if valid_f1s else torch.tensor(0.0)
    
    return {
        'f1_per_class': [f.item() if not torch.isnan(f) else float('nan') for f in f1_scores],
        'macro_f1': macro_f1.item()
    }


def save_validation_predictions(model, fixed_val_set, epoch, output_dir, device, 
                                 use_aux_model=False, num_classes=3):
    """
    Save fixed validation predictions for visual comparison across epochs.
    
    Args:
        model: Model in eval mode
        fixed_val_set: FixedValidationSet instance
        epoch: Current epoch number
        output_dir: Directory to save predictions
        device: Device
        use_aux_model: Whether model returns (main_out, aux_out) tuple
        num_classes: Number of classes for main output
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(fixed_val_set):
            image = sample['image']
            target_main = sample['target_main']
            sample_id = sample['sample_id']
            clone_type = sample['clone_type']
            
            if use_aux_model:
                main_out, aux_out = model(image)
            else:
                main_out = model(image)
                aux_out = None
            
            # Get predictions
            pred_main = main_out.argmax(dim=1).cpu().numpy()[0]  # [H, W]
            target_main_np = target_main.cpu().numpy()[0]
            
            # Save visualization
            save_path = os.path.join(output_dir, f'epoch{epoch:03d}_sample{sample_id}_{clone_type}.npz')
            np.savez_compressed(
                save_path,
                pred_main=pred_main,
                target_main=target_main_np,
                image=image.cpu().numpy()[0].transpose(1, 2, 0),  # [H, W, 3]
                clone_type=clone_type
            )
            
            # Also save auxiliary predictions if available
            if aux_out is not None:
                for key in ['manidet', 'simidet']:
                    if key in aux_out:
                        pred_aux = (aux_out[key] > 0.5).float().cpu().numpy()[0, 0]
                        target_aux = sample['target_aux'][key].cpu().numpy()[0]
                        np.savez_compressed(
                            os.path.join(output_dir, f'epoch{epoch:03d}_sample{sample_id}_{key}.npz'),
                            pred=pred_aux,
                            target=target_aux
                        )
    
    model.train()


if __name__ == '__main__':
    # Test dataset loading
    data_root = r'C:\ml_work\synthetic_datasets\experiment_subset'
    csv_dir = r'C:\ml_work\synthetic_datasets\experiment_subset'
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root, csv_dir, batch_size=4, img_size=256, num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Target main shape: {batch['target_main'].shape}")
    print(f"Target main unique: {batch['target_main'].unique()}")
    print(f"Aux targets: {[(k, v.shape) for k, v in batch['target_aux'].items()]}")
    print(f"Sample IDs: {batch['sample_id'][:4]}")
    print(f"Clone types: {batch['clone_type'][:4]}")
    
    # Test fixed validation set
    fixed_val = FixedValidationSet(val_loader, num_samples=5)
    print(f"\nFixed val samples: {len(fixed_val)}")
    for sample in fixed_val:
        print(f"  Sample ID: {sample['sample_id']}, Clone: {sample['clone_type']}")
    
    # Test F1 computation
    pred = torch.randint(0, 3, (2, 256, 256))
    target = torch.randint(0, 3, (2, 256, 256))
    f1_dict = compute_f1_score(pred, target)
    print(f"\nF1 scores: {f1_dict}")
    
    print("\nDataset test passed!")