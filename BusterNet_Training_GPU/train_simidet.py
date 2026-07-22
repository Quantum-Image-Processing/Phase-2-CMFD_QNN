import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import SyntheticCMFDataset
from model_simidet import SimiDetBranch
from model_utils import masked_bce_loss
from test_stage_a_infer import load_weights_if_given

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=None, help='Number of samples to overfit on (e.g., 8)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-freq', type=int, default=10, help='Save predictions every N epochs')
    return parser.parse_args()

def save_prediction(pred_tensor, image_tensor, mask_tensor, epoch, idx):
    # Save inputs and predictions for visual verification during overfitting
    out_dir = f'outputs/overfit_sanity/epoch_{epoch}'
    os.makedirs(out_dir, exist_ok=True)
    
    pred_bin = (pred_tensor.squeeze().detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
    gt_mask = (mask_tensor.squeeze().detach().cpu().numpy()).astype(np.uint8) * 255
    img = (image_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    Image.fromarray(img).save(f'{out_dir}/sample_{idx}_img.png')
    Image.fromarray(gt_mask).save(f'{out_dir}/sample_{idx}_gt.png')
    Image.fromarray(pred_bin).save(f'{out_dir}/sample_{idx}_pred.png')

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset
    print("Loading dataset...")
    train_dataset = SyntheticCMFDataset(manifest_csv='coco_subset/forgery_manifest_with_masks.csv', root='coco_subset', img_size=256)
    
    if args.max_samples is not None:
        print(f"Subsetting dataset to first {args.max_samples} samples for overfitting test...")
        indices = list(range(args.max_samples))
        train_dataset = Subset(train_dataset, indices)
        batch_size = min(4, args.max_samples)
    else:
        batch_size = 4
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. Model
    print("Building model...")
    model = SimiDetBranch().to(device)
    
    ckpt_path = 'simidet_vgg_weights_new.pth'
    if os.path.exists(ckpt_path):
        load_weights_if_given(model, ckpt_path, 'SimiDet')
    
    # 3. Freeze parameters (only train auxiliary head)
    for name, param in model.named_parameters():
        if 'decoder.pred' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # 4. Loss and Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # 5. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mm'].to(device).float()
            loss_mask = batch['loss_mask'].to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(images)
            m_pred = outputs['m_pred']
            
            loss = masked_bce_loss(
                pred=m_pred,
                target=masks,
                loss_mask=loss_mask
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Save predictions periodically
            if (epoch + 1) % args.save_freq == 0 and i == 0:
                for b_idx in range(images.shape[0]):
                    save_prediction(m_pred[b_idx], images[b_idx], masks[b_idx], epoch + 1, b_idx)
                    
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
            
    print("Training complete.")

if __name__ == '__main__':
    main()
