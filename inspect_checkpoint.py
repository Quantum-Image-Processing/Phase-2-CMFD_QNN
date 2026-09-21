import torch

checkpoint_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU\checkpoint_epoch20_expA.pth'
print(f"Loading checkpoint from: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint loaded successfully")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print(f"Model state dict keys (first 10): {list(model_state.keys())[:10]}")
        print(f"Model state dict has {len(model_state)} keys")
    else:
        print("'model_state_dict' not found in checkpoint")
        print(f"Available keys: {list(checkpoint.keys())}")
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()