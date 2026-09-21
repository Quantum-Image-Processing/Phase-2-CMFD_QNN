print("TEST WITH MODEL LOADING START")

print("About to import os")
import os
print("Os imported")

print("About to import torch")
import torch
print("Torch imported")

print("About to import sys and modify path")
import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')
print("Path modified")

print("About to import local modules")
from model_busternet import BusterNet, build_busternet_full
from model_simidet import build_simidet
from model_manidet import build_manidet
from preprocessing import BusterNetPreprocess
from dataset import SyntheticCMFDataset
print("Local modules imported")

print("About to define main")
def main():
    print("MAIN FUNCTION START")
    
    # Paths
    checkpoint_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU\checkpoint_epoch20_expA.pth'
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint path exists: {os.path.exists(checkpoint_path)}")
    
    if os.path.exists(checkpoint_path):
        print("About to load checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded, epoch: {checkpoint.get('epoch', 'unknown')}")
        
        print("About to create model")
        model = build_busternet_full(target_size=256)
        print("Model created")
        
        print("About to load state dict")
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Model loaded successfully")
    else:
        print("Checkpoint file does not exist!")
    
print("About to call main")
main()
print("SCRIPT END")