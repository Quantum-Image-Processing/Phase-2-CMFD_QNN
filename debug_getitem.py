import os
import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')
from dataset import SyntheticCMFDataset

print("Testing dataset __getitem__...")
manifest_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test\manifest.csv'
root_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test'

try:
    print("Creating dataset...")
    dataset = SyntheticCMFDataset(manifest_csv=manifest_path, root=root_path, img_size=256)
    print(f"Dataset created successfully. Length: {len(dataset)}")
    
    print("Trying to get first item...")
    item = dataset[0]
    print(f"First item retrieved successfully!")
    print(f"Item keys: {list(item.keys())}")
    print(f"Image shape: {item['image'].shape}")
    print(f"MC shape: {item['mc'].shape}")
    print(f"MM shape: {item['mm'].shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()