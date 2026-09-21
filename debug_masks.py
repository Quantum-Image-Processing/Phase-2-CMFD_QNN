import os
import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')
from dataset import SyntheticCMFDataset

print("Testing dataset loading - checking masks...")
manifest_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test\manifest.csv'
root_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test'

# Try to load just the first row to see what's happening
with open(manifest_path, newline='', encoding='utf-8') as f:
    import csv
    reader = csv.DictReader(f)
    rows = [r for r in reader]
    if len(rows) > 0:
        first_row = rows[0]
        print(f"First row file_name: '{first_row.get('file_name', 'NOT FOUND')}'")
        print(f"First row mc_file_name: '{first_row.get('mc_file_name', 'NOT FOUND')}'")
        print(f"First row mm_file_name: '{first_row.get('mm_file_name', 'NOT FOUND')}'")
        
        # Test image path construction (this worked before)
        val = first_row.get('file_name', '').strip()
        print(f"\nImage val: '{val}'")
        if val:
            p3 = os.path.join(root_path, val)
            print(f"Image constructed path p3: '{p3}'")
            print(f"Image path exists: {os.path.exists(p3)}")
        
        # Test mask path construction
        mc_val = first_row.get('mc_file_name', '').strip()
        print(f"\nMC val: '{mc_val}'")
        if mc_val:
            # This is what _find_mask_path does for primary_col='mc_file_name'
            candidates = []
            # If the manifest already includes a relative path from the dataset root
            candidates.append(os.path.join(root_path, mc_val))
            # If it includes a path under a different folder, also try the expected folder
            candidates.append(os.path.join(root_path, 'masks_3class', os.path.basename(mc_val)))
            candidates.append(os.path.join(root_path, os.path.basename(mc_val)))
            
            print(f"MC candidates:")
            for i, p in enumerate(candidates):
                print(f"  {i+1}. '{p}' -> exists: {os.path.exists(p)}")
                if os.path.exists(p):
                    print(f"      FOUND MATCH!")
                    break
        
        # Test mm path construction  
        mm_val = first_row.get('mm_file_name', '').strip()
        print(f"\nMM val: '{mm_val}'")
        if mm_val:
            # This is what _find_mask_path does for primary_col='mm_file_name'
            candidates = []
            # If the manifest already includes a relative path from the dataset root
            candidates.append(os.path.join(root_path, mm_val))
            # If it includes a path under a different folder, also try the expected folder
            candidates.append(os.path.join(root_path, 'masks_binary', os.path.basename(mm_val)))
            candidates.append(os.path.join(root_path, os.path.basename(mm_val)))
            
            print(f"MM candidates:")
            for i, p in enumerate(candidates):
                print(f"  {i+1}. '{p}' -> exists: {os.path.exists(p)}")
                if os.path.exists(p):
                    print(f"      FOUND MATCH!")
                    break