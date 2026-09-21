import os
import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')
from dataset import SyntheticCMFDataset

print("Testing dataset loading...")
manifest_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test\manifest.csv'
root_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test'

print(f"Manifest exists: {os.path.exists(manifest_path)}")
print(f"Root exists: {os.path.exists(root_path)}")

# Try to load just the first row to see what's happening
with open(manifest_path, newline='', encoding='utf-8') as f:
    import csv
    reader = csv.DictReader(f)
    rows = [r for r in reader]
    print(f"Total rows: {len(rows)}")
    if len(rows) > 0:
        first_row = rows[0]
        print(f"First row keys: {list(first_row.keys())}")
        print(f"First row file_name: '{first_row.get('file_name', 'NOT FOUND')}'")
        print(f"First row category: '{first_row.get('category', 'NOT FOUND')}'")
        
        # Test the path construction
        val = first_row.get('file_name', '').strip()
        print(f"val: '{val}'")
        if val:
            p3 = os.path.join(root_path, val)
            print(f"Constructed path p3: '{p3}'")
            print(f"Path exists: {os.path.exists(p3)}")
            
            # Also check what's in the images directory
            images_dir = os.path.join(root_path, 'images')
            print(f"Images dir exists: {os.path.exists(images_dir)}")
            if os.path.exists(images_dir):
                files = os.listdir(images_dir)
                print(f"Number of files in images dir: {len(files)}")
                if len(files) > 0:
                    print(f"First file in images dir: {files[0]}")
                    
                    # Check if our file is there
                    expected_file = os.path.basename(val)
                    print(f"Expected file (basename): '{expected_file}'")
                    print(f"Expected file in images dir: {expected_file in files}")