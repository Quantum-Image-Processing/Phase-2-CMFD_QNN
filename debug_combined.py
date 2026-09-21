import os
import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')
from dataset import SyntheticCMFDataset
from torch.utils.data import Dataset

print("Testing CombinedDataset...")
manifest_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test\manifest.csv'
root_path = r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test'

class CombinedDataset(Dataset):
    """Wrapper around SyntheticCMFDataset with category filtering."""
    def __init__(self, manifest_csv, root_dir, img_size=256, category=None):
        print(f"CombinedDataset init: manifest={manifest_csv}, root={root_dir}, category={category}")
        self.base_dataset = SyntheticCMFDataset(manifest_csv=manifest_csv, root=root_dir, img_size=img_size)
        self.category = category
        
        if category:
            # For non-COCO categories, match exact category name
            candidate_indices = [
                i for i, row in enumerate(self.base_dataset.rows)
                if row.get('category') == category
            ]
            print(f"Looking for category '{category}' - found {len(candidate_indices)} candidates")
        else:
            # COCO subset has empty category (two consecutive commas in CSV)
            candidate_indices = [
                i for i, row in enumerate(self.base_dataset.rows)
                if not row.get('category') or row.get('category') == ''
            ]
            print(f"Looking for COCO (empty category) - found {len(candidate_indices)} candidates")
        
        # Verify files exist
        self.indices = []
        skipped = 0
        skipped_details = []
        for i in candidate_indices:
            try:
                item = self.base_dataset[i]
                self.indices.append(i)
            except FileNotFoundError as e:
                skipped += 1
                if skipped <= 5:  # Only show first 5 skipped items for debugging
                    row = self.base_dataset.rows[i]
                    skipped_details.append(f"Index {i}: {row.get('sample_id', 'NO_ID')} - {str(e)[:100]}")
                pass  # Skip missing files
            except Exception as e:
                skipped += 1
                if skipped <= 5:  # Only show first 5 skipped items for debugging
                    row = self.base_dataset.rows[i]
                    skipped_details.append(f"Index {i}: {row.get('sample_id', 'NO_ID')} - {type(e).__name__}: {str(e)[:100]}")
                pass  # Skip any other errors
        
        cat_str = f" ({category})" if category else " (COCO)"
        print(f"Loaded {len(self.indices)} valid samples from {manifest_csv}{cat_str} (skipped {skipped} missing)")
        if skipped_details:
            print("Skipped samples details:")
            for detail in skipped_details:
                print(f"  {detail}")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

# Test with COCO (empty category)
try:
    print("\n=== Testing COCO dataset ===")
    coco_dataset = CombinedDataset(manifest_path, root_path, img_size=256, category='coco')
    print(f"COCO dataset length: {len(coco_dataset)}")
    if len(coco_dataset) > 0:
        print("Trying to get first item...")
        item = coco_dataset[0]
        print(f"First COCO item retrieved successfully!")
except Exception as e:
    print(f"Error in COCO dataset: {e}")
    import traceback
    traceback.print_exc()

# Test with defacto
try:
    print("\n=== Testing DEFACTO dataset ===")
    defacto_dataset = CombinedDataset(manifest_path, root_path, img_size=256, category='defacto')
    print(f"DEFACTO dataset length: {len(defacto_dataset)}")
    if len(defacto_dataset) > 0:
        print("Trying to get first item...")
        item = defacto_dataset[0]
        print(f"First DEFACTO item retrieved successfully!")
except Exception as e:
    print(f"Error in DEFACTO dataset: {e}")
    import traceback
    traceback.print_exc()