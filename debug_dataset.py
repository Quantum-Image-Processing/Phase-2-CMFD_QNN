import sys
sys.path.append(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\BusterNet_Training_GPU')
from dataset import SyntheticCMFDataset

print("Loading dataset...")
dataset = SyntheticCMFDataset(r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test\manifest.csv', r'C:\Users\dines\OneDrive\Desktop\Phase-1-QNN\combined_dataset\test', 256)
print('Total dataset size:', len(dataset))

# Count categories
categories = {}
for i in range(len(dataset)):
    sample = dataset[i]
    category = sample.get('category', '')
    if category == '':
        category = '(empty/COCO)'
    if category not in categories:
        categories[category] = 0
    categories[category] += 1

print('\nCategory distribution:')
for category, count in categories.items():
    print(f'  {category}: {count}')

# Show first few samples of each category
print('\nFirst few samples by category:')
shown = {}
for i in range(len(dataset)):
    sample = dataset[i]
    category = sample.get('category', '')
    if category == '':
        category = '(empty/COCO)'
    if category not in shown:
        shown[category] = 0
    if shown[category] < 3:  # Show first 3 of each category
        print('  [{}] {}: {}'.format(category, sample['sample_id'], sample['file_name']))
        shown[category] += 1