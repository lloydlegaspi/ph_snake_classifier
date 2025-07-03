import os
import shutil
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_dataset_splits(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the dataset into training, validation, and testing sets with a 70-15-15 split
    for each class to ensure fair representation.
    
    Args:
        dataset_path (str): Path to the main dataset folder
        output_path (str): Path where the split dataset will be saved
        train_ratio (float): Ratio of training set
        val_ratio (float): Ratio of validation set  
        test_ratio (float): Ratio of test set
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
    
    # List all subdirectories in the dataset folder
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    # Process each class separately
    for class_name in tqdm(class_folders, desc="Processing classes"):
        class_path = os.path.join(dataset_path, class_name)
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        # Create class subfolders in each split
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)
        
        # Split the data
        train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=seed)
        
        # The remaining ratio is val_ratio + test_ratio, which is 0.3
        # To get the correct proportions, we adjust the test_size for the second split
        val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)
        
        # Copy files to respective directories
        for file_name in train_files:
            shutil.copy(
                os.path.join(class_path, file_name),
                os.path.join(output_path, 'train', class_name, file_name)
            )
        
        for file_name in val_files:
            shutil.copy(
                os.path.join(class_path, file_name),
                os.path.join(output_path, 'val', class_name, file_name)
            )
            
        for file_name in test_files:
            shutil.copy(
                os.path.join(class_path, file_name),
                os.path.join(output_path, 'test', class_name, file_name)
            )
            
        print(f"{class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

def verify_splits(output_path):
    """
    Verify that the dataset was split correctly
    
    Args:
        output_path (str): Path to the split dataset
    """
    splits = ['train', 'val', 'test']
    all_counts = {}
    
    for split in splits:
        split_path = os.path.join(output_path, split)
        class_folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
        
        split_counts = {}
        for class_name in class_folders:
            class_path = os.path.join(split_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            split_counts[class_name] = len(image_files)
            
        all_counts[split] = split_counts
    
    # Print summary
    print("\nDataset Split Summary:")
    print("-" * 60)
    print(f"{'Class':<20} {'Train':<10} {'Validation':<10} {'Test':<10} {'Total':<10} {'Train %':<10} {'Val %':<10} {'Test %':<10}")
    print("-" * 60)
    
    for class_name in all_counts['train'].keys():
        train_count = all_counts['train'].get(class_name, 0)
        val_count = all_counts['val'].get(class_name, 0)
        test_count = all_counts['test'].get(class_name, 0)
        total = train_count + val_count + test_count
        
        train_pct = train_count / total * 100 if total > 0 else 0
        val_pct = val_count / total * 100 if total > 0 else 0
        test_pct = test_count / total * 100 if total > 0 else 0
        
        print(f"{class_name:<20} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10} {train_pct:.1f}%{'':<5} {val_pct:.1f}%{'':<5} {test_pct:.1f}%")
    
    # Calculate totals
    train_total = sum(all_counts['train'].values())
    val_total = sum(all_counts['val'].values())
    test_total = sum(all_counts['test'].values())
    grand_total = train_total + val_total + test_total
    
    train_pct = train_total / grand_total * 100 if grand_total > 0 else 0
    val_pct = val_total / grand_total * 100 if grand_total > 0 else 0
    test_pct = test_total / grand_total * 100 if grand_total > 0 else 0
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {train_total:<10} {val_total:<10} {test_total:<10} {grand_total:<10} {train_pct:.1f}%{'':<5} {val_pct:.1f}%{'':<5} {test_pct:.1f}%")
    
if __name__ == "__main__":
    # Define paths
    dataset_path = "SnakeDataset"
    balanced_output_path = "BalancedSnakeDataset"
    split_output_path = "SplitSnakeDataset"
    preprocessed_output_path = "PreprocessedSnakeDataset"
    
    # Split dataset
    print("\nSplitting dataset into train, validation, and test sets...")
    create_dataset_splits(balanced_output_path, split_output_path)
    
    # Verify splits
    verify_splits(split_output_path)