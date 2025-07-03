import os
import random
import shutil
from tqdm import tqdm
import numpy as np

def create_balanced_dataset(source_path, output_path, images_per_class=100, seed=42):
    """
    Randomly select a specified number of images per species from the source dataset.
    
    Args:
        source_path (str): Path to the original dataset folder
        output_path (str): Path where the balanced dataset will be saved
        images_per_class (int): Number of images to select for each class
        seed (int): Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # List all subdirectories (species) in the dataset folder
    class_folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
    
    print(f"Found {len(class_folders)} species in the dataset.")
    
    # Process each class separately
    for class_name in tqdm(class_folders, desc="Creating balanced dataset"):
        class_path = os.path.join(source_path, class_name)
        output_class_path = os.path.join(output_path, class_name)
        
        # Create class subfolder in output directory
        os.makedirs(output_class_path, exist_ok=True)
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        # Randomly select images_per_class images (or all if fewer than that)
        selected_files = image_files
        if len(image_files) > images_per_class:
            selected_files = random.sample(image_files, images_per_class)
        
        # Copy selected files to output directory
        for file_name in selected_files:
            shutil.copy(
                os.path.join(class_path, file_name),
                os.path.join(output_class_path, file_name)
            )
            
        print(f"{class_name}: Selected {len(selected_files)}/{len(image_files)} images")
    
    return output_path

if __name__ == "__main__":
    # Define paths
    source_dataset_path = "SnakeDataset"
    balanced_dataset_path = "BalancedSnakeDataset"
    split_output_path = "SplitBalancedDataset"
    
    # Create balanced dataset (100 images per species)
    print("\nCreating balanced dataset with 100 images per species...")
    create_balanced_dataset(source_dataset_path, balanced_dataset_path, images_per_class=100)
    