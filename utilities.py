import os
import re
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def count_images_in_directory(directory):
    """
    Count images in a directory and its subdirectories
    
    Args:
        directory (str): Path to the directory to count images in
        
    Returns:
        int: Total number of images found
    """
    total = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                total += 1
    return total

def standardize_image_filenames(dataset_path, preserve_originals=False):
    """
    Rename all image files in the dataset using a standardized naming convention
    
    Args:
        dataset_path (str): Path to the dataset (with train/val/test splits)
        preserve_originals (bool): If True, copy files instead of renaming them
        
    Returns:
        dict: Statistics about renamed files
    """
    stats = {'renamed': 0, 'errors': 0, 'mapping': {}}
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: Split path {split_path} does not exist. Skipping.")
            continue
            
        # Get all class folders
        class_folders = [f for f in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, f))]
        
        # Process each class
        for class_name in tqdm(class_folders, desc=f"Standardizing filenames in {split} set"):
            class_path = os.path.join(split_path, class_name)
            
            # Get simplified class name for filenames (remove special chars)
            simple_class_name = re.sub(r'[^\w]', '_', class_name).lower()
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpeg', '.jpg', '.png', '.gif', '.bmp'))]
            
            # Sort to ensure consistent numbering across runs
            image_files.sort()
            
            # Rename files using standard convention
            for i, filename in enumerate(image_files):
                # Create standardized name: class_split_0001.ext
                orig_path = os.path.join(class_path, filename)
                _, ext = os.path.splitext(filename)
                new_filename = f"{simple_class_name}_{split}_{i+1:04d}{ext.lower()}"
                new_path = os.path.join(class_path, new_filename)
                
                # Skip if the file already has the standardized name
                if filename == new_filename:
                    continue
                    
                # If target already exists with different name, use a different number
                counter = i + 1
                while os.path.exists(new_path) and new_path != orig_path:
                    counter += 1
                    new_filename = f"{simple_class_name}_{split}_{counter:04d}{ext.lower()}"
                    new_path = os.path.join(class_path, new_filename)
                
                try:
                    if preserve_originals:
                        shutil.copy2(orig_path, new_path)
                    else:
                        os.rename(orig_path, new_path)
                    
                    stats['renamed'] += 1
                    stats['mapping'][filename] = new_filename
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")
                    stats['errors'] += 1
    
    print(f"Standardized {stats['renamed']} filenames with {stats['errors']} errors")
    return stats