import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from utilities import count_images_in_directory, standardize_image_filenames
from split_dataset import verify_splits
from data_augmentation import preprocess_image

def create_output_structure(input_path, output_path):
    """
    Create the output directory structure based on input dataset
    
    Args:
        input_path (str): Path to the input dataset
        output_path (str): Path where the output structure will be created
    """
    for split in ['train', 'val', 'test']:
        split_input_path = os.path.join(input_path, split)
        if not os.path.exists(split_input_path):
            print(f"Warning: Split path {split_input_path} does not exist. Skipping.")
            continue
            
        # Create split directory
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
        
        # Get class folders
        class_folders = [f for f in os.listdir(split_input_path) 
                        if os.path.isdir(os.path.join(split_input_path, f))]
        
        # Create class subfolders in each split
        for class_name in class_folders:
            os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)

def preprocess_dataset(input_path, output_path):
    """
    Preprocess all images in the dataset and save them in a new location
    
    Args:
        input_path (str): Path to the input dataset (already split)
        output_path (str): Path where the preprocessed dataset will be saved
    
    Returns:
        tuple: Dictionaries with counts of processed and error images per split
    """
    processed_counts = {'train': 0, 'val': 0, 'test': 0}
    error_counts = {'train': 0, 'val': 0, 'test': 0}
    
    # First, standardize filenames in the input directory
    print("Standardizing filenames...")
    standardize_image_filenames(input_path)
    
    # Create output directory structure
    create_output_structure(input_path, output_path)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_input_path = os.path.join(input_path, split)
        if not os.path.exists(split_input_path):
            continue
            
        class_folders = [f for f in os.listdir(split_input_path) 
                        if os.path.isdir(os.path.join(split_input_path, f))]
        
        # Process each class
        for class_name in tqdm(class_folders, desc=f"Preprocessing {split} set"):
            class_path = os.path.join(input_path, split, class_name)
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            
            # Process each image
            for file_name in image_files:
                input_file_path = os.path.join(class_path, file_name)
                output_file_path = os.path.join(output_path, split, class_name, file_name)
                
                # Skip if already processed
                if os.path.exists(output_file_path):
                    processed_counts[split] += 1
                    continue
                
                # Preprocess and save
                try:
                    preprocessed_image = preprocess_image(input_file_path)
                    if preprocessed_image is not None:
                        # Save as PNG to avoid further compression artifacts
                        cv2.imwrite(output_file_path, 
                                  cv2.cvtColor(
                                      (preprocessed_image * 255).astype(np.uint8), 
                                      cv2.COLOR_RGB2BGR)
                                 )
                        processed_counts[split] += 1
                    else:
                        error_counts[split] += 1
                        print(f"Could not preprocess: {input_file_path}")
                except Exception as e:
                    error_counts[split] += 1
                    print(f"Error processing {input_file_path}: {e}")
    
    return processed_counts, error_counts

def verify_and_balance_splits(output_path, expected_counts=None):
    """
    Verify splits and optionally balance them to match expected counts
    
    Args:
        output_path (str): Path to the split dataset
        expected_counts (dict, optional): Expected number of images per class per split
    """
    splits = ['train', 'val', 'test']
    all_counts = {}
    imbalanced_classes = []
    
    # Get actual counts
    for split in splits:
        split_path = os.path.join(output_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: Split path {split_path} does not exist. Skipping.")
            continue
            
        class_folders = [f for f in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, f))]
        
        split_counts = {}
        for class_name in class_folders:
            class_path = os.path.join(split_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            split_counts[class_name] = len(image_files)
            
        all_counts[split] = split_counts
    
    # Print initial summary
    print("\nCurrent Dataset Split Summary:")
    print("-" * 60)
    print(f"{'Class':<40} {'Train':<10} {'Validation':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    # Default expected counts if not provided
    if not expected_counts:
        expected_counts = {
            'train': 70,
            'val': 15,
            'test': 15
        }
    
    # Check for imbalances
    for class_name in all_counts['train'].keys():
        train_count = all_counts['train'].get(class_name, 0)
        val_count = all_counts['val'].get(class_name, 0)
        test_count = all_counts['test'].get(class_name, 0)
        total = train_count + val_count + test_count
        
        print(f"{class_name:<40} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
        
        # Check if any of the splits doesn't match expected counts
        if train_count != expected_counts['train'] or val_count != expected_counts['val'] or test_count != expected_counts['test']:
            imbalanced_classes.append(class_name)
    
    print("-" * 60)
    
    if imbalanced_classes:
        print(f"\nFound {len(imbalanced_classes)} imbalanced classes: {', '.join(imbalanced_classes)}")
        print("Please check these classes and ensure they have consistent counts.")
    else:
        print("All classes have consistent counts.")
    
    return all_counts, imbalanced_classes

def summarize_preprocessing_results(input_path, output_path, processed_counts, error_counts=None):
    """
    Summarize the results of preprocessing
    
    Args:
        input_path (str): Path to the input dataset
        output_path (str): Path to the preprocessed dataset
        processed_counts (dict): Dictionary with counts of preprocessed images per split
        error_counts (dict, optional): Dictionary with counts of errors per split
    """
    print("\n=== Preprocessing Results ===")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Preprocessed train images: {processed_counts['train']}")
    print(f"Preprocessed validation images: {processed_counts['val']}")
    print(f"Preprocessed test images: {processed_counts['test']}")
    print(f"Total preprocessed images: {sum(processed_counts.values())}")
    
    if error_counts:
        print(f"\nErrors in train set: {error_counts['train']}")
        print(f"Errors in validation set: {error_counts['val']}")
        print(f"Errors in test set: {error_counts['test']}")
        print(f"Total errors: {sum(error_counts.values())}")
    
    print("============================")

if __name__ == "__main__":
    # Define paths
    split_dataset_path = "SplitSnakeDataset"
    preprocessed_output_path = "PreprocessedSnakeDataset"
    
    # Count original images
    original_count = count_images_in_directory(split_dataset_path)
    print(f"Original dataset contains {original_count} images")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    processed_counts, error_counts = preprocess_dataset(split_dataset_path, preprocessed_output_path)
    
    # Show preprocessing results
    summarize_preprocessing_results(
        split_dataset_path,
        preprocessed_output_path,
        processed_counts,
        error_counts
    )
    
    # Verify preprocessed dataset
    print("\nVerifying preprocessed dataset structure...")
    verify_splits(preprocessed_output_path)
    
    # Check for imbalances
    print("\nChecking for class imbalances...")
    verify_and_balance_splits(preprocessed_output_path)
    
    print("\nPreprocessing completed successfully!")