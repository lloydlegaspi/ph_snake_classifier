import os
import numpy as np
import cv2
from tqdm import tqdm
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, VerticalFlip, 
    RandomBrightnessContrast, RandomCrop, RandomScale
)
from utilities import count_images_in_directory

def create_augmentation_pipeline():
    """
    Create an augmentation pipeline for training images based on methodology
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    return Compose([
        # Random rotation
        RandomRotate90(p=0.5),
        
        # Horizontal and vertical flipping
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        
        # Brightness and contrast adjustments
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Random cropping (95% of original area)
        RandomCrop(height=int(224 * 0.95), width=int(224 * 0.95), p=0.5),
        
        # Random scaling (zoom in/out)
        RandomScale(scale_limit=0.2, p=0.5)
    ])

def apply_augmentation(image, augmentation_pipeline):
    """
    Apply augmentation to an image
    
    Args:
        image (numpy.ndarray): Input image
        augmentation_pipeline: Augmentation pipeline created with albumentations
        
    Returns:
        numpy.ndarray: Augmented image
    """
    augmented = augmentation_pipeline(image=image)
    return augmented['image']

def preprocess_image(image_path, output_size=(224, 224)):
    """
    Preprocess an image for the deep learning model
    
    Args:
        image_path (str): Path to the image file
        output_size (tuple): Target size for the image (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image = cv2.resize(image, output_size)
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def create_output_structure(input_path, output_path, split='train'):
    """
    Create the output directory structure for augmentation
    
    Args:
        input_path (str): Path to the input dataset
        output_path (str): Path where the output structure will be created
        split (str): Dataset split to process (default: 'train')
    """
    # Create output directory for the split
    os.makedirs(os.path.join(output_path, split), exist_ok=True)
    
    # Get class folders
    split_input_path = os.path.join(input_path, split)
    if not os.path.exists(split_input_path):
        print(f"Warning: Split path {split_input_path} does not exist.")
        return False
        
    class_folders = [f for f in os.listdir(split_input_path) 
                    if os.path.isdir(os.path.join(split_input_path, f))]
    
    # Create class subfolders
    for class_name in class_folders:
        os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)
    
    return True

def augment_dataset(input_path, output_path, num_augmentations=3):
    """
    Apply augmentation to training images and save them in the output path
    
    Args:
        input_path (str): Path to the input dataset (already split)
        output_path (str): Path where the augmented dataset will be saved
        num_augmentations (int): Number of augmented versions to create per training image
        
    Returns:
        tuple: Count of original and augmented images
    """
    # Create augmentation pipeline
    augmentation_pipeline = create_augmentation_pipeline()
    
    # We only augment the training set
    split = 'train'
    
    # Create output structure
    if not create_output_structure(input_path, output_path, split):
        print("Failed to create output structure. Aborting augmentation.")
        return 0, 0
    
    # Get class folders
    class_folders = [f for f in os.listdir(os.path.join(input_path, split)) 
                     if os.path.isdir(os.path.join(input_path, split, f))]
    
    # Count original images
    original_count = count_images_in_directory(os.path.join(input_path, split))
    
    # Track augmented images count
    augmented_count = 0
    
    # Process each class in training set
    for class_name in tqdm(class_folders, desc="Augmenting training images"):
        class_path = os.path.join(input_path, split, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        # Process each image
        for file_name in image_files:
            input_file_path = os.path.join(class_path, file_name)
            
            # Load and preprocess the image
            preprocessed_image = preprocess_image(input_file_path)
            if preprocessed_image is None:
                continue
                
            filename_without_ext = os.path.splitext(file_name)[0]
            
            # Create multiple augmented versions
            for i in range(num_augmentations):
                # Apply augmentation
                augmented_image = apply_augmentation(preprocessed_image, augmentation_pipeline)
                
                # Save augmented image
                augmented_file_path = os.path.join(
                    output_path, split, class_name, 
                    f"{filename_without_ext}_aug_{i+1}.png"
                )
                
                # Skip if already exists
                if os.path.exists(augmented_file_path):
                    augmented_count += 1
                    continue
                
                try:
                    cv2.imwrite(augmented_file_path, 
                              cv2.cvtColor(
                                  (augmented_image * 255).astype(np.uint8), 
                                  cv2.COLOR_RGB2BGR)
                             )
                    augmented_count += 1
                except Exception as e:
                    print(f"Error saving augmented image {augmented_file_path}: {e}")
    
    return original_count, augmented_count

def summarize_augmentation_results(input_path, output_path, original_count, augmented_count):
    """
    Summarize the results of data augmentation
    
    Args:
        input_path (str): Path to the input dataset
        output_path (str): Path to the output dataset with augmentations
        original_count (int): Number of original images
        augmented_count (int): Number of augmented images created
    """
    print("\n=== Data Augmentation Results ===")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Original training images: {original_count}")
    print(f"Augmented images created: {augmented_count}")
    print(f"Total training images after augmentation: {original_count + augmented_count}")
    print(f"Augmentation factor: {(original_count + augmented_count) / original_count:.2f}x")
    print("================================")

if __name__ == "__main__":
    # Define paths - use preprocessed data as input
    preprocessed_path = "PreprocessedSnakeDataset"
    augmented_output_path = "AugmentedSnakeDataset"
    
    # Apply augmentation to training images
    print("\nApplying augmentation to training images...")
    
    # Track counts for results summary
    original_count, augmented_count = augment_dataset(
        preprocessed_path, 
        augmented_output_path, 
        num_augmentations=3
    )
    
    # Show results
    summarize_augmentation_results(
        preprocessed_path,
        augmented_output_path,
        original_count,
        augmented_count
    )
    
    print("\nData augmentation completed successfully!")