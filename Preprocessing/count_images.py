import os
import matplotlib.pyplot as plt

def count_images_per_class(dataset_path):
    """
    Count the number of images in each class subfolder
    
    Args:
        dataset_path (str): Path to the main dataset folder
        
    Returns:
        dict: Dictionary with class names as keys and image counts as values
    """
    class_counts = {}
    
    # List all subdirectories in the dataset folder
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    print(f"Found {len(class_folders)} classes in the dataset.")
    
    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        # Count only image files
        image_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        class_counts[class_name] = len(image_files)
        
    return class_counts

def plot_class_distribution(class_counts):
    """
    Plot the distribution of images across classes
    
    Args:
        class_counts (dict): Dictionary with class names and their image counts
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts)
    plt.xlabel('Snake Species')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images Across Snake Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution_balanced.png')
    plt.close()
    
    print("Class distribution plot saved as 'class_distribution.png'")

if __name__ == "__main__":
    # Define paths
    dataset_path = "SnakeDataset"
    balanced_output_path = "BalancedSnakeDataset"
    split_output_path = "SplitSnakeDataset"
    preprocessed_output_path = "PreprocessedSnakeDataset"
    
    # Count images per class
    print("Counting images per class...")
    class_counts = count_images_per_class(balanced_output_path)
    
    # Display results
    print("\nImage count per class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    
    # Plot class distribution
    plot_class_distribution(class_counts)