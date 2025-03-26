import os
import glob

def get_dataset_paths(dataset_path):
    """Generate paths for images and labels folders."""
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')
    return images_path, labels_path

def count_files(folder_path, extensions):
    """Count files with specified extensions in a folder."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    return len(files)

def check_dataset_consistency(dataset_path):
    """Check if the number of images matches the number of labels."""
    images_path, labels_path = get_dataset_paths(dataset_path)
    
    # Count images (.jpg, .png) and labels (.txt)
    image_count = count_files(images_path, ('.jpg', '.jpeg', '.png'))
    label_count = count_files(labels_path, ('.txt',))
    
    # Check consistency
    is_consistent = image_count == label_count
    status = "CONSISTENT" if is_consistent else "INCONSISTENT"
    
    # Print report
    print(f"Dataset Consistency Check:")
    print(f"- Images folder: {images_path}")
    print(f"- Labels folder: {labels_path}")
    print(f"- Number of images: {image_count}")
    print(f"- Number of labels: {label_count}")
    print(f"- Status: {status}")
    
    if not is_consistent:
        print("\nWarning: Mismatch detected! Possible issues:")
        if image_count > label_count:
            print("  - Some images are missing label files.")
        else:
            print("  - Some labels are missing image files.")
    
    return is_consistent

# Example usage
dataset_path = '/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_blender_v0.3'
check_dataset_consistency(dataset_path)