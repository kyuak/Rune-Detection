import os
import random
import shutil
from pathlib import Path

def create_combined_dataset(
    dataset1_path="/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_official_v1.0",
    dataset2_path="/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_blender_v0.3",
    output_path="/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_combine_v1.0",
    dataset1_ratio=1.0,  # Take 100% of dataset1
    dataset2_ratio=0.5,  # Take 50% of dataset2 (or specify exact number)
    random_seed=42       # For reproducible results
):
    """
    Create a combined dataset from two source datasets.
    
    Args:
        dataset1_path: Path to first dataset (will take ratio1 of these)
        dataset2_path: Path to second dataset (will take ratio2 of these)
        output_path: Where to save the combined dataset
        dataset1_ratio: Ratio (0-1) or exact number of samples to take from dataset1
        dataset2_ratio: Ratio (0-1) or exact number of samples to take from dataset2
        random_seed: Seed for reproducible random sampling
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Convert to Path objects
    dataset1 = Path(dataset1_path)
    dataset2 = Path(dataset2_path)
    combined_dataset = Path(output_path)
    
    # Create output directories
    (combined_dataset / "images").mkdir(parents=True, exist_ok=True)
    (combined_dataset / "labels").mkdir(parents=True, exist_ok=True)
    
    # Counter for sequential numbering
    counter = 0
    
    def process_dataset(dataset_path, ratio):
        nonlocal counter
        images = sorted(list((dataset_path / "images").glob("*")))
        labels = sorted(list((dataset_path / "labels").glob("*")))
        
        # Determine number of samples to take
        if ratio > 1:  # Exact number specified
            n_samples = min(int(ratio), len(images))
        else:  # Ratio specified
            n_samples = int(len(images) * ratio)
        
        # Select samples
        selected_indices = random.sample(range(len(images)), n_samples) if n_samples < len(images) else range(len(images))
        
        for idx in selected_indices:
            img_path = images[idx]
            label_path = labels[idx]
            
            # Verify the label file exists
            if not label_path.exists():
                print(f"Warning: Missing label file {label_path}")
                continue
                
            # Get file extensions
            img_ext = img_path.suffix
            label_ext = label_path.suffix
            
            # Create new filename with 8-digit zero-padded number
            new_filename = f"{counter:08d}"
            
            # Copy files
            shutil.copy(img_path, combined_dataset / "images" / f"{new_filename}{img_ext}")
            shutil.copy(label_path, combined_dataset / "labels" / f"{new_filename}{label_ext}")
            
            counter += 1
        
        return len(selected_indices)
    
    print("Processing datasets...")
    ds1_count = process_dataset(dataset1, dataset1_ratio)
    ds2_count = process_dataset(dataset2, dataset2_ratio)
    
    print("\nCombined dataset created successfully!")
    print(f"Output path: {combined_dataset}")
    print(f"Total samples: {counter}")
    print(f"  - From dataset1 ({dataset1.name}): {ds1_count}")
    print(f"  - From dataset2 ({dataset2.name}): {ds2_count}")

if __name__ == "__main__":
    # 1. Original configuration (all from official, 2000 from blender)
    create_combined_dataset(
        dataset1_ratio=1.0,       # Take all from dataset1
        dataset2_ratio=2000,      # Take exactly 2000 from dataset2
        random_seed=42
    )