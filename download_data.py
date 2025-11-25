"""
Download CHAOS dataset from Kaggle using kagglehub.
"""
import kagglehub
import os
import shutil

def download_chaos_dataset(output_dir="data/raw"):
    """
    Download the CHAOS Combined CT-MR dataset from Kaggle.
    
    Args:
        output_dir: Directory to store the downloaded dataset
    """
    print("=" * 60)
    print("Downloading CHAOS Dataset from Kaggle...")
    print("=" * 60)
    
    # Download latest version
    path = kagglehub.dataset_download("omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ")
    
    print(f"\n✓ Dataset downloaded to: {path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy to our project structure
    print(f"\nCopying dataset to {output_dir}...")
    
    # Note: kagglehub downloads to cache, we'll create a symlink or copy
    if os.path.isdir(path):
        # Create symlink for efficiency (avoid duplicating large files)
        symlink_path = os.path.join(output_dir, "chaos_dataset")
        if os.path.exists(symlink_path):
            print(f"Symlink already exists at {symlink_path}")
        else:
            os.symlink(path, symlink_path)
            print(f"✓ Created symlink: {symlink_path} -> {path}")
    
    print("\n" + "=" * 60)
    print("Dataset Download Complete!")
    print("=" * 60)
    print(f"\nDataset location: {output_dir}/chaos_dataset")
    print("\nNext steps:")
    print("1. Explore the dataset structure")
    print("2. Implement preprocessing pipeline")
    print("3. Start training!")
    
    return path

if __name__ == "__main__":
    download_chaos_dataset()
