#!/usr/bin/env python3
"""
Verify that the CT-to-MRI synthesis project environment is correctly set up.
"""

import sys
from pathlib import Path


def check_imports():
    """Check that all required packages can be imported."""
    print("=" * 60)
    print("Checking Python Package Imports...")
    print("=" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'monai': 'MONAI',
        'SimpleITK': 'SimpleITK',
        'nibabel': 'NiBabel',
        'scipy': 'SciPy',
        'skimage': 'scikit-image',
        'matplotlib': 'Matplotlib',
        'tensorboard': 'TensorBoard',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'pydicom': 'pydicom',
        'numpy': 'NumPy'
    }
    
    failed = []
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def check_custom_modules():
    """Check custom project modules."""
    print("\n" + "=" * 60)
    print("Checking Custom Modules...")
    print("=" * 60)
    
    try:
        from src.utils.metrics import compute_psnr, compute_ssim
        print("✓ src.utils.metrics - OK")
    except Exception as e:
        print(f"✗ src.utils.metrics - FAILED: {e}")
        return False
    
    try:
        from src.utils.visualization import visualize_comparison
        print("✓ src.utils.visualization - OK")
    except Exception as e:
        print(f"✗ src.utils.visualization - FAILED: {e}")
        return False
    
    try:
        from src.utils.radiomics import RadiomicsExtractor
        extractor = RadiomicsExtractor()
        print("✓ src.utils.radiomics - OK")
        
        # Test feature extraction
        import numpy as np
        test_img = np.random.rand(128, 128).astype(np.float32)
        features = extractor.extract_features(test_img)
        assert len(features) == 27, f"Expected 27 features, got {len(features)}"
        print(f"  → Extracted {len(features)} radiomics features")
    except Exception as e:
        print(f"✗ src.utils.radiomics - FAILED: {e}")
        return False
    
    print("\n✅ All custom modules working!")
    return True


def check_dataset():
    """Check if CHAOS dataset is downloaded."""
    print("\n" + "=" * 60)
    print("Checking Dataset...")
    print("=" * 60)
    
    dataset_path = Path("data/raw/chaos_dataset")
    
    if not dataset_path.exists():
        print("✗ CHAOS dataset not found at data/raw/chaos_dataset")
        print("  Run: python download_data.py")
        return False
    
    train_path = dataset_path / "CHAOS_Train_Sets" / "Train_Sets"
    
    if not train_path.exists():
        print("✗ CHAOS training data not found")
        return False
    
    ct_path = train_path / "CT"
    mr_path = train_path / "MR"
    
    if ct_path.exists() and mr_path.exists():
        ct_patients = len([p for p in ct_path.iterdir() if p.is_dir()])
        mr_patients = len([p for p in mr_path.iterdir() if p.is_dir()])
        
        print(f"✓ CHAOS dataset found")
        print(f"  → CT patients: {ct_patients}")
        print(f"  → MR patients: {mr_patients}")
        print(f"  → Location: {dataset_path}")
        return True
    else:
        print("✗ CT or MR directories not found")
        return False


def check_project_structure():
    """Check that project directories exist."""
    print("\n" + "=" * 60)
    print("Checking Project Structure...")
    print("=" * 60)
    
    required_dirs = [
        "src/data",
        "src/models",
        "src/losses",
        "src/training",
        "src/inference",
        "src/utils",
        "configs",
        "notebooks",
        "data/raw",
        "data/processed",
        "tests"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ Project structure complete!")
    else:
        print("\n⚠️  Some directories are missing")
    
    return all_exist


def print_versions():
    """Print versions of key packages."""
    print("\n" + "=" * 60)
    print("Package Versions")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch:    {torch.__version__}")
    except:
        pass
    
    try:
        import monai
        print(f"MONAI:      {monai.__version__}")
    except:
        pass
    
    try:
        import SimpleITK as sitk
        print(f"SimpleITK:  {sitk.Version.VersionString()}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"NumPy:      {np.__version__}")
    except:
        pass
    
    print(f"Python:     {sys.version.split()[0]}")


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("CT-to-MRI Synthesis - Environment Verification")
    print("=" * 60)
    
    checks = [
        ("Package Imports", check_imports),
        ("Custom Modules", check_custom_modules),
        ("Dataset", check_dataset),
        ("Project Structure", check_project_structure)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print_versions()
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 Environment setup complete! Ready for Phase 2.")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("⚠️  Some checks failed. Please review above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
