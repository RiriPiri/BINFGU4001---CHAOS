#!/usr/bin/env python3
"""
Preprocess CHAOS dataset: registration, normalization, and patch extraction.

This script processes all paired CT-MRI volumes and saves patches to disk.
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.dicom_loader import DICOMVolumeLoader, get_available_patients
from data.preprocessing import CTMRIPreprocessor, extract_patches_2d




def is_test_patient(patient_id: str, data_root: str = "data/raw/chaos_dataset") -> bool:
    """
    Determine if a patient ID belongs to the test set.
    
    Args:
        patient_id: Patient ID
        data_root: Root directory of CHAOS dataset
    
    Returns:
        True if patient is from test set, False if from train set
    """
    data_root = Path(data_root)
    test_ct_dir = data_root / "CHAOS_Test_Sets" / "Test_Sets" / "CT" / patient_id
    return test_ct_dir.exists()


def find_paired_patients(
    data_root: str = "data/raw/chaos_dataset",
    include_test_set: bool = False
) -> list:
    """
    Find patients that have both CT and MRI data.
    
    Args:
        data_root: Root directory of CHAOS dataset
        include_test_set: If True, include patients from test set
        
    Returns:
        List of patient IDs with both modalities
    """
    patients = get_available_patients(data_root, include_test_set=include_test_set)
    paired = patients['paired']
    
    print(f"Found {len(paired)} paired patients: {paired}")
    if include_test_set:
        train_patients = [p for p in paired if not is_test_patient(p, data_root)]
        test_patients = [p for p in paired if is_test_patient(p, data_root)]
        print(f"  Train set: {len(train_patients)} patients {train_patients}")
        print(f"  Test set: {len(test_patients)} patients {test_patients}")
    return paired


def preprocess_patient(
    patient_id: str,
    loader: DICOMVolumeLoader,
    preprocessor: CTMRIPreprocessor,
    patch_size: int = 128,
    stride: int = 64,
    use_deformable: bool = False,
    data_root: str = "data/raw/chaos_dataset"
) -> dict:
    """
    Preprocess a single patient's CT-MRI pair.
    
    Args:
        patient_id: Patient ID
        loader: DICOM loader instance
        preprocessor: Preprocessor instance
        patch_size: Patch size for extraction
        stride: Stride for sliding window
        use_deformable: Use deformable registration
        data_root: Dataset root directory
        
    Returns:
        Dictionary with processed data
    """
    # Determine if patient is from test set
    from_test_set = is_test_patient(patient_id, data_root)
    set_name = "Test" if from_test_set else "Train"
    
    print(f"\n{'='*60}")
    print(f"Processing Patient {patient_id} ({set_name} Set)")
    print(f"{'='*60}")
    
    # Load volumes
    print("Loading volumes...")
    ct_volume, ct_metadata = loader.load_ct_volume(
        patient_id, data_root, from_test_set=from_test_set
    )
    mri_volume, mri_metadata = loader.load_mri_volume(
        patient_id, data_root=data_root, from_test_set=from_test_set
    )
    
    print(f"CT shape: {ct_volume.shape}, spacing: {ct_metadata['pixel_spacing']}, "
          f"slice thickness: {ct_metadata['slice_thickness']} mm")
    print(f"MRI shape: {mri_volume.shape}, spacing: {mri_metadata['pixel_spacing']}, "
          f"slice thickness: {mri_metadata['slice_thickness']} mm")
    
    # Normalize
    print("\nNormalizing intensities...")
    ct_normalized = preprocessor.normalize_ct(ct_volume)
    mri_normalized = preprocessor.normalize_mri(mri_volume)
    
    # Register CT to MRI
    print("\nRegistering CT to MRI...")
    ct_spacing = tuple(ct_metadata['pixel_spacing']) + (ct_metadata['slice_thickness'],)
    mri_spacing = tuple(mri_metadata['pixel_spacing']) + (mri_metadata['slice_thickness'],)
    
    try:
        ct_registered, transform = preprocessor.register_ct_to_mri(
            ct_normalized,
            mri_normalized,
            ct_spacing=ct_spacing,
            mri_spacing=mri_spacing,
            use_deformable=use_deformable
        )
    except Exception as e:
        print(f"Registration failed for patient {patient_id}: {e}")
        print("Skipping this patient...")
        return None
    
    # Extract patches
    print("\nExtracting patches...")
    ct_patches, ct_coords = extract_patches_2d(
        ct_registered,
        patch_size=patch_size,
        stride=stride,
        min_foreground_ratio=0.1
    )
    
    mri_patches, mri_coords = extract_patches_2d(
        mri_normalized,
        patch_size=patch_size,
        stride=stride,
        min_foreground_ratio=0.1
    )
    
    # Ensure same number of patches (they should be from same spatial grid)
    min_patches = min(len(ct_patches), len(mri_patches))
    ct_patches = ct_patches[:min_patches]
    mri_patches = mri_patches[:min_patches]
    ct_coords = ct_coords[:min_patches]
    mri_coords = mri_coords[:min_patches]
    
    print(f"Extracted {len(ct_patches)} paired patches")
    print(f"Patch shape: {ct_patches[0].shape}")
    print(f"CT value range: [{ct_patches.min():.3f}, {ct_patches.max():.3f}]")
    print(f"MRI value range: [{mri_patches.min():.3f}, {mri_patches.max():.3f}]")
    
    return {
        'patient_id': patient_id,
        'ct_patches': ct_patches,
        'mri_patches': mri_patches,
        'ct_coords': ct_coords,
        'mri_coords': mri_coords,
        'ct_volume_shape': ct_registered.shape,
        'mri_volume_shape': mri_normalized.shape,
        'metadata': {
            'ct': ct_metadata,
            'mri': mri_metadata
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess CHAOS dataset")
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/raw/chaos_dataset',
        help='Root directory of CHAOS dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed patches'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=128,
        help='Patch size (default: 128)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=64,
        help='Stride for patch extraction (default: 64)'
    )
    parser.add_argument(
        '--deformable',
        action='store_true',
        help='Use deformable registration (slower but more accurate)'
    )
    parser.add_argument(
        '--include_test_set',
        action='store_true',
        help='Include test set patients (expands from 8 to 16 patients)'
    )
    parser.add_argument(
        '--max_patients',
        type=int,
        default=None,
        help='Maximum number of patients to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loader and preprocessor
    loader = DICOMVolumeLoader(verbose=False)
    preprocessor = CTMRIPreprocessor(verbose=True)
    
    # Find paired patients
    paired_patients = find_paired_patients(args.data_root, include_test_set=args.include_test_set)
    
    if args.max_patients:
        paired_patients = paired_patients[:args.max_patients]
        print(f"Processing only first {args.max_patients} patients")
    
    # Process each patient
    processed_data = []
    failed_patients = []
    
    for patient_id in tqdm(paired_patients, desc="Processing patients"):
        result = preprocess_patient(
            patient_id,
            loader,
            preprocessor,
            patch_size=args.patch_size,
            stride=args.stride,
            use_deformable=args.deformable,
            data_root=args.data_root
        )
        
        if result is not None:
            processed_data.append(result)
            
            # Save individual patient data
            patient_file = output_dir / f"patient_{patient_id}_patches.pkl"
            with open(patient_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved to {patient_file}")
        else:
            failed_patients.append(patient_id)
    
    # Save combined dataset info
    dataset_info = {
        'num_patients': len(processed_data),
        'patient_ids': [d['patient_id'] for d in processed_data],
        'total_patches': sum(len(d['ct_patches']) for d in processed_data),
        'patch_size': args.patch_size,
        'stride': args.stride,
        'failed_patients': failed_patients
    }
    
    info_file = output_dir / 'dataset_info.pkl'
    with open(info_file, 'wb') as f:
        pickle.dump(dataset_info, f)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(processed_data)} patients")
    print(f"Failed: {len(failed_patients)} patients {failed_patients}")
    print(f"Total patches: {dataset_info['total_patches']}")
    print(f"Output directory: {output_dir}")
    print(f"\nDataset info saved to: {info_file}")
    
    # Print per-patient statistics
    print(f"\nPer-patient patch counts:")
    for data in processed_data:
        print(f"  Patient {data['patient_id']}: {len(data['ct_patches'])} patches")


if __name__ == "__main__":
    main()
