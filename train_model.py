#!/usr/bin/env python3
"""
Main script to train CT-to-MRI synthesis model.

Usage:
    python train_model.py --config configs/train_config.yaml
    python train_model.py --resume checkpoints/latest.pth
"""

import torch
import yaml
import argparse
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import UNetGenerator, PatchEncoder
from losses import CombinedLoss
from data import PairedPatchDataset, create_dataloaders
from training.train import Trainer, get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_preprocessed_data(data_dir: str):
    """
    Load preprocessed patches from disk.
    
    Args:
        data_dir: Directory containing preprocessed .pkl files
        
    Returns:
        Lists of patches, coordinates, and patient IDs
    """
    data_dir = Path(data_dir)
    
    # Load dataset info
    with open(data_dir / 'dataset_info.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    
    patient_ids = dataset_info['patient_ids']
    
    # Load each patient's data
    ct_patches_list = []
    mri_patches_list = []
    ct_coords_list = []
    mri_coords_list = []
    
    for patient_id in patient_ids:
        patient_file = data_dir / f'patient_{patient_id}_patches.pkl'
        
        with open(patient_file, 'rb') as f:
            data = pickle.load(f)
        
        ct_patches_list.append(data['ct_patches'])
        mri_patches_list.append(data['mri_patches'])
        ct_coords_list.append(data['ct_coords'])
        mri_coords_list.append(data['mri_coords'])
    
    return ct_patches_list, mri_patches_list, ct_coords_list, mri_coords_list, patient_ids


def main(args):
    """Main training function."""
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'data_dir': 'data/processed',
            'batch_size': 8,
            'num_workers': 4,
            'num_negatives': 8,
            'learning_rate': 1e-4,
            'epochs': 100,
            'log_interval': 10,
            'save_interval': 10,
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'train_ratio': 0.75,
            # Loss weights
            'lambda_pixel': 1.0,
            'lambda_contrastive': 0.1,
            'lambda_radiomics': 0.01,
            # Model config
            'base_channels': 64,
            'embedding_dim': 128
        }
    
    print("="*60)
    print("CT-to-MRI Synthesis Training")
    print("="*60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    ct_patches_list, mri_patches_list, ct_coords_list, mri_coords_list, patient_ids = \
        load_preprocessed_data(config['data_dir'])
    
    print(f"Loaded {len(patient_ids)} patients: {patient_ids}")
    total_patches = sum(len(patches) for patches in ct_patches_list)
    print(f"Total patches: {total_patches}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        ct_patches_list,
        mri_patches_list,
        ct_coords_list,
        mri_coords_list,
        patient_ids,
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size'],
        num_negatives=config['num_negatives'],
        num_workers=config['num_workers']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create models
    print("\nInitializing models...")
    generator = UNetGenerator(
        in_channels=1,
        out_channels=1,
        base_channels=config['base_channels'],
        bilinear=True
    )
    
    encoder = PatchEncoder(
        in_channels=1,
        base_channels=config['base_channels'],
        embedding_dim=config['embedding_dim'],
        use_projection_head=True
    )
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"Generator parameters: {gen_params:,}")
    print(f"Encoder parameters: {enc_params:,}")
    print(f"Total parameters: {gen_params + enc_params:,}")
    
    # Create loss function
    loss_fn = CombinedLoss(
        pixel_loss_type='l1',
        lambda_pixel=config['lambda_pixel'],
        lambda_contrastive=config['lambda_contrastive'],
        lambda_radiomics=config['lambda_radiomics'],
        use_radiomics=True
    )
    
    # Create trainer
    trainer = Trainer(
        generator=generator,
        encoder=encoder,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n" + "="*60)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CT-to-MRI synthesis model")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    main(args)
