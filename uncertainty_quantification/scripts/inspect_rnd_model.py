#!/usr/bin/env python3
"""
Inspect RND model checkpoints and verify they can be loaded.

This utility helps verify trained models by:
1. Loading checkpoint metadata
2. Checking model architecture
3. Verifying weights can be loaded
4. Displaying training statistics

Usage:
    # Inspect single model
    python -m uncertainty_quantification.scripts.inspect_rnd_model \\
        --model-path uncertainty_quantification/rnd_save_models/spatial_agentview/model.ckpt

    # Inspect all models
    python -m uncertainty_quantification.scripts.inspect_rnd_model --all
"""

import argparse
import json
from pathlib import Path

import torch

from uncertainty_quantification.rnd_models import RND_OE


def inspect_checkpoint(checkpoint_path: Path):
    """Inspect a single checkpoint file."""
    print(f"\n{'=' * 70}")
    print(f"Inspecting: {checkpoint_path}")
    print('=' * 70)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False
    
    # Display model configuration
    print("\n📋 Model Configuration:")
    model_config = checkpoint.get('model_config', {})
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Display training statistics
    print("\n📊 Training Statistics:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    if train_losses:
        print(f"  Initial Train Loss: {train_losses[0]:.6f}")
        print(f"  Final Train Loss: {train_losses[-1]:.6f}")
        print(f"  Train Loss Reduction: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")
    
    if val_losses:
        print(f"  Initial Val Loss: {val_losses[0]:.6f}")
        print(f"  Final Val Loss: {val_losses[-1]:.6f}")
        print(f"  Val Loss Reduction: {(1 - val_losses[-1]/val_losses[0])*100:.1f}%")
    
    # Display hyperparameters
    print("\n⚙️ Hyperparameters:")
    hyperparams = checkpoint.get('hyperparameters', {})
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Verify model can be loaded
    print("\n🔍 Verifying Model Loading:")
    try:
        model = RND_OE.load_checkpoint(checkpoint_path, device='cpu')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        
        # Test forward pass
        test_input = torch.randn(10, model_config.get('obs_embedding_dim', 2048))
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  ✓ Forward pass successful")
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Test output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False
    
    # Check for metadata.json
    metadata_path = checkpoint_path.parent / 'metadata.json'
    if metadata_path.exists():
        print(f"\n📄 Metadata file: {metadata_path}")
        print("  ✓ Found")
    else:
        print(f"\n📄 Metadata file: Not found")
    
    # Check for training plot
    plot_path = checkpoint_path.parent / 'training_progress.png'
    if plot_path.exists():
        print(f"📈 Training plot: {plot_path}")
        print("  ✓ Found")
    else:
        print(f"📈 Training plot: Not found")
    
    # Check for TensorBoard logs
    tb_dir = checkpoint_path.parent / 'tensorboard'
    if tb_dir.exists():
        tb_files = list(tb_dir.glob('events.out.tfevents.*'))
        print(f"📊 TensorBoard logs: {tb_dir}")
        print(f"  ✓ Found ({len(tb_files)} event files)")
    else:
        print(f"📊 TensorBoard logs: Not found")
    
    print(f"\n{'=' * 70}")
    print("✅ Checkpoint is valid and loadable")
    print('=' * 70)
    
    return True


def inspect_all_models(models_dir: Path):
    """Inspect all models in the directory."""
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("\nPlease train models first using:")
        print("  ./uncertainty_quantification/scripts/train_all_rnd.sh")
        return
    
    # Find all model.ckpt files
    checkpoints = list(models_dir.glob("*/model.ckpt"))
    
    if not checkpoints:
        print(f"❌ No model checkpoints found in: {models_dir}")
        return
    
    print(f"\n{'=' * 70}")
    print(f"Found {len(checkpoints)} model checkpoints")
    print('=' * 70)
    
    # Inspect each checkpoint
    valid_count = 0
    for checkpoint_path in sorted(checkpoints):
        success = inspect_checkpoint(checkpoint_path)
        if success:
            valid_count += 1
    
    # Summary
    print(f"\n\n{'=' * 70}")
    print("Summary")
    print('=' * 70)
    print(f"Total models: {len(checkpoints)}")
    print(f"Valid models: {valid_count}")
    print(f"Invalid models: {len(checkpoints) - valid_count}")
    
    if valid_count == len(checkpoints):
        print("\n✅ All models are valid and loadable!")
    else:
        print(f"\n⚠️ {len(checkpoints) - valid_count} model(s) failed validation")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect RND model checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to specific model checkpoint to inspect"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all models in the save directory"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="uncertainty_quantification/rnd_save_models",
        help="Directory containing trained models"
    )
    
    args = parser.parse_args()
    
    if args.all:
        inspect_all_models(Path(args.models_dir))
    elif args.model_path:
        inspect_checkpoint(Path(args.model_path))
    else:
        print("Please specify either --model-path or --all")
        print("\nExamples:")
        print("  # Inspect specific model")
        print("  python -m uncertainty_quantification.scripts.inspect_rnd_model \\")
        print("    --model-path uncertainty_quantification/rnd_save_models/spatial_agentview/model.ckpt")
        print("\n  # Inspect all models")
        print("  python -m uncertainty_quantification.scripts.inspect_rnd_model --all")


if __name__ == "__main__":
    main()
