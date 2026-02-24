#!/usr/bin/env python3
"""
Train a single RND model for uncertainty quantification.

This script trains one RND model on token embeddings collected from LIBERO
demonstrations. It uses memory-efficient chunked data loading to handle large
datasets without loading everything into RAM.

Usage:
    # Train RND for spatial task, agentview camera
    python -m uncertainty_quantification.scripts.train_rnd \\
        --task-type spatial \\
        --camera agentview \\
        --epochs 100 \\
        --batch-size 256 \\
        --lr 1e-4

    # Quick test with fewer epochs
    python -m uncertainty_quantification.scripts.train_rnd \\
        --task-type spatial \\
        --camera agentview \\
        --epochs 10 \\
        --batch-size 128

    # Use CPU if no GPU available
    python -m uncertainty_quantification.scripts.train_rnd \\
        --task-type spatial \\
        --camera agentview \\
        --device cpu
"""

import argparse
from pathlib import Path
import torch

from uncertainty_quantification.rnd_models.rnd_trainer import RNDTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RND model for uncertainty quantification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset arguments
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=["spatial", "object", "goal", "long"],
        help="Task type to train on"
    )
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        choices=["agentview", "wrist"],
        help="Camera to train RND for"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="uncertainty_quantification/rnd_dataset",
        help="Root directory containing chunked token datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="uncertainty_quantification/rnd_save_models",
        help="Directory to save trained models"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine scheduler"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (rest for validation)"
    )
    
    # Model architecture
    parser.add_argument(
        "--obs-embedding-dim",
        type=int,
        default=2048,
        help="Dimension of input embeddings (SigLIP token dimension)"
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=512,
        help="Dimension of RND feature space"
    )
    parser.add_argument(
        "--rnd-loss",
        type=str,
        default="mse",
        choices=["mse", "l2"],
        help="RND loss function type"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Construct paths
    dataset_path = Path(args.dataset_dir) / args.task_type
    output_path = Path(args.output_dir) / f"{args.task_type}_{args.camera}"
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print(f"\nPlease run data collection first:")
        print(f"  python -m uncertainty_quantification.scripts.collect_rnd_training_data \\")
        print(f"    --task-type {args.task_type}")
        return
    
    print("\n" + "=" * 70)
    print("RND Model Training")
    print("=" * 70)
    print(f"Task Type: {args.task_type}")
    print(f"Camera: {args.camera}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print("=" * 70 + "\n")
    
    # Initialize trainer
    trainer = RNDTrainer(
        dataset_dir=dataset_path,
        camera=args.camera,
        output_dir=output_path,
        device=args.device,
        seed=args.seed,
    )
    
    # Train model
    training_info = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        patience=args.patience,
        train_val_split=args.train_val_split,
        obs_embedding_dim=args.obs_embedding_dim,
        output_size=args.output_size,
        rnd_loss=args.rnd_loss,
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {training_info['best_val_loss']:.6f}")
    print(f"Total epochs: {training_info['final_epoch']}")
    print(f"Checkpoint: {training_info['checkpoint_path']}")
    print("=" * 70 + "\n")
    
    print("To view training progress in TensorBoard:")
    print(f"  tensorboard --logdir {output_path / 'tensorboard'}")
    print()


if __name__ == "__main__":
    main()
