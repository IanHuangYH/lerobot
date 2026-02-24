"""
RND Trainer for uncertainty quantification.

Handles training of RND models on chunked token embeddings with:
- Memory-efficient loading via ChunkedRNDDataset
- Early stopping with validation
- TensorBoard logging
- Full checkpoint metadata
"""

import os
import copy
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from uncertainty_quantification.rnd_models.rnd_models import RND_OE
from uncertainty_quantification.dataset.chunk_loader import ChunkedRNDDataset


class RNDTrainer:
    """
    Trainer for RND models with memory-efficient chunked data loading.
    
    Features:
    - On-demand chunk loading (avoids loading full dataset into RAM)
    - Early stopping with validation split
    - TensorBoard logging for train/val losses
    - Cosine learning rate scheduling
    - Full checkpoint saving with metadata
    
    Usage:
        trainer = RNDTrainer(
            dataset_dir="rnd_dataset/spatial",
            camera="agentview",
            output_dir="rnd_save_models/spatial_agentview",
            device="cuda"
        )
        trainer.train(epochs=100, batch_size=256, lr=1e-4)
    """
    
    def __init__(
        self,
        dataset_dir: Path,
        camera: str,
        output_dir: Path,
        device: str = "cuda",
        seed: Optional[int] = None,
        max_cached_chunks: int = 4,
    ):
        """
        Initialize RND trainer.
        
        Args:
            dataset_dir: Path to dataset directory (e.g., "rnd_dataset/spatial")
            camera: Camera name ("agentview" or "wrist")
            output_dir: Directory to save checkpoints and logs
            device: Device for training ("cuda" or "cpu")
            seed: Random seed for reproducibility
            max_cached_chunks: Max chunks to keep in memory (4=~8GB, 2=~4GB)
        """
        self.dataset_dir = Path(dataset_dir)
        self.camera = camera
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.max_cached_chunks = max_cached_chunks
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        
        print(f"RNDTrainer initialized:")
        print(f"  Dataset: {self.dataset_dir}")
        print(f"  Camera: {self.camera}")
        print(f"  Output: {self.output_dir}")
        print(f"  Device: {self.device}")
        print(f"  Seed: {self.seed}")
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-4,
        lr_min: float = 1e-6,
        patience: int = 10,
        train_val_split: float = 0.9,
        obs_embedding_dim: int = 2048,
        output_size: int = 512,
        rnd_loss: str = "mse",
    ) -> Dict[str, Any]:
        """
        Train RND model with early stopping.
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            lr: Initial learning rate
            lr_min: Minimum learning rate for cosine scheduler
            patience: Early stopping patience (epochs without improvement)
            train_val_split: Fraction of data for training (rest for validation)
            obs_embedding_dim: Dimension of input embeddings
            output_size: Dimension of RND feature space
            rnd_loss: Loss function type ("mse" or "l2")
        
        Returns:
            training_info: Dict with final statistics and paths
        """
        # Store hyperparameters
        hyperparameters = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lr_min": lr_min,
            "patience": patience,
            "train_val_split": train_val_split,
            "obs_embedding_dim": obs_embedding_dim,
            "output_size": output_size,
            "rnd_loss": rnd_loss,
            "seed": self.seed,
            "device": self.device,
            "dataset_dir": str(self.dataset_dir),
            "camera": self.camera,
        }
        
        # Log hyperparameters to TensorBoard
        self.writer.add_text("Hyperparameters", self._dict_to_markdown(hyperparameters))
        
        print("\n" + "=" * 60)
        print("Training Configuration:")
        print("=" * 60)
        for key, value in hyperparameters.items():
            print(f"  {key}: {value}")
        print("=" * 60 + "\n")
        
        # Load dataset (memory-efficient chunked loading)
        print("Loading dataset...")
        dataset = ChunkedRNDDataset(
            self.dataset_dir, 
            self.camera,
            max_cached_chunks=self.max_cached_chunks
        )
        print(f"Dataset loaded: {len(dataset):,} tokens total")
        print(f"Cache config: {dataset.max_cached_chunks} chunks (~{dataset.max_cached_chunks * 2:.0f} GB)")

        
        # Split into train and validation
        train_size = int(train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed) if self.seed else None
        )
        
        print(f"Train set: {len(train_dataset):,} tokens ({train_val_split*100:.1f}%)")
        print(f"Val set:   {len(val_dataset):,} tokens ({(1-train_val_split)*100:.1f}%)")
        
        # Create data loaders (num_workers=0 for chunk caching to work)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Important: Keep 0 for chunk caching
            pin_memory=True if self.device == "cuda" else False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Important: Keep 0 for chunk caching
            pin_memory=True if self.device == "cuda" else False,
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}\n")
        
        # Initialize model
        print("Initializing RND model...")
        model = RND_OE(
            obs_embedding_dim=obs_embedding_dim,
            output_size=output_size,
            rnd_loss=rnd_loss,
            seed=self.seed,
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable\n")
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr_min
        )
        
        # Training state
        best_val_loss = float('inf')
        best_state_dict = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print("Starting training...\n")
        progress_bar = tqdm(range(epochs), desc="Training Progress", position=0)
        
        for epoch in progress_bar:
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            
            # Add batch-level progress bar
            batch_progress = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                leave=False,
                position=1
            )
            
            for batch in batch_progress:
                batch = batch.to(self.device).float()  # (batch_size, 2048) - convert to float32
                
                # Forward pass
                loss = model(batch).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
                # Update batch progress bar with current loss
                batch_progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            batch_progress.close()
            
            # Compute average training loss
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            
            # Add validation progress bar
            val_progress = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val]  ",
                leave=False,
                position=1
            )
            
            with torch.no_grad():
                for batch in val_progress:
                    batch = batch.to(self.device).float()  # Convert to float32
                    loss = model(batch).mean()
                    epoch_val_loss += loss.item()
                    val_progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            val_progress.close()
            
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", epoch_train_loss, epoch)
            self.writer.add_scalar("Loss/val", epoch_val_loss, epoch)
            self.writer.add_scalar("Learning_rate", current_lr, epoch)
            
            # Update progress bar
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {epoch_train_loss:.6f} | "
                f"Val: {epoch_val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Check for improvement
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
                # Save best checkpoint
                model.save_checkpoint(
                    save_path=self.output_dir / "best_model.ckpt",
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    optimizer_state=optimizer.state_dict(),
                    train_losses=train_losses,
                    val_losses=val_losses,
                    hyperparameters=hyperparameters,
                )
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
            
            # Check for NaN
            if torch.isnan(torch.tensor(epoch_train_loss)):
                print(f"\nNaN loss detected at epoch {epoch+1}. Stopping training.")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Total epochs: {epoch+1}")
        
        # Print cache statistics
        cache_stats = dataset.get_cache_stats()
        print(f"\nDataset cache statistics:")
        print(f"  Cache hits: {cache_stats['cache_hits']:,}")
        print(f"  Cache misses: {cache_stats['cache_misses']:,}")
        print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
        print(f"  Final cached chunks: {cache_stats['cached_chunks']}/{cache_stats['max_cached_chunks']}")
        print(f"  Estimated memory: {cache_stats['estimated_memory_gb']:.1f} GB")

        
        # Load best model state
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        
        # Save final model with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        final_model_name = f"model_epoch_{epoch+1}_loss_{best_val_loss:.4f}_seed_{self.seed}_{timestamp}.ckpt"
        
        model.save_checkpoint(
            save_path=self.output_dir / final_model_name,
            epoch=epoch,
            best_val_loss=best_val_loss,
            optimizer_state=optimizer.state_dict(),
            train_losses=train_losses,
            val_losses=val_losses,
            hyperparameters=hyperparameters,
        )
        
        # Also save as "model.ckpt" for easy loading
        model.save_checkpoint(
            save_path=self.output_dir / "model.ckpt",
            epoch=epoch,
            best_val_loss=best_val_loss,
            optimizer_state=optimizer.state_dict(),
            train_losses=train_losses,
            val_losses=val_losses,
            hyperparameters=hyperparameters,
        )
        
        # Plot training progress
        self._plot_training_progress(train_losses, val_losses)
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\nCheckpoints saved to: {self.output_dir}")
        print(f"  - best_model.ckpt (best validation loss)")
        print(f"  - model.ckpt (final model)")
        print(f"  - {final_model_name} (timestamped)")
        print(f"\nTensorBoard logs: {self.output_dir / 'tensorboard'}")
        
        return {
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "checkpoint_path": str(self.output_dir / "model.ckpt"),
        }
    
    def _plot_training_progress(self, train_losses: list, val_losses: list):
        """Plot and save training/validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", linewidth=2)
        plt.plot(val_losses, label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MSE)", fontsize=12)
        plt.title("RND Training Progress", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "training_progress.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Training plot saved: {save_path}")
    
    def _dict_to_markdown(self, d: dict) -> str:
        """Convert dictionary to markdown table for TensorBoard."""
        lines = ["| Parameter | Value |", "|-----------|-------|"]
        for key, value in d.items():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines)
