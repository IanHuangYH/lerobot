"""
Example usage of the chunked RND dataset loader.

This demonstrates how to load and iterate through the chunked dataset
for RND model training.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from uncertainty_quantification.dataset import (
    load_rnd_dataset,
    get_chunk_info,
    ChunkedRNDDataset,
)


def example_load_full_dataset():
    """
    Example 1: Load entire dataset into memory (small datasets only).
    
    Warning: Only use this for small datasets (<20GB). For larger datasets,
    use ChunkedRNDDataset instead.
    """
    dataset_dir = Path("uncertainty_quantification/rnd_dataset/spatial")
    
    # Load all agentview tokens at once
    agentview_tokens = load_rnd_dataset(dataset_dir, "agentview")
    print(f"Loaded agentview tokens: {agentview_tokens.shape}")
    # Expected: (N, 2048) where N = total_frames * tokens_per_frame
    
    # Load all wrist tokens
    wrist_tokens = load_rnd_dataset(dataset_dir, "wrist")
    print(f"Loaded wrist tokens: {wrist_tokens.shape}")


def example_get_info():
    """
    Example 2: Get dataset information without loading data.
    """
    dataset_dir = Path("uncertainty_quantification/rnd_dataset/spatial")
    
    num_chunks, max_frames, total_tokens = get_chunk_info(dataset_dir)
    print(f"Dataset info:")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Max frames per chunk: {max_frames}")
    print(f"  Total tokens per camera: {total_tokens:,}")


def example_chunked_dataset():
    """
    Example 3: Use ChunkedRNDDataset for memory-efficient training (recommended).
    
    This is the recommended approach for large datasets as it loads chunks
    on-demand rather than loading everything into memory.
    """
    dataset_dir = Path("uncertainty_quantification/rnd_dataset/spatial")
    
    # Create dataset (loads only metadata)
    agentview_dataset = ChunkedRNDDataset(dataset_dir, "agentview")
    print(f"Dataset length: {len(agentview_dataset):,} tokens")
    
    # Create dataloader with shuffling
    dataloader = DataLoader(
        agentview_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,  # Set to 0 for chunk caching to work
        drop_last=True,
    )
    
    # Iterate through batches for RND training
    for batch_idx, batch in enumerate(dataloader):
        # batch shape: (256, 2048)
        print(f"Batch {batch_idx}: {batch.shape}")
        
        # Train RND model here
        # rnd_loss = train_rnd_step(batch)
        
        # Only show first few batches in example
        if batch_idx >= 2:
            break


def example_both_cameras():
    """
    Example 4: Train with both cameras simultaneously.
    """
    dataset_dir = Path("uncertainty_quantification/rnd_dataset/spatial")
    
    # Create datasets for both cameras
    agentview_dataset = ChunkedRNDDataset(dataset_dir, "agentview")
    wrist_dataset = ChunkedRNDDataset(dataset_dir, "wrist")
    
    # Create dataloaders
    batch_size = 256
    agentview_loader = DataLoader(agentview_dataset, batch_size=batch_size, shuffle=True)
    wrist_loader = DataLoader(wrist_dataset, batch_size=batch_size, shuffle=True)
    
    # Train both RND models together
    for (agentview_batch, wrist_batch) in zip(agentview_loader, wrist_loader):
        print(f"Agentview batch: {agentview_batch.shape}")
        print(f"Wrist batch: {wrist_batch.shape}")
        
        # Train both RND models
        # agentview_loss = train_rnd_agentview(agentview_batch)
        # wrist_loss = train_rnd_wrist(wrist_batch)
        
        break  # Only show first batch in example


if __name__ == "__main__":
    print("=" * 60)
    print("RND Dataset Loading Examples")
    print("=" * 60)
    print()
    
    print("Example 1: Get dataset info")
    print("-" * 60)
    example_get_info()
    print()
    
    print("Example 2: Chunked dataset (recommended)")
    print("-" * 60)
    example_chunked_dataset()
    print()
    
    print("Example 3: Both cameras")
    print("-" * 60)
    example_both_cameras()
    print()
