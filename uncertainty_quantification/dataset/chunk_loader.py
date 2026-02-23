"""
Utilities to load chunked RND training datasets.

The RND dataset is saved as multiple chunk files to avoid memory issues:
- agentview_tokens_00000.pt, agentview_tokens_00001.pt, ...
- wrist_tokens_00000.pt, wrist_tokens_00001.pt, ...
- collection_stats.json (metadata)
"""

import json
from pathlib import Path
from typing import Tuple

import torch


def load_rnd_dataset(
    dataset_dir: Path,
    camera: str,
) -> torch.Tensor:
    """
    Load complete RND training dataset for a camera by concatenating all chunks.
    
    Warning: This loads all data into memory. For large datasets (>50GB),
    consider using ChunkedRNDDataset for on-demand loading instead.
    
    Args:
        dataset_dir: Path to task dataset directory (e.g., rnd_dataset/spatial/)
        camera: Camera name ('agentview' or 'wrist')
        
    Returns:
        Tensor of shape (N, 2048) containing all tokens
    """
    dataset_dir = Path(dataset_dir)
    
    # Load metadata
    stats_file = dataset_dir / "collection_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    num_chunks = stats['num_chunks']
    
    # Load and concatenate all chunks
    chunks = []
    for i in range(num_chunks):
        chunk_file = dataset_dir / f"{camera}_tokens_{i:05d}.pt"
        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing chunk file: {chunk_file}")
        chunks.append(torch.load(chunk_file))
    
    return torch.cat(chunks, dim=0)


def get_chunk_info(dataset_dir: Path) -> Tuple[int, int, int]:
    """
    Get information about the chunked dataset.
    
    Args:
        dataset_dir: Path to task dataset directory
        
    Returns:
        (num_chunks, max_frames_per_chunk, total_tokens)
    """
    dataset_dir = Path(dataset_dir)
    stats_file = dataset_dir / "collection_stats.json"
    
    if not stats_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    return stats['num_chunks'], stats['max_frames_per_chunk'], stats['total_tokens_agentview']


class ChunkedRNDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for RND training with on-demand chunk loading.
    
    Loads chunks only when needed to minimize memory usage. Suitable for
    large datasets that don't fit in RAM.
    
    Example usage:
        dataset = ChunkedRNDDataset("rnd_dataset/spatial", "agentview")
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        for batch in dataloader:
            # batch shape: (256, 2048)
            train_rnd(batch)
    """
    
    def __init__(self, dataset_dir: Path, camera: str):
        """
        Args:
            dataset_dir: Path to task dataset directory
            camera: Camera name ('agentview' or 'wrist')
        """
        self.dataset_dir = Path(dataset_dir)
        self.camera = camera
        
        # Load metadata
        stats_file = self.dataset_dir / "collection_stats.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {stats_file}")
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        self.num_chunks = stats['num_chunks']
        self.max_frames_per_chunk = stats['max_frames_per_chunk']
        self.tokens_per_frame = stats['tokens_per_frame']
        self.total_tokens = stats[f'total_tokens_{camera}']
        
        # Load first chunk to determine actual chunk size
        first_chunk = torch.load(self.dataset_dir / f"{camera}_tokens_00000.pt")
        self.chunk_size = first_chunk.shape[0]
        
        # Cache for current loaded chunk
        self._current_chunk_idx = 0
        self._current_chunk = first_chunk
    
    def __len__(self) -> int:
        """Return total number of tokens across all chunks."""
        return self.total_tokens
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get token at given index (lazy-loads chunks as needed).
        
        Args:
            idx: Token index (0 to len-1)
            
        Returns:
            Token embedding of shape (2048,)
        """
        if idx < 0 or idx >= self.total_tokens:
            raise IndexError(f"Index {idx} out of range [0, {self.total_tokens})")
        
        # Determine which chunk contains this token
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        # Load chunk if not currently cached
        if self._current_chunk_idx != chunk_idx:
            chunk_file = self.dataset_dir / f"{self.camera}_tokens_{chunk_idx:05d}.pt"
            self._current_chunk = torch.load(chunk_file)
            self._current_chunk_idx = chunk_idx
        
        return self._current_chunk[local_idx]
