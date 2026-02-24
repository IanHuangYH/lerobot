"""
Utilities to load chunked RND training datasets.

The RND dataset is saved as multiple chunk files to avoid memory issues:
- agentview_tokens_00000.pt, agentview_tokens_00001.pt, ...
- wrist_tokens_00000.pt, wrist_tokens_00001.pt, ...
- collection_stats.json (metadata)
"""

import json
from pathlib import Path
from typing import Tuple, Dict
from collections import OrderedDict

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
    
    Uses LRU cache to keep multiple chunks in memory, balancing speed and memory.
    
    Memory-Speed Trade-off:
    - max_cached_chunks=1: ~2 GB RAM, slower (frequent disk reads)
    - max_cached_chunks=4: ~8 GB RAM, faster (fewer disk reads)
    - max_cached_chunks=8: ~16 GB RAM, fastest (rare disk reads)
    
    Example usage:
        dataset = ChunkedRNDDataset(
            "rnd_dataset/object", 
            "agentview",
            max_cached_chunks=4  # Keep 4 chunks (~8 GB) in memory
        )
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
        
        for batch in dataloader:
            # batch shape: (256, 2048)
            train_rnd(batch)
    """
    
    def __init__(self, dataset_dir: Path, camera: str, max_cached_chunks: int = 4, ratio_preload: float = 0.6):
        """
        Args:
            dataset_dir: Path to task dataset directory
            camera: Camera name ('agentview' or 'wrist')
            max_cached_chunks: Maximum number of chunks to keep in memory (default: 4)
                              Higher = faster training but more RAM
                              Recommended: 4 for 16GB RAM, 2 for 8GB RAM
        """
        self.dataset_dir = Path(dataset_dir)
        self.camera = camera
        self.max_cached_chunks = max_cached_chunks
        
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
        
        # LRU cache for multiple chunks (OrderedDict maintains insertion order)
        self._chunk_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        
        # Statistics for monitoring cache performance
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Smart pre-loading: if caching >60% of chunks, pre-load them upfront
        if max_cached_chunks >= ratio_preload * self.num_chunks:
            chunks_to_preload = min(max_cached_chunks, self.num_chunks)
            print(f"Cache size ({max_cached_chunks}) >= 60% of chunks ({self.num_chunks})")
            print(f"Pre-loading {chunks_to_preload} chunks into cache (~{chunks_to_preload * 2:.0f} GB)...")
            
            # Pre-load chunks into cache
            self._chunk_cache[0] = first_chunk  # Already loaded
            for i in range(1, chunks_to_preload):
                chunk_file = self.dataset_dir / f"{camera}_tokens_{i:05d}.pt"
                chunk = torch.load(chunk_file)
                self._chunk_cache[i] = chunk
                if (i + 1) % 5 == 0 or i == chunks_to_preload - 1:
                    print(f"  Loaded {i+1}/{chunks_to_preload} chunks...")
            
            print(f"✓ Pre-loaded {chunks_to_preload} chunks! Cache ready, Memory: ~{chunks_to_preload * 2:.0f} GB")
        else:
            # Just load first chunk for normal caching
            print(f"Using lazy chunk caching: {max_cached_chunks}/{self.num_chunks} chunks (~{max_cached_chunks * 2:.0f} GB)")
            self._chunk_cache[0] = first_chunk  # Cache first chunk
    
    def __len__(self) -> int:
        """Return total number of tokens across all chunks."""
        return self.total_tokens
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get token at given index (lazy-loads chunks as needed with LRU caching).
        
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
        
        # Check if chunk is in cache
        if chunk_idx in self._chunk_cache:
            # Cache hit! Move to end (most recently used)
            self._chunk_cache.move_to_end(chunk_idx)
            self._cache_hits += 1
            chunk = self._chunk_cache[chunk_idx]
        else:
            # Cache miss - need to load from disk
            self._cache_misses += 1
            chunk_file = self.dataset_dir / f"{self.camera}_tokens_{chunk_idx:05d}.pt"
            chunk = torch.load(chunk_file)
            
            # Add to cache
            self._chunk_cache[chunk_idx] = chunk
            
            # Evict oldest chunk if cache is full
            if len(self._chunk_cache) > self.max_cached_chunks:
                # Remove least recently used (first item)
                evicted_idx = next(iter(self._chunk_cache))
                del self._chunk_cache[evicted_idx]
        
        return chunk[local_idx]
    
    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get cache performance statistics.
        
        Returns:
            Dict with cache hit rate and memory usage info
        """
        total_accesses = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached_chunks': len(self._chunk_cache),
            'max_cached_chunks': self.max_cached_chunks,
            'estimated_memory_gb': len(self._chunk_cache) * self.chunk_size * 2048 * 4 / (1024**3)
        }
