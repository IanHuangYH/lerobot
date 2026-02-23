"""
Data collection for RND training on LIBERO demonstrations.

This module extracts SigLIP vision encoder embeddings from LIBERO demonstration data
to train Random Network Distillation (RND) models for uncertainty quantification.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_siglip_embeddings(
    policy,
    images_dict: Dict[str, np.ndarray],
    expected_image_keys: List[str] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Extract SigLIP vision encoder embeddings from images using Pi0.5 policy.
    
    Args:
        policy: Pi0.5 policy instance with frozen vision encoder
        images_dict: Dictionary mapping camera names to images
            Expected keys: 'agentview_rgb', 'eye_in_hand_rgb'
            Image shape: (H, W, C) in uint8 [0, 255]
        expected_image_keys: List of keys the policy expects for images.
            If None, defaults to ['image', 'image2']
        device: Device to run inference on
        
    Returns:
        Dictionary mapping camera names to token embeddings:
        {
            'agentview': torch.Tensor of shape (1, 256, 2048),
            'wrist': torch.Tensor of shape (1, 256, 2048)
        }
    """
    # Use default keys if not provided
    if expected_image_keys is None:
        expected_image_keys = ['image', 'image2']
    
    # Map LIBERO camera names to expected keys
    # Assume first key is agentview, second is wrist (eye_in_hand)
    if len(expected_image_keys) < 2:
        raise ValueError(f"Need at least 2 image keys, got {len(expected_image_keys)}")
    
    camera_mapping = {
        'agentview_rgb': expected_image_keys[0],
        'eye_in_hand_rgb': expected_image_keys[1],
    }
    
    # Prepare batch dictionary for Pi0.5
    batch = {}
    for libero_cam, policy_key in camera_mapping.items():
        if libero_cam not in images_dict:
            raise ValueError(f"Missing camera {libero_cam} in images_dict")
        
        img = images_dict[libero_cam]  # (H, W, C) uint8
        
        # Convert to torch tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img_tensor = img_tensor.unsqueeze(0)
        
        # Convert to channels-first: (1, H, W, C) -> (1, C, H, W)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        
        batch[policy_key] = img_tensor.to(device)
    
    # Extract embeddings using Pi0.5's preprocessing and vision encoder
    with torch.no_grad():
        # Preprocess images (handles resizing, normalization to [-1, 1])
        images, img_masks = policy._preprocess_images(batch)
        
        # Extract embeddings via embed_image (which calls get_image_features)
        embeddings = {}
        for idx, (img, camera_key) in enumerate(zip(images, camera_mapping.keys())):
            # Get SigLIP embeddings
            img_emb = policy.model.paligemma_with_expert.embed_image(img)  # (1, 256, 2048)
            
            # Store with simplified camera name
            cam_name = 'agentview' if 'agentview' in camera_key else 'wrist'
            embeddings[cam_name] = img_emb
    
    return embeddings


def sample_tokens_from_embeddings(
    embeddings: torch.Tensor,
    num_tokens: int = 64,
    random_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Randomly sample tokens from embedding tensor.
    
    Args:
        embeddings: Tensor of shape (1, 256, 2048)
        num_tokens: Number of tokens to sample (default: 64)
        random_seed: Random seed for reproducibility
        
    Returns:
        Sampled tokens of shape (num_tokens, 2048)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # embeddings shape: (1, 256, 2048)
    embeddings = embeddings.squeeze(0)  # (256, 2048)
    
    # Randomly sample token indices
    num_total_tokens = embeddings.shape[0]
    if num_tokens > num_total_tokens:
        raise ValueError(f"num_tokens ({num_tokens}) > total tokens ({num_total_tokens})")
    
    indices = torch.randperm(num_total_tokens)[:num_tokens]
    sampled = embeddings[indices]  # (num_tokens, 2048)
    
    return sampled


def load_libero_demo_file(demo_file_path: Path) -> List[Dict]:
    """
    Load demonstrations from a LIBERO HDF5 file.
    
    Args:
        demo_file_path: Path to .hdf5 demo file
        
    Returns:
        List of demo dictionaries, each containing:
        {
            'agentview_rgb': np.ndarray of shape (T, H, W, C),
            'eye_in_hand_rgb': np.ndarray of shape (T, H, W, C),
            'actions': np.ndarray of shape (T, 7),
            'demo_id': int
        }
    """
    demos = []
    
    with h5py.File(demo_file_path, 'r') as f:
        # Get all demo keys (demo_0, demo_1, ...)
        demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        
        for demo_key in demo_keys:
            demo_group = f['data'][demo_key]
            
            # Extract images and actions
            demo_data = {
                'agentview_rgb': demo_group['obs']['agentview_rgb'][:],
                'eye_in_hand_rgb': demo_group['obs']['eye_in_hand_rgb'][:],
                'actions': demo_group['actions'][:],
                'demo_id': int(demo_key.split('_')[1])
            }
            
            demos.append(demo_data)
    
    return demos


def collect_rnd_dataset(
    policy,
    task_type: str,
    libero_dataset_dir: Path,
    output_dir: Path,
    num_episodes: Optional[int] = None,
    tokens_per_frame: int = 64,
    max_frames_per_chunk: int = 2000,
    device: str = "cuda",
    save_full_tokens: bool = False,
) -> Dict[str, int]:
    """
    Collect RND training dataset for a specific LIBERO task type.
    
    Saves data in chunks to avoid memory issues. Each chunk contains embeddings
    from up to max_frames_per_chunk frames and is saved with sequential naming:
    {camera}_tokens_00000.pt, {camera}_tokens_00001.pt, etc.
    
    Args:
        policy: Pi0.5 policy instance (frozen, evaluation mode)
        task_type: LIBERO task type ('spatial', 'object', 'goal', 'long')
        libero_dataset_dir: Path to LIBERO datasets directory
        output_dir: Path to save collected datasets
        num_episodes: Number of episodes to use (None = use all available)
        tokens_per_frame: Number of tokens to sample per camera per frame
        max_frames_per_chunk: Maximum frames per chunk file (default: 2000)
        device: Device to run inference on
        save_full_tokens: If True, save all 256 tokens instead of sampling
        
    Returns:
        Dictionary with collection statistics:
        {
            'task_type': str,
            'total_episodes': int,
            'total_frames': int,
            'tokens_per_frame': int,
            'max_frames_per_chunk': int,
            'num_chunks': int,
            'last_chunk_frames': int,
            'total_tokens_agentview': int,
            'total_tokens_wrist': int
        }
    """
    logger.info(f"Collecting RND dataset for task type: {task_type}")
    logger.info(f"Tokens per frame: {256 if save_full_tokens else tokens_per_frame}")
    logger.info(f"Max frames per chunk: {max_frames_per_chunk}")
    
    print(f"Starting data collection for task: {task_type}", flush=True)
    print(f"Tokens per frame: {256 if save_full_tokens else tokens_per_frame}", flush=True)
    print(f"Max frames per chunk: {max_frames_per_chunk}", flush=True)
    
    # Get expected image keys from policy config
    expected_image_keys = list(policy.config.image_features.keys())
    logger.info(f"Using image feature keys from policy: {expected_image_keys}")
    
    # Setup paths
    task_dir = libero_dataset_dir / f"libero_{task_type}"
    if not task_dir.exists():
        raise ValueError(f"Task directory not found: {task_dir}")
    
    output_task_dir = output_dir / task_type
    output_task_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all demo files
    demo_files = sorted(task_dir.glob("*.hdf5"))
    logger.info(f"Found {len(demo_files)} demo files")
    print(f"Found {len(demo_files)} .hdf5 demo files", flush=True)
    
    # Quick scan to get total episodes and frames (for progress estimation)
    print("Scanning dataset for size estimation...", flush=True)
    logger.info("Scanning dataset for size estimation...")
    total_episodes_available = 0
    total_frames_available = 0
    for demo_file in demo_files:
        with h5py.File(demo_file, 'r') as f:
            num_demos = len([k for k in f['data'].keys() if k.startswith('demo_')])
            total_episodes_available += num_demos
            # Quick frame count
            for demo_key in [k for k in f['data'].keys() if k.startswith('demo_')]:
                total_frames_available += f['data'][demo_key]['obs']['agentview_rgb'].shape[0]
    
    episodes_to_collect = num_episodes if num_episodes is not None else total_episodes_available
    logger.info(f"\nDataset Overview:")
    logger.info(f"  Total episodes available: {total_episodes_available}")
    logger.info(f"  Total frames available: {total_frames_available:,}")
    logger.info(f"  Episodes to collect: {episodes_to_collect}")
    if num_episodes is None:
        logger.info(f"  Frames to collect: ~{total_frames_available:,} (all)")
    else:
        avg_frames_per_ep = total_frames_available / total_episodes_available
        logger.info(f"  Frames to collect: ~{int(episodes_to_collect * avg_frames_per_ep):,} (estimated)")
    logger.info("")
    
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset Overview for {task_type}:", flush=True)
    print(f"  Total episodes available: {total_episodes_available}", flush=True)
    print(f"  Total frames available: {total_frames_available:,}", flush=True)
    print(f"  Episodes to collect: {episodes_to_collect}", flush=True)
    if num_episodes is None:
        print(f"  Frames to collect: ~{total_frames_available:,} (all)", flush=True)
    else:
        avg_frames_per_ep = total_frames_available / total_episodes_available
        print(f"  Frames to collect: ~{int(episodes_to_collect * avg_frames_per_ep):,} (estimated)", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Initialize chunk tracking
    chunk_idx = 0
    chunk_frames = 0
    total_frames = 0
    total_episodes = 0
    
    # Storage for current chunk
    agentview_tokens_chunk = []
    wrist_tokens_chunk = []
    
    def save_chunk():
        """Save current chunk to disk and clear memory."""
        nonlocal chunk_idx, chunk_frames
        nonlocal agentview_tokens_chunk, wrist_tokens_chunk
        
        if len(agentview_tokens_chunk) == 0:
            return
        
        # Calculate chunk info
        tokens_in_chunk = len(agentview_tokens_chunk) * agentview_tokens_chunk[0].shape[0]
        chunk_size_mb = tokens_in_chunk * 2048 * 4 / (1024**2)  # Each token is 2048 floats, 4 bytes each
        
        print(f"💾 Saving chunk {chunk_idx:05d}: {chunk_frames} frames, {tokens_in_chunk:,} tokens/camera, ~{chunk_size_mb:.1f} MB/camera", flush=True)
        logger.info(f"💾 Saving chunk {chunk_idx:05d}: {chunk_frames} frames, {tokens_in_chunk:,} tokens/camera, ~{chunk_size_mb:.1f} MB/camera")
        
        # Concatenate tokens for this chunk
        agentview_chunk = torch.cat(agentview_tokens_chunk, dim=0)
        wrist_chunk = torch.cat(wrist_tokens_chunk, dim=0)
        
        # Save with zero-padded index
        torch.save(agentview_chunk, output_task_dir / f"agentview_tokens_{chunk_idx:05d}.pt")
        torch.save(wrist_chunk, output_task_dir / f"wrist_tokens_{chunk_idx:05d}.pt")
        
        # Clear memory
        del agentview_chunk, wrist_chunk, agentview_tokens_chunk, wrist_tokens_chunk
        agentview_tokens_chunk = []
        wrist_tokens_chunk = []
        chunk_frames = 0
        chunk_idx += 1
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Process each demo file
    for file_idx, demo_file in enumerate(demo_files, 1):
        print(f"\n{'='*60}", flush=True)
        print(f"Processing file {file_idx}/{len(demo_files)}: {demo_file.name}", flush=True)
        print(f"{'='*60}", flush=True)
        logger.info(f"Starting file {file_idx}/{len(demo_files)}: {demo_file.name}")
        
        demos = load_libero_demo_file(demo_file)
        file_episodes_count = 0
        file_frames_count = 0
        
        # Calculate total frames in this file for progress tracking
        total_frames_in_file = sum(demo['agentview_rgb'].shape[0] for demo in demos)
        print(f"File contains {len(demos)} episodes, {total_frames_in_file} total frames", flush=True)
        logger.info(f"File contains {len(demos)} episodes, {total_frames_in_file} total frames")
        
        # Create progress bar for this file
        frames_processed_in_file = 0
        pbar = tqdm(
            total=total_frames_in_file,
            desc=f"File {file_idx}/{len(demo_files)}",
            unit="frames",
            leave=False,
            file=sys.stdout,
            ncols=100
        )
        
        for demo in demos:
            # Check if we've reached the episode limit
            if num_episodes is not None and total_episodes >= num_episodes:
                break
            
            num_frames = demo['agentview_rgb'].shape[0]
            total_frames += num_frames
            total_episodes += 1
            file_episodes_count += 1
            file_frames_count += num_frames
            
            # Process each frame in the episode
            for frame_idx in range(num_frames):
                # Prepare images for current frame
                images_dict = {
                    'agentview_rgb': demo['agentview_rgb'][frame_idx],
                    'eye_in_hand_rgb': demo['eye_in_hand_rgb'][frame_idx],
                }
                
                # Extract embeddings from vision encoder
                embeddings = extract_siglip_embeddings(
                    policy, images_dict, expected_image_keys, device
                )
                
                # Process each camera's embeddings
                for cam_name, emb in embeddings.items():
                    if save_full_tokens:
                        # Save all 256 tokens
                        tokens = emb.squeeze(0).cpu()  # (256, 2048)
                    else:
                        # Sample specified number of tokens
                        tokens = sample_tokens_from_embeddings(
                            emb, 
                            num_tokens=tokens_per_frame
                        ).cpu()  # (tokens_per_frame, 2048)
                    
                    # Accumulate tokens for current chunk
                    if cam_name == 'agentview':
                        agentview_tokens_chunk.append(tokens)
                    else:
                        wrist_tokens_chunk.append(tokens)
                
                chunk_frames += 1
                frames_processed_in_file += 1
                
                # Update progress bar
                pbar.update(1)
                
                # Periodic GPU cache clearing
                if chunk_frames % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Save chunk when limit reached
                if chunk_frames >= max_frames_per_chunk:
                    save_chunk()
        
        # Close progress bar for this file
        pbar.close()
        
        # Print summary for this file
        print(f"✓ Completed {demo_file.name}:", flush=True)
        print(f"  This file: {file_episodes_count} episodes, {file_frames_count} frames", flush=True)
        print(f"  Overall Progress: {total_episodes} episodes, {total_frames} frames, {chunk_idx} chunks saved", flush=True)
        print(f"  Files remaining: {len(demo_files) - file_idx}\n", flush=True)
        
        logger.info(f"✓ Completed {demo_file.name}: {file_episodes_count} episodes, {file_frames_count} frames")
        logger.info(f"  Overall Progress: {total_episodes} episodes, {total_frames} frames, {chunk_idx} chunks saved")
        logger.info(f"  Files remaining: {len(demo_files) - file_idx}\n")
        
        # Check if we've reached the episode limit
        if num_episodes is not None and total_episodes >= num_episodes:
            break
    
    # Save final chunk (may be smaller than max_frames_per_chunk)
    last_chunk_frames = chunk_frames
    if chunk_frames > 0:
        print(f"\n💾 Saving final chunk...", flush=True)
        save_chunk()
    
    # Calculate final statistics
    tokens_per_saved_frame = 256 if save_full_tokens else tokens_per_frame
    total_tokens_per_camera = total_frames * tokens_per_saved_frame
    
    stats = {
        'task_type': task_type,
        'total_episodes': total_episodes,
        'total_frames': total_frames,
        'tokens_per_frame': tokens_per_saved_frame,
        'max_frames_per_chunk': max_frames_per_chunk,
        'num_chunks': chunk_idx,
        'last_chunk_frames': last_chunk_frames,
        'total_tokens_agentview': total_tokens_per_camera,
        'total_tokens_wrist': total_tokens_per_camera,
    }
    
    print(f"\n{'='*60}", flush=True)
    print(f"Collection complete for {task_type}:", flush=True)
    print(f"  Episodes: {stats['total_episodes']}", flush=True)
    print(f"  Frames: {stats['total_frames']}", flush=True)
    print(f"  Chunks: {stats['num_chunks']}", flush=True)
    print(f"  Tokens per camera: {stats['total_tokens_agentview']:,}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    logger.info(f"Collection complete:")
    logger.info(f"  Episodes: {stats['total_episodes']}")
    logger.info(f"  Frames: {stats['total_frames']}")
    logger.info(f"  Chunks: {stats['num_chunks']}")
    logger.info(f"  Tokens per camera: {stats['total_tokens_agentview']:,}")
    
    # Save statistics
    import json
    with open(output_task_dir / "collection_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats
