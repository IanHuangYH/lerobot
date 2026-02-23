"""
Data collection for RND training on LIBERO demonstrations.

This module extracts SigLIP vision encoder embeddings from LIBERO demonstration data
to train Random Network Distillation (RND) models for uncertainty quantification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_siglip_embeddings(
    policy,
    images_dict: Dict[str, np.ndarray],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Extract SigLIP vision encoder embeddings from images using Pi0.5 policy.
    
    Args:
        policy: Pi0.5 policy instance with frozen vision encoder
        images_dict: Dictionary mapping camera names to images
            Expected keys: 'agentview_rgb', 'eye_in_hand_rgb'
            Image shape: (H, W, C) in uint8 [0, 255]
        device: Device to run inference on
        
    Returns:
        Dictionary mapping camera names to token embeddings:
        {
            'agentview': torch.Tensor of shape (1, 256, 2048),
            'wrist': torch.Tensor of shape (1, 256, 2048)
        }
    """
    # Map LIBERO camera names to Pi0.5 expected names
    camera_mapping = {
        'agentview_rgb': 'image',
        'eye_in_hand_rgb': 'image2',
    }
    
    # Prepare batch dictionary for Pi0.5
    batch = {}
    for libero_cam, pi05_cam in camera_mapping.items():
        if libero_cam not in images_dict:
            raise ValueError(f"Missing camera {libero_cam} in images_dict")
        
        img = images_dict[libero_cam]  # (H, W, C) uint8
        
        # Convert to torch tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img_tensor = img_tensor.unsqueeze(0)
        
        # Convert to channels-first: (1, H, W, C) -> (1, C, H, W)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        
        batch[pi05_cam] = img_tensor.to(device)
    
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
    device: str = "cuda",
    save_full_tokens: bool = False,
) -> Dict[str, int]:
    """
    Collect RND training dataset for a specific LIBERO task type.
    
    Args:
        policy: Pi0.5 policy instance (frozen, evaluation mode)
        task_type: LIBERO task type ('spatial', 'object', 'goal', 'long')
        libero_dataset_dir: Path to LIBERO datasets directory
        output_dir: Path to save collected datasets
        num_episodes: Number of episodes to use (None = use all available)
        tokens_per_frame: Number of tokens to sample per camera per frame
        device: Device to run inference on
        save_full_tokens: If True, save all 256 tokens instead of sampling
        
    Returns:
        Dictionary with collection statistics:
        {
            'total_episodes': int,
            'total_frames': int,
            'total_tokens_agentview': int,
            'total_tokens_wrist': int
        }
    """
    logger.info(f"Collecting RND dataset for task type: {task_type}")
    logger.info(f"Tokens per frame: {tokens_per_frame if not save_full_tokens else 256}")
    
    # Setup paths
    task_dir = libero_dataset_dir / f"libero_{task_type}"
    if not task_dir.exists():
        raise ValueError(f"Task directory not found: {task_dir}")
    
    output_task_dir = output_dir / task_type
    output_task_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all demo files
    demo_files = sorted(task_dir.glob("*.hdf5"))
    logger.info(f"Found {len(demo_files)} demo files")
    
    # Initialize token storage
    agentview_tokens = []
    wrist_tokens = []
    
    total_frames = 0
    total_episodes = 0
    
    # Process each demo file
    for demo_file in tqdm(demo_files, desc=f"Processing {task_type} files"):
        demos = load_libero_demo_file(demo_file)
        
        for demo in demos:
            # Check if we've reached the episode limit
            if num_episodes is not None and total_episodes >= num_episodes:
                break
            
            num_frames = demo['agentview_rgb'].shape[0]
            total_frames += num_frames
            total_episodes += 1
            
            # Process each frame
            for frame_idx in range(num_frames):
                # Prepare images for current frame
                images_dict = {
                    'agentview_rgb': demo['agentview_rgb'][frame_idx],
                    'eye_in_hand_rgb': demo['eye_in_hand_rgb'][frame_idx],
                }
                
                # Extract embeddings
                embeddings = extract_siglip_embeddings(policy, images_dict, device)
                
                # Sample or save full tokens
                for cam_name, emb in embeddings.items():
                    if save_full_tokens:
                        # Save all 256 tokens
                        tokens = emb.squeeze(0).cpu()  # (256, 2048)
                    else:
                        # Sample tokens
                        tokens = sample_tokens_from_embeddings(
                            emb, 
                            num_tokens=tokens_per_frame
                        ).cpu()  # (tokens_per_frame, 2048)
                    
                    # Store tokens
                    if cam_name == 'agentview':
                        agentview_tokens.append(tokens)
                    else:
                        wrist_tokens.append(tokens)
        
        # Check if we've reached the episode limit
        if num_episodes is not None and total_episodes >= num_episodes:
            break
    
    # Concatenate all tokens
    logger.info("Concatenating tokens...")
    agentview_tokens = torch.cat(agentview_tokens, dim=0)  # (N, 2048)
    wrist_tokens = torch.cat(wrist_tokens, dim=0)  # (N, 2048)
    
    # Save datasets
    logger.info("Saving datasets...")
    torch.save(agentview_tokens, output_task_dir / "agentview_tokens.pt")
    torch.save(wrist_tokens, output_task_dir / "wrist_tokens.pt")
    
    stats = {
        'total_episodes': total_episodes,
        'total_frames': total_frames,
        'total_tokens_agentview': agentview_tokens.shape[0],
        'total_tokens_wrist': wrist_tokens.shape[0],
    }
    
    logger.info(f"Collection complete:")
    logger.info(f"  Episodes: {stats['total_episodes']}")
    logger.info(f"  Frames: {stats['total_frames']}")
    logger.info(f"  Agentview tokens: {stats['total_tokens_agentview']}")
    logger.info(f"  Wrist tokens: {stats['total_tokens_wrist']}")
    
    # Save statistics
    import json
    with open(output_task_dir / "collection_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats
