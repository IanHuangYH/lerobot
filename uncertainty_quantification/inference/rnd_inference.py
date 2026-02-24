"""
RND inference utilities for uncertainty quantification during policy evaluation.

This module provides functions to:
1. Load trained RND models based on task type
2. Compute uncertainty scores from vision encoder embeddings
3. Generate spatial uncertainty heatmaps
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from uncertainty_quantification.rnd_models.rnd_models import RND_OE


def extract_task_type_from_env_name(env_name: str) -> str:
    """
    Extract LIBERO task type from environment name.
    
    Args:
        env_name: Environment name (e.g., 'libero_object_0', 'libero_spatial_5')
    
    Returns:
        task_type: One of 'spatial', 'object', 'goal', 'long'
    
    Raises:
        ValueError: If task type cannot be determined
    
    Examples:
        >>> extract_task_type_from_env_name('libero_object_0')
        'object'
        >>> extract_task_type_from_env_name('libero_spatial_5')
        'spatial'
        >>> extract_task_type_from_env_name('libero_goal_3')
        'goal'
        >>> extract_task_type_from_env_name('libero_10_7')
        'long'
    """
    # Handle libero_10 (long horizon tasks)
    if 'libero_10' in env_name.lower():
        return 'long'
    
    # Extract task type from pattern: libero_{type}_{id}
    pattern = r'libero_(\w+)_\d+'
    match = re.search(pattern, env_name.lower())
    
    if match:
        task_type = match.group(1)
        valid_types = ['spatial', 'object', 'goal', 'long']
        
        if task_type in valid_types:
            return task_type
        else:
            raise ValueError(
                f"Unknown LIBERO task type: {task_type}. "
                f"Expected one of {valid_types}"
            )
    
    # Fallback: try to extract directly
    for task_type in ['spatial', 'object', 'goal']:
        if task_type in env_name.lower():
            return task_type
    
    raise ValueError(
        f"Could not extract task type from environment name: {env_name}. "
        f"Expected format: 'libero_{{type}}_{{id}}' where type is spatial/object/goal/long"
    )


def load_rnd_models_for_task(
    task_type: str,
    cameras: list[str] = ['agentview', 'wrist'],
    rnd_models_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, RND_OE]:
    """
    Load RND models for a specific task type and cameras.
    
    Args:
        task_type: LIBERO task type ('spatial', 'object', 'goal', 'long')
        cameras: List of camera names to load models for
        rnd_models_dir: Directory containing trained RND models
                       (default: lerobot/uncertainty_quantification/rnd_save_models/)
        device: Device to load models on
    
    Returns:
        camera_models: Dict mapping camera names to loaded RND models
                      e.g., {'agentview': RND_OE, 'wrist': RND_OE}
    
    Raises:
        FileNotFoundError: If checkpoint files are not found
        ValueError: If task_type is invalid
    """
    valid_types = ['spatial', 'object', 'goal', 'long']
    if task_type not in valid_types:
        raise ValueError(
            f"Invalid task type: {task_type}. Expected one of {valid_types}"
        )
    
    # Default RND models directory
    if rnd_models_dir is None:
        # Assuming this script is called from lerobot/ directory
        rnd_models_dir = Path(__file__).parent.parent / 'rnd_save_models'
    else:
        rnd_models_dir = Path(rnd_models_dir)
    
    if not rnd_models_dir.exists():
        raise FileNotFoundError(
            f"RND models directory not found: {rnd_models_dir}. "
            f"Make sure Phase 2 (RND training) has been completed."
        )
    
    camera_models = {}
    
    for camera in cameras:
        # Checkpoint path: {task_type}_{camera}/model.ckpt
        model_dir = rnd_models_dir / f"{task_type}_{camera}"
        checkpoint_path = model_dir / "model.ckpt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"RND checkpoint not found: {checkpoint_path}. "
                f"Expected format: {rnd_models_dir}/{task_type}_{camera}/model.ckpt"
            )
        
        logging.info(f"Loading RND model for {task_type}/{camera} from {checkpoint_path}")
        
        # Load model
        model = RND_OE.load_checkpoint(checkpoint_path, device=device)
        model.eval()  # Set to evaluation mode
        
        camera_models[camera] = model
    
    logging.info(
        f"Successfully loaded {len(camera_models)} RND models for task type '{task_type}'"
    )
    
    return camera_models


def compute_uncertainty_scores(
    image_embeddings: Dict[str, torch.Tensor],
    rnd_models: Dict[str, RND_OE],
    return_spatial_maps: bool = True,
) -> Dict[str, any]:
    """
    Compute uncertainty scores from image embeddings using RND models.
    
    Args:
        image_embeddings: Dict mapping camera names to SigLIP embeddings
                         Format: {'agentview': (1, 256, 2048), 'wrist': (1, 256, 2048)}
        rnd_models: Dict mapping camera names to RND models
        return_spatial_maps: Whether to return spatial uncertainty maps (16x16)
    
    Returns:
        uncertainty_dict: Dictionary containing:
            - 'overall': float - Combined uncertainty across cameras
            - '{camera}': float - Per-camera uncertainty score
            - 'spatial_maps': Dict[str, Tensor] - Spatial maps (16x16) if requested
            - 'token_uncertainties': Dict[str, Tensor] - Per-token uncertainties (256,)
    
    Example:
        >>> uncertainties = compute_uncertainty_scores(
        ...     image_embeddings={'agentview': emb1, 'wrist': emb2},
        ...     rnd_models={'agentview': model1, 'wrist': model2}
        ... )
        >>> print(uncertainties['overall'])  # 0.0234
        >>> print(uncertainties['spatial_maps']['agentview'].shape)  # (16, 16)
    """
    uncertainties = {}
    token_uncertainties = {}
    spatial_maps = {}
    
    camera_scores = []
    
    for camera, embeddings in image_embeddings.items():
        if camera not in rnd_models:
            logging.warning(
                f"Camera '{camera}' has embeddings but no RND model. Skipping."
            )
            continue
        
        model = rnd_models[camera]
        
        # embeddings shape: (1, 256, 2048)
        # Remove batch dimension for processing
        tokens = embeddings.squeeze(0)  # (256, 2048)
        
        with torch.no_grad():
            # Batch process all tokens through RND
            token_uncertainty = model(tokens)  # (256,) - per-token uncertainty scores
        
        # Store per-token uncertainties
        token_uncertainties[camera] = token_uncertainty
        
        # Compute overall camera uncertainty (mean across tokens)
        camera_uncertainty = token_uncertainty.mean().item()
        uncertainties[camera] = camera_uncertainty
        camera_scores.append(camera_uncertainty)
        
        # Generate spatial uncertainty map if requested
        if return_spatial_maps:
            # Reshape (256,) -> (16, 16) spatial map
            spatial_map = token_uncertainty.view(16, 16)
            spatial_maps[camera] = spatial_map
    
    # Compute overall uncertainty (mean across cameras)
    if len(camera_scores) > 0:
        uncertainties['overall'] = sum(camera_scores) / len(camera_scores)
    else:
        uncertainties['overall'] = 0.0
    
    # Add spatial maps and token uncertainties to output
    if return_spatial_maps:
        uncertainties['spatial_maps'] = spatial_maps
    
    uncertainties['token_uncertainties'] = token_uncertainties
    
    return uncertainties


def upsample_spatial_map(
    spatial_map: torch.Tensor,
    target_size: tuple = (224, 224),
) -> torch.Tensor:
    """
    Upsample spatial uncertainty map to match image resolution.
    
    Args:
        spatial_map: Uncertainty map of shape (16, 16)
        target_size: Target image size (default: 224x224 for SigLIP input)
    
    Returns:
        upsampled_map: Uncertainty map of shape target_size
    
    Example:
        >>> spatial_map = uncertainties['spatial_maps']['agentview']  # (16, 16)
        >>> heatmap = upsample_spatial_map(spatial_map, (224, 224))  # (224, 224)
    """
    # Add batch and channel dimensions: (16, 16) -> (1, 1, 16, 16)
    spatial_map = spatial_map.unsqueeze(0).unsqueeze(0)
    
    # Upsample using bilinear interpolation
    upsampled = F.interpolate(
        spatial_map,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    
    # Remove batch and channel dimensions: (1, 1, H, W) -> (H, W)
    upsampled = upsampled.squeeze(0).squeeze(0)
    
    return upsampled
