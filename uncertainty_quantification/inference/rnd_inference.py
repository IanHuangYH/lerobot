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
        # Checkpoint path: {task_type}_{camera}/best_model.ckpt (or model.ckpt as fallback)
        model_dir = rnd_models_dir / f"{task_type}_{camera}"
        checkpoint_path = model_dir / "best_model.ckpt"
        
        # Fallback to model.ckpt if best_model.ckpt doesn't exist
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / "model.ckpt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"RND checkpoint not found in {model_dir}. "
                f"Expected 'best_model.ckpt' or 'model.ckpt'. "
                f"Make sure Phase 2 training completed successfully for {task_type}_{camera}."
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
        
        # embeddings shape: (batch_size, 256, 2048) where batch_size is number of parallel envs
        batch_size = embeddings.shape[0]
        num_tokens = embeddings.shape[1]
        
        # Reshape to (batch_size * num_tokens, 2048) to process all tokens from all batches
        tokens = embeddings.reshape(-1, embeddings.shape[-1])  # (batch_size * 256, 2048)
        
        # Convert to float32 if needed (Pi0.5 uses bfloat16, RND models use float32)
        if tokens.dtype != torch.float32:
            tokens = tokens.to(dtype=torch.float32)
        
        # Ensure tokens are on the same device as the RND model
        model_device = next(model.parameters()).device
        if tokens.device != model_device:
            tokens = tokens.to(model_device)
        
        with torch.no_grad():
            # Batch process all tokens through RND
            token_uncertainty = model(tokens)
            
            # Reshape back to (batch_size, num_tokens)
            token_uncertainty = token_uncertainty.reshape(batch_size, num_tokens)
        
        # Store per-token uncertainties (keep batch dimension)
        token_uncertainties[camera] = token_uncertainty  # (batch_size, 256)
        
        # Compute overall camera uncertainty (mean across tokens for each batch item)
        camera_uncertainty = token_uncertainty.mean(dim=-1)  # (batch_size,)
        uncertainties[camera] = camera_uncertainty  # Keep as tensor
        camera_scores.append(camera_uncertainty)
        
        # Generate spatial uncertainty map if requested
        if return_spatial_maps:
            # Reshape (batch_size, 256) -> (batch_size, 16, 16) spatial map
            spatial_map = token_uncertainty.view(batch_size, 16, 16)
            spatial_maps[camera] = spatial_map
    
    # Compute overall uncertainty (mean across cameras for each batch item)
    if len(camera_scores) > 0:
        # Stack camera scores: list of (batch_size,) -> (num_cameras, batch_size)
        stacked_scores = torch.stack(camera_scores, dim=0)  # (num_cameras, batch_size)
        uncertainties['overall'] = stacked_scores.mean(dim=0)  # (batch_size,)
    else:
        # No valid cameras, return zeros
        batch_size = next(iter(image_embeddings.values())).shape[0]
        uncertainties['overall'] = torch.zeros(batch_size)
    
    # Add spatial maps and token uncertainties to output
    if return_spatial_maps:
        uncertainties['spatial_maps'] = spatial_maps
    
    uncertainties['token_uncertainties'] = token_uncertainties
    
    return uncertainties


def load_rnd_models_for_policy(
    policy,
    task: str,
    cameras: list[str] = ['agentview', 'wrist'],
    rnd_models_dir: Optional[Path] = None,
    device: str = "cpu",
) -> None:
    """
    Load RND models and inject them into a policy for uncertainty prediction.
    
    This is the main entry point for enabling uncertainty quantification during evaluation.
    
    Args:
        policy: Policy object (e.g., PI05Policy instance)
        task: Task name or comma-separated list of tasks (e.g., 'libero_object' or 'libero_object_0')
        cameras: List of camera names to load models for
        rnd_models_dir: Directory containing trained RND models
        device: Device to load models on
    
    Raises:
        ValueError: If task type cannot be determined or models not found
        AttributeError: If policy doesn't support uncertainty prediction
    
    Example:
        >>> policy = PI05Policy(config)
        >>> load_rnd_models_for_policy(policy, 'libero_object', device='cuda:0')
        >>> # Now policy.get_uncertainty_scores() will work
    """
    # Check if policy supports uncertainty prediction
    if not hasattr(policy, 'enable_uncertainty_prediction'):
        raise AttributeError(
            f"Policy type '{type(policy).__name__}' does not support uncertainty prediction. "
            "Make sure you're using PI05Policy with the uncertainty extensions."
        )
    
    # Extract task type from task name
    # Handle comma-separated tasks (take the first one)
    if ',' in task:
        task = task.split(',')[0].strip()
    
    try:
        task_type = extract_task_type_from_env_name(task)
        logging.info(f"Detected task type: {task_type} from task name: {task}")
    except ValueError as e:
        raise ValueError(
            f"Could not determine task type from task name '{task}'. "
            f"Expected format like 'libero_object' or 'libero_spatial_0'. Error: {e}"
        )
    
    # Load RND models for this task type
    try:
        rnd_models = load_rnd_models_for_task(
            task_type=task_type,
            cameras=cameras,
            rnd_models_dir=rnd_models_dir,
            device=device,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Failed to load RND models for task type '{task_type}'. "
            f"Make sure Phase 2 training has been completed. Error: {e}"
        )
    
    # Inject RND models into policy
    policy.enable_uncertainty_prediction(rnd_models)
    
    logging.info(
        f"Successfully enabled uncertainty prediction on policy with {len(rnd_models)} RND models "
        f"for task type '{task_type}'"
    )


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


def extract_episode_uncertainty(all_uncertainty_scores: list, batch_idx: int) -> list:
    """
    Extract a single episode's uncertainty scores from batched rollout data.
    
    This function processes uncertainty data collected during batched policy evaluation,
    extracting the scores for a specific episode (batch element) from the batched tensors.
    
    Args:
        all_uncertainty_scores: List of rollout step dictionaries, each containing:
            - 'step': int - Rollout step index
            - 'uncertainty': dict with batched tensors:
                - 'overall': Tensor of shape (batch_size,)
                - '{camera}': Tensor of shape (batch_size,) for each camera
                - 'spatial_maps': Dict[str, Tensor] of shape (batch_size, 16, 16)
                - 'token_uncertainties': Dict[str, Tensor] of shape (batch_size, 256)
        batch_idx: Which batch sample to extract (0 to batch_size-1)
    
    Returns:
        List of per-step uncertainty dictionaries with batch dimension removed:
            - Scalars for overall/camera scores
            - (16, 16) tensors for spatial maps
            - (256,) tensors for token uncertainties
    
    Example:
        >>> batched_scores = [{'step': 0, 'uncertainty': {'overall': tensor([0.1, 0.2])}}]
        >>> episode_0 = extract_episode_uncertainty(batched_scores, 0)
        >>> episode_0[0]['uncertainty']['overall']  # 0.1 (scalar)
    """
    extracted_scores = []
    
    for step_data in all_uncertainty_scores:
        extracted_step = {
            'step': step_data['step'],
            'uncertainty': {}
        }
        
        # Extract batch_idx from uncertainty dict
        for key, value in step_data['uncertainty'].items():
            if key == 'spatial_maps':
                # Handle nested spatial maps dict
                extracted_step['uncertainty']['spatial_maps'] = {}
                for camera, spatial_map in value.items():
                    # spatial_map shape: (batch_size, 16, 16)
                    if isinstance(spatial_map, torch.Tensor) and spatial_map.dim() == 3:
                        extracted_step['uncertainty']['spatial_maps'][camera] = spatial_map[batch_idx]
                    else:
                        extracted_step['uncertainty']['spatial_maps'][camera] = spatial_map
            
            elif key == 'token_uncertainties':
                # Handle nested token uncertainties dict
                extracted_step['uncertainty']['token_uncertainties'] = {}
                for camera, token_unc in value.items():
                    # token_unc shape: (batch_size, 256)
                    if isinstance(token_unc, torch.Tensor) and token_unc.dim() == 2:
                        extracted_step['uncertainty']['token_uncertainties'][camera] = token_unc[batch_idx]
                    else:
                        extracted_step['uncertainty']['token_uncertainties'][camera] = token_unc
            
            else:
                # Handle overall and per-camera scores (tensors of shape (batch_size,))
                if isinstance(value, torch.Tensor) and value.dim() == 1:
                    extracted_step['uncertainty'][key] = value[batch_idx].item()  # Convert to scalar
                else:
                    extracted_step['uncertainty'][key] = value
        
        extracted_scores.append(extracted_step)
    
    return extracted_scores
