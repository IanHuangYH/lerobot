#!/usr/bin/env python
"""
Visualize action-to-image attention maps from saved Pi0.5 attention weights.

This script extracts attention from action tokens to image patches and creates heatmap visualizations.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def extract_action_to_img_attention(
    attention_weights: torch.Tensor,
    action_indices: List[int],
    num_img_patches: int = 256,
    camera_index: int = 0,
    num_cameras: int = 1,
    head_aggregation: str = "mean",
) -> np.ndarray:
    """
    Extract attention from action tokens to image patches.
    
    Args:
        attention_weights: Tensor of shape [batch, heads, query_len, key_len]
                          where query_len = 50 (suffix tokens: 1 timestep + 49 actions)
                          and key_len depends on num_cameras:
                            - 1 camera: 506 (256 + 200 language + 50 suffix)
                            - 2 cameras: 762 (512 + 200 language + 50 suffix)  
                            - 3 cameras: 1018 (768 + 200 language + 50 suffix) ← Pi0.5 LIBERO default
        action_indices: List of action timestep indices to extract (e.g., [0, 1, 2, ...])
                       Note: These are 0-indexed action timesteps, which map to suffix tokens 1-49
        num_img_patches: Number of patches per camera (default 256 for 16x16 grid)
        camera_index: Which camera to extract (0=first, 1=second, etc.)
        num_cameras: Total number of cameras in the scene
        head_aggregation: How to aggregate attention heads ("mean", "max", or "sum")
    
    Returns:
        Array of shape [len(action_indices), sqrt(num_img_patches), sqrt(num_img_patches)]
    """
    # attention_weights shape: [batch=1, heads=8, query=50, key=total_patches+lang+suffix]
    
    # Remove batch dimension
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.squeeze(0)  # [heads, query, key]
    
    # Calculate patch range for the specified camera
    start_idx = camera_index * num_img_patches
    end_idx = start_idx + num_img_patches
    
    # Extract attention to this camera's patches
    action_to_img = attention_weights[:, :, start_idx:end_idx]  # [heads, query=50, img_patches=256]
    
    # Map action indices to suffix token indices
    # Suffix tokens: [0=timestep, 1=action_0, 2=action_1, ..., 49=action_48]
    # So action_idx=0 → token_idx=1, action_idx=1 → token_idx=2, etc.
    suffix_token_indices = [idx + 1 for idx in action_indices]
    
    # Select specific action tokens
    action_to_img = action_to_img[:, suffix_token_indices, :]  # [heads, len(action_indices), 256]
    
    # Aggregate across heads
    if head_aggregation == "mean":
        action_to_img = action_to_img.mean(dim=0)  # [len(action_indices), 256]
    elif head_aggregation == "max":
        action_to_img = action_to_img.max(dim=0)[0]
    elif head_aggregation == "sum":
        action_to_img = action_to_img.sum(dim=0)
    else:
        raise ValueError(f"Unknown head_aggregation: {head_aggregation}")
    
    # Reshape to 2D grid (assuming square grid)
    grid_size = int(np.sqrt(num_img_patches))
    assert grid_size * grid_size == num_img_patches, f"num_img_patches must be square, got {num_img_patches}"
    
    action_to_img = action_to_img.reshape(len(action_indices), grid_size, grid_size)
    
    # Convert to float32 before numpy (bfloat16 not supported by numpy)
    return action_to_img.to(torch.float32).cpu().numpy()


def visualize_attention_heatmap(
    attention_map: np.ndarray,
    save_path: Path,
    img_size: int = 224,
    colormap: str = "hot",
    title: str = None,
    model_space_to_camera_space: bool = False,
):
    """
    Visualize a single attention heatmap and save as PNG.
    
    COORDINATE SYSTEMS:
    - Model space: What the model sees after LiberoProcessor (ALL cameras flipped [::-1, ::-1])
    - Camera space: Original camera orientation (what we want to visualize on)
    
    Args:
        attention_map: 2D array of shape [grid_size, grid_size] (e.g., [16, 16])
                      NOTE: This is in MODEL SPACE (flipped by LiberoProcessor)
        save_path: Path to save the PNG file
        img_size: Target image size for upsampling (default 224)
        colormap: Matplotlib colormap name
        title: Optional title for the plot
        model_space_to_camera_space: If True, transform attention from model space back to camera space
                                     This undoes LiberoProcessor's flip: [::-1, ::-1]
    """
    from scipy.ndimage import zoom
    
    # Transform from model space to camera space if requested
    # LiberoProcessor does: torch.flip(img, dims=[2,3]) which is equivalent to [::-1, ::-1]
    # To reverse this, we apply the same flip operation
    if model_space_to_camera_space:
        attention_map = np.flip(attention_map, axis=(0, 1))  # Undo LiberoProcessor flip
    
    # Upsample to target image size
    grid_size = attention_map.shape[0]
    zoom_factor = img_size / grid_size
    upsampled = zoom(attention_map, zoom_factor, order=1)  # Bilinear interpolation
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    
    # Plot heatmap
    im = ax.imshow(upsampled, cmap=colormap, vmin=0, vmax=attention_map.max())
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize action-to-image attention maps from Pi0.5")
    parser.add_argument(
        "--attention_file",
        type=str,
        required=True,
        help="Path to saved attention .pt file (e.g., episode_00000_attention.pt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save visualization PNG files"
    )
    parser.add_argument(
        "--rollout_step",
        type=int,
        default=0,
        help="Which rollout step to visualize (default: 0, first action prediction)"
    )
    parser.add_argument(
        "--denoising_step",
        type=int,
        default=9,
        help="Which denoising step to visualize (default: 9, final denoising step)"
    )
    parser.add_argument(
        "--act_vis_dim",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Which action timesteps to visualize (default: 0-9, first 10 actions)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[17],
        help="Which transformer layers to visualize (default: 17, last layer)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Output image size for heatmaps (default: 224)"
    )
    parser.add_argument(
        "--head_aggregation",
        type=str,
        choices=["mean", "max", "sum"],
        default="mean",
        help="How to aggregate attention heads (default: mean)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="hot",
        help="Matplotlib colormap for heatmaps (default: hot)"
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Which camera to visualize (0=first camera, 1=second camera, etc.)"
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=None,
        help="Total number of cameras (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Load attention file
    print(f"Loading attention from: {args.attention_file}")
    data = torch.load(args.attention_file, map_location='cpu')
    
    # Extract rollout steps
    rollout_steps = data['rollout_steps']
    
    # Find the requested rollout step
    rollout_data = None
    for step_data in rollout_steps:
        if step_data['rollout_step'] == args.rollout_step:
            rollout_data = step_data
            break
    
    if rollout_data is None:
        available_steps = [s['rollout_step'] for s in rollout_steps]
        raise ValueError(
            f"Rollout step {args.rollout_step} not found. Available steps: {available_steps}"
        )
    
    # Extract attention maps for the denoising step
    attention_maps = rollout_data['attention_maps']
    if args.denoising_step not in attention_maps:
        available_steps = list(attention_maps.keys())
        raise ValueError(
            f"Denoising step {args.denoising_step} not found. Available steps: {available_steps}"
        )
    
    denoising_data = attention_maps[args.denoising_step]
    layer_attentions = denoising_data['attention_weights']
    
    # Auto-detect number of cameras if not specified
    sample_attention = layer_attentions[list(layer_attentions.keys())[0]]
    key_len = sample_attention.shape[-1]  # e.g., 506, 762, or 1018
    
    if args.num_cameras is None:
        # Heuristic: key_len = num_cameras * 256 + 200 language + 50 suffix
        # Pi0.5 uses tokenizer_max_length=200 for language tokens
        # 1 camera: 256 + 200 + 50 = 506
        # 2 cameras: 512 + 200 + 50 = 762
        # 3 cameras: 768 + 200 + 50 = 1018 ← Pi0.5 LIBERO (2 real + 1 empty)
        # 4 cameras: 1024 + 200 + 50 = 1274
        lang_suffix = 200 + 50  # language + suffix tokens
        num_patches = key_len - lang_suffix
        args.num_cameras = num_patches // 256
        
        if num_patches % 256 != 0:
            print(f"Warning: key_len={key_len} doesn't match expected pattern. "
                  f"Calculated {args.num_cameras} cameras with {num_patches % 256} extra patches.")
        
        print(f"Auto-detected {args.num_cameras} camera(s) from key_len={key_len} "
              f"({num_patches} image patches + {lang_suffix} lang+suffix)")
    
    # Validate camera_index
    if args.camera_index >= args.num_cameras:
        raise ValueError(f"camera_index={args.camera_index} but only {args.num_cameras} cameras detected")
    
    camera_names = ["agentview", "wrist_view", "camera_2", "camera_3"]
    camera_name = camera_names[args.camera_index] if args.camera_index < len(camera_names) else f"camera_{args.camera_index}"
    
    print(f"Rollout step: {args.rollout_step}")
    print(f"Denoising step: {args.denoising_step}")
    print(f"Number of cameras: {args.num_cameras}")
    print(f"Visualizing camera: {args.camera_index} ({camera_name})")
    print(f"Available layers: {list(layer_attentions.keys())}")
    print(f"Action indices to visualize: {args.act_vis_dim}")
    print(f"Layers to visualize: {args.layers}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize each requested layer
    for layer_idx in args.layers:
        if layer_idx not in layer_attentions:
            print(f"Warning: Layer {layer_idx} not found, skipping")
            continue
        
        attention_weights = layer_attentions[layer_idx]
        print(f"\nProcessing Layer {layer_idx}:")
        print(f"  Attention shape: {attention_weights.shape}")
        
        # Extract action-to-image attention
        action_to_img = extract_action_to_img_attention(
            attention_weights,
            action_indices=args.act_vis_dim,
            num_img_patches=256,
            camera_index=args.camera_index,
            num_cameras=args.num_cameras,
            head_aggregation=args.head_aggregation,
        )
        
        print(f"  Extracted action-to-img shape: {action_to_img.shape}")
        
        # Visualize each action
        for i, action_idx in enumerate(args.act_vis_dim):
            heatmap = action_to_img[i]
            
            # Create filename
            filename = f"attention_act_to_img_{camera_name}_layer_{layer_idx:02d}_act_{action_idx:02d}.png"
            save_path = output_dir / filename
            
            # Create title
            title = f"Layer {layer_idx} | Action {action_idx} | {camera_name}\nRollout Step {args.rollout_step} | Denoising Step {args.denoising_step}"
            
            # COORDINATE SYSTEM TRANSFORMATION:
            # 
            # 1. Attention weights are in MODEL SPACE:
            #    - LiberoProcessor flips ALL cameras: torch.flip(dims=[2,3]) = [::-1, ::-1]
            #    - Model attends to these flipped patches
            # 
            # 2. We want to visualize on CAMERA SPACE:
            #    - For LIBERO render() (used in MP4 videos):
            #      * Agentview: ALSO flipped [::-1, ::-1] 
            #      * Wrist: NOT flipped (raw camera)
            # 
            # 3. Transformation logic:
            #    - Agentview: model_space (flipped) → camera_space (also flipped) 
            #                 → NO transformation needed (both flipped)
            #    - Wrist: model_space (flipped) → camera_space (NOT flipped)
            #                 → NEED transformation (undo LiberoProcessor flip)
            #
            # Therefore: Apply inverse transform for all cameras EXCEPT agentview
            model_space_to_camera_space = (args.camera_index >= 1)
            
            # Visualize and save
            visualize_attention_heatmap(
                heatmap,
                save_path=save_path,
                img_size=args.img_size,
                colormap=args.colormap,
                title=title,
                model_space_to_camera_space=model_space_to_camera_space,
            )
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"  Total files: {len(args.layers) * len(args.act_vis_dim)}")


if __name__ == "__main__":
    main()
