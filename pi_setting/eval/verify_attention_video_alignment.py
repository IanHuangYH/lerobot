#!/usr/bin/env python
"""
Comprehensive verification script to check alignment between attention heatmaps and MP4 video frames.

This script reads attention .pt files and MP4 videos, then creates side-by-side visualizations
showing video frames next to their corresponding attention heatmaps for multiple timesteps.

COORDINATE SYSTEM NOTES:
- MODEL SPACE: LiberoProcessor flips ALL cameras [::-1, ::-1]
- VIDEO SPACE (MP4): Only agentview is flipped in render(), wrist is NOT flipped
- Therefore: 
  * Agentview: No transformation needed (both spaces flipped)
  * Wrist: Need to undo LiberoProcessor flip to align with video
"""

import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from scipy.ndimage import zoom


def load_attention_data(attention_path: Path, rollout_step: int, denoising_step: int = 9, layer: int = 17):
    """
    Load attention weights from .pt file for a specific rollout step.
    
    Args:
        attention_path: Path to episode_XXXXX_attention.pt file
        rollout_step: Which rollout step to load (0-13 for 14 total steps)
        denoising_step: Which denoising step to use (default: 9, final step)
        layer: Which transformer layer to use (default: 17, last layer)
    
    Returns:
        Attention weights tensor of shape [1, 8, 50, 1018]
    """
    data = torch.load(attention_path, map_location='cpu')
    rollout_steps = data['rollout_steps']
    
    # Find the requested rollout step
    rollout_data = None
    for step_data in rollout_steps:
        if step_data['rollout_step'] == rollout_step:
            rollout_data = step_data
            break
    
    if rollout_data is None:
        available_steps = [s['rollout_step'] for s in rollout_steps]
        raise ValueError(f"Rollout step {rollout_step} not found. Available: {available_steps}")
    
    # Extract attention for specified denoising step and layer
    attention_maps = rollout_data['attention_maps']
    if denoising_step not in attention_maps:
        raise ValueError(f"Denoising step {denoising_step} not found")
    
    denoising_data = attention_maps[denoising_step]
    layer_attentions = denoising_data['attention_weights']
    
    if layer not in layer_attentions:
        raise ValueError(f"Layer {layer} not found")
    
    return layer_attentions[layer]


def extract_action_to_img_attention(
    attention_weights: torch.Tensor,
    action_idx: int,
    camera_index: int,
    num_img_patches: int = 256,
    num_cameras: int = 3,
    head_aggregation: str = "mean",
) -> np.ndarray:
    """
    Extract attention from a specific action token to image patches of a specific camera.
    
    Args:
        attention_weights: [1, 8, 50, 1018] or [8, 50, 1018]
        action_idx: Which action timestep (0-48)
        camera_index: Which camera (0=agentview, 1=wrist, 2=empty)
        num_img_patches: Patches per camera (256 for 16x16 grid)
        num_cameras: Total cameras (3 for LIBERO)
        head_aggregation: "mean", "max", or "sum"
    
    Returns:
        2D attention map of shape [16, 16]
    """
    # Remove batch dim if present
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.squeeze(0)  # [8, 50, 1018]
    
    # Calculate patch range for this camera
    start_idx = camera_index * num_img_patches
    end_idx = start_idx + num_img_patches
    
    # Extract attention to this camera's patches
    # [8, 50, 1018] -> [8, 50, 256]
    action_to_img = attention_weights[:, :, start_idx:end_idx]
    
    # Map action index to suffix token index
    # Suffix tokens: [0=timestep, 1=action_0, 2=action_1, ..., 49=action_48]
    suffix_token_idx = action_idx + 1
    
    # Select this action token: [8, 50, 256] -> [8, 256]
    action_to_img = action_to_img[:, suffix_token_idx, :]
    
    # Aggregate across heads
    if head_aggregation == "mean":
        action_to_img = action_to_img.mean(dim=0)  # [256]
    elif head_aggregation == "max":
        action_to_img = action_to_img.max(dim=0)[0]
    elif head_aggregation == "sum":
        action_to_img = action_to_img.sum(dim=0)
    
    # Reshape to 2D grid
    grid_size = int(np.sqrt(num_img_patches))
    action_to_img = action_to_img.reshape(grid_size, grid_size)  # [16, 16]
    
    return action_to_img.to(torch.float32).cpu().numpy()


def load_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """
    Load a specific frame from MP4 video.
    
    Args:
        video_path: Path to .mp4 file
        frame_idx: Frame index to load
    
    Returns:
        RGB image array of shape [H, W, 3]
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
    
    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def split_concatenated_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split concatenated frame into agentview (left) and wrist (right).
    
    Args:
        frame: Concatenated frame of shape [H, W, 3] where W = 2*H
    
    Returns:
        Tuple of (agentview, wrist), each of shape [H, H, 3]
    """
    H, W, C = frame.shape
    assert W == 2 * H, f"Expected width={2*H} for concatenated cameras, got {W}"
    
    mid = W // 2
    agentview = frame[:, :mid, :]
    wrist = frame[:, mid:, :]
    
    return agentview, wrist


def create_attention_heatmap(
    attention_map: np.ndarray,
    img_size: int,
    colormap: str = "hot",
) -> np.ndarray:
    """
    Create attention heatmap image with colormap (includes alpha channel).
    
    Args:
        attention_map: 2D array [16, 16]
        img_size: Target size for upsampling
        colormap: Matplotlib colormap name
    
    Returns:
        RGBA image array [img_size, img_size, 4] with alpha channel
    """
    # Upsample to target size
    grid_size = attention_map.shape[0]
    zoom_factor = img_size / grid_size
    upsampled = zoom(attention_map, zoom_factor, order=1)  # Bilinear interpolation
    
    # Normalize to [0, 1]
    if attention_map.max() > 0:
        upsampled = upsampled / attention_map.max()
    
    # Apply colormap (keep alpha channel)
    cmap = plt.get_cmap(colormap)
    colored = cmap(upsampled)  # RGBA [img_size, img_size, 4]
    
    # Convert to uint8
    colored = (colored * 255).astype(np.uint8)
    
    return colored


def overlay_heatmap_on_image(
    video_frame: np.ndarray,
    attention_heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay attention heatmap on top of video frame with transparency.
    
    Args:
        video_frame: RGB image [H, W, 3]
        attention_heatmap: RGBA heatmap [H, W, 4]
        alpha: Transparency level for heatmap (0.0 = invisible, 1.0 = opaque)
    
    Returns:
        Combined RGB image [H, W, 3]
    """
    # Ensure same size
    if video_frame.shape[:2] != attention_heatmap.shape[:2]:
        attention_heatmap = cv2.resize(
            attention_heatmap, 
            (video_frame.shape[1], video_frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Convert video frame to float for blending
    video_float = video_frame.astype(np.float32)
    
    # Extract RGB and alpha from heatmap
    heatmap_rgb = attention_heatmap[:, :, :3].astype(np.float32)
    heatmap_alpha = attention_heatmap[:, :, 3].astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Apply additional alpha scaling
    heatmap_alpha = heatmap_alpha * alpha
    
    # Blend: result = video * (1 - alpha) + heatmap * alpha
    blended = video_float * (1 - heatmap_alpha[:, :, np.newaxis]) + heatmap_rgb * heatmap_alpha[:, :, np.newaxis]
    
    # Convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended


def create_sidebyside_with_overlay(
    video_frame: np.ndarray,
    attention_heatmap: np.ndarray,
    alpha: float = 0.5,
    title: str = "",
) -> np.ndarray:
    """
    Create visualization with overlay on left and pure heatmap on right.
    
    Args:
        video_frame: RGB image [H, W, 3]
        attention_heatmap: RGBA heatmap [H, W, 4]
        alpha: Transparency for overlay
        title: Title text
    
    Returns:
        Combined image [H, 2*W, 3] with title bar on top
    """
    # Create overlay (left side)
    overlay = overlay_heatmap_on_image(video_frame, attention_heatmap, alpha)
    
    # Create pure heatmap (right side) - convert RGBA to RGB
    pure_heatmap = attention_heatmap[:, :, :3]
    
    # Ensure same size
    if overlay.shape != pure_heatmap.shape:
        pure_heatmap = cv2.resize(
            pure_heatmap,
            (overlay.shape[1], overlay.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Concatenate horizontally: [overlay | pure_heatmap]
    combined = np.hstack([overlay, pure_heatmap])
    
    # Add title if provided
    if title:
        # Create title bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        # Calculate appropriate font scale based on image width
        # Start with a reasonable font scale and check if text fits
        font_scale = 0.5
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        
        # Scale down font if text is too wide
        if text_size[0] > combined.shape[1] - 20:
            font_scale = font_scale * (combined.shape[1] - 20) / text_size[0]
            text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        
        # Add white bar at top
        title_height = text_size[1] + 20
        title_bar = np.ones((title_height, combined.shape[1], 3), dtype=np.uint8) * 255
        
        # Add text (centered)
        text_x = (combined.shape[1] - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        cv2.putText(title_bar, title, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Combine with image
        combined = np.vstack([title_bar, combined])
    
    return combined


def main():
    parser = argparse.ArgumentParser(description="Verify attention-video alignment")
    parser.add_argument(
        "--attention_path",
        type=str,
        required=True,
        help="Path to attention .pt file (e.g., episode_00000_attention.pt)"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to MP4 video file (e.g., eval_episode_0.mp4)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save output PNG files"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 5, 10],
        help="Rollout timesteps to visualize (e.g., 0 10 20 30)"
    )
    parser.add_argument(
        "--action_idx",
        type=int,
        default=0,
        help="Which action timestep to visualize (default: 0, first action)"
    )
    parser.add_argument(
        "--denoising_step",
        type=int,
        default=9,
        help="Which denoising step to use (default: 9)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=17,
        help="Which transformer layer to use (default: 17)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="hot",
        help="Matplotlib colormap for heatmaps"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency for attention overlay (0.0=invisible, 1.0=opaque, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    attention_path = Path(args.attention_path)
    video_path = Path(args.video_path)
    output_path = Path(args.output_path)
    
    # Validate paths
    if not attention_path.exists():
        raise FileNotFoundError(f"Attention file not found: {attention_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Attention file: {attention_path}")
    print(f"Video file: {video_path}")
    print(f"Output directory: {output_path}")
    print(f"Timesteps to visualize: {args.timesteps}")
    print(f"Action index: {args.action_idx}")
    print(f"Layer: {args.layer}, Denoising step: {args.denoising_step}")
    print()
    
    # Process each timestep
    for timestep in args.timesteps:
        print(f"Processing timestep {timestep}...")
        
        try:
            # Load attention weights
            attention_weights = load_attention_data(
                attention_path,
                rollout_step=timestep,
                denoising_step=args.denoising_step,
                layer=args.layer
            )
            
            # Load video frame
            video_frame = load_video_frame(video_path, timestep)
            
            # Split into agentview and wrist
            agentview_frame, wrist_frame = split_concatenated_frame(video_frame)
            
            # Get frame size
            img_size = agentview_frame.shape[0]
            
            # Extract attention for both cameras
            agentview_attention = extract_action_to_img_attention(
                attention_weights,
                action_idx=args.action_idx,
                camera_index=0,  # agentview
            )
            
            wrist_attention = extract_action_to_img_attention(
                attention_weights,
                action_idx=args.action_idx,
                camera_index=1,  # wrist
            )
            
            # COORDINATE TRANSFORMATION:
            # - Agentview: MODEL SPACE (flipped by LiberoProcessor) → VIDEO SPACE (also flipped in render())
            #              → No transformation needed (both flipped)
            # - Wrist: MODEL SPACE (flipped by LiberoProcessor) → VIDEO SPACE (NOT flipped in render())
            #          → Need to undo LiberoProcessor flip to align with video
            wrist_attention = np.flip(wrist_attention, axis=(0, 1))  # Undo LiberoProcessor flip
            
            # Create heatmaps
            agentview_heatmap = create_attention_heatmap(
                agentview_attention,
                img_size=img_size,
                colormap=args.colormap
            )
            
            wrist_heatmap = create_attention_heatmap(
                wrist_attention,
                img_size=img_size,
                colormap=args.colormap
            )
            
            # Create side-by-side visualizations
            agentview_combined = create_sidebyside_with_overlay(
                agentview_frame,
                agentview_heatmap,
                alpha=args.alpha,
                title=f"Agentview | Timestep {timestep} | Action {args.action_idx}"
            )
            
            wrist_combined = create_sidebyside_with_overlay(
                wrist_frame,
                wrist_heatmap,
                alpha=args.alpha,
                title=f"Wrist | Timestep {timestep} | Action {args.action_idx}"
            )
            
            # Save PNGs
            agentview_path = output_path / f"timestep_{timestep:03d}_agentview.png"
            wrist_path = output_path / f"timestep_{timestep:03d}_wrist.png"
            
            cv2.imwrite(str(agentview_path), cv2.cvtColor(agentview_combined, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_combined, cv2.COLOR_RGB2BGR))
            
            print(f"  ✓ Saved: {agentview_path.name}")
            print(f"  ✓ Saved: {wrist_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error processing timestep {timestep}: {e}")
            continue
    
    print(f"\n✓ All visualizations saved to: {output_path}")


if __name__ == "__main__":
    main()
