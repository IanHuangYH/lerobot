#!/usr/bin/env python
"""
Verification script to check alignment between uncertainty heatmaps and MP4 video frames.

This script reads uncertainty .pt files and MP4 videos, then creates 2x2 grid visualizations
showing video frames with uncertainty heatmap overlays for specified timesteps.

Output format (2x2 grid):
┌─────────────────────┬─────────────────────┐
│   Agentview         │   Wrist             │
│   + Overlay         │   + Overlay         │
│   Score: 0.145      │   Score: 0.089      │
├─────────────────────┼─────────────────────┤
│   Agentview         │   Wrist             │
│   Pure Heatmap      │   Pure Heatmap      │
└─────────────────────┴─────────────────────┘
       Overall Uncertainty: 0.117
"""

import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.ndimage import zoom


def load_uncertainty_data(uncertainty_path: Path, timestep: int) -> Dict:
    """
    Load uncertainty data from .pt file for a specific timestep.
    
    Args:
        uncertainty_path: Path to episode_XXXXX_uncertainty.pt file
        timestep: Which timestep to load (0 to num_steps-1)
    
    Returns:
        Dictionary containing uncertainty scores and spatial maps:
        {
            'overall': float,
            'agentview': float,
            'wrist': float,
            'spatial_maps': {
                'agentview': torch.Tensor (16, 16),
                'wrist': torch.Tensor (16, 16)
            }
        }
    """
    data = torch.load(uncertainty_path, map_location='cpu')
    rollout_steps = data['rollout_steps']
    
    # Find the requested timestep
    step_data = None
    for step in rollout_steps:
        if step['step'] == timestep:
            step_data = step
            break
    
    if step_data is None:
        available_steps = [s['step'] for s in rollout_steps]
        raise ValueError(f"Timestep {timestep} not found. Available: {available_steps}")
    
    return step_data['uncertainty']


def extract_spatial_map(uncertainty_data: Dict, camera: str) -> np.ndarray:
    """
    Extract spatial uncertainty map for a specific camera.
    
    Args:
        uncertainty_data: Dictionary from load_uncertainty_data()
        camera: Camera name ('agentview' or 'wrist')
    
    Returns:
        2D numpy array of shape (16, 16)
    """
    spatial_maps = uncertainty_data.get('spatial_maps', {})
    if camera not in spatial_maps:
        raise ValueError(f"Camera '{camera}' not found in spatial_maps. Available: {list(spatial_maps.keys())}")
    
    spatial_map = spatial_maps[camera]
    if isinstance(spatial_map, torch.Tensor):
        spatial_map = spatial_map.cpu().numpy()
    
    return spatial_map


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


def create_uncertainty_heatmap(
    uncertainty_map: np.ndarray,
    img_size: int,
    colormap: str = "viridis",
) -> np.ndarray:
    """
    Create uncertainty heatmap image with colormap (includes alpha channel).
    
    Args:
        uncertainty_map: 2D array [16, 16]
        img_size: Target size for upsampling
        colormap: Matplotlib colormap name ('viridis', 'plasma', 'hot', etc.)
    
    Returns:
        RGBA image array [img_size, img_size, 4] with alpha channel
    """
    # Upsample to target size
    grid_size = uncertainty_map.shape[0]
    zoom_factor = img_size / grid_size
    upsampled = zoom(uncertainty_map, zoom_factor, order=0)  # Nearest-neighbor (preserves patch boundaries)
    
    # Normalize to [0, 1]
    if uncertainty_map.max() > 0:
        upsampled = upsampled / uncertainty_map.max()
    
    # Apply colormap (keep alpha channel)
    cmap = plt.get_cmap(colormap)
    colored = cmap(upsampled)  # RGBA [img_size, img_size, 4]
    
    # Convert to uint8
    colored = (colored * 255).astype(np.uint8)
    
    return colored


def overlay_heatmap_on_image(
    video_frame: np.ndarray,
    uncertainty_heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay uncertainty heatmap on top of video frame with transparency.
    
    Args:
        video_frame: RGB image [H, W, 3]
        uncertainty_heatmap: RGBA heatmap [H, W, 4]
        alpha: Transparency level for heatmap (0.0 = invisible, 1.0 = opaque)
    
    Returns:
        Combined RGB image [H, W, 3]
    """
    # Ensure same size
    if video_frame.shape[:2] != uncertainty_heatmap.shape[:2]:
        uncertainty_heatmap = cv2.resize(
            uncertainty_heatmap, 
            (video_frame.shape[1], video_frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Convert video frame to float for blending
    video_float = video_frame.astype(np.float32)
    
    # Extract RGB and alpha from heatmap
    heatmap_rgb = uncertainty_heatmap[:, :, :3].astype(np.float32)
    heatmap_alpha = uncertainty_heatmap[:, :, 3].astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Apply additional alpha scaling
    heatmap_alpha = heatmap_alpha * alpha
    
    # Blend: result = video * (1 - alpha) + heatmap * alpha
    blended = video_float * (1 - heatmap_alpha[:, :, np.newaxis]) + heatmap_rgb * heatmap_alpha[:, :, np.newaxis]
    
    # Convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended


def add_text_to_image(
    image: np.ndarray,
    text: str,
    position: str = "top",
    font_scale: float = 0.6,
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Add text overlay to image with background.
    
    Args:
        image: RGB image [H, W, 3]
        text: Text to display
        position: 'top' or 'bottom'
        font_scale: Font size multiplier
        thickness: Font thickness
        bg_color: Background color (R, G, B)
        text_color: Text color (R, G, B)
    
    Returns:
        Image with text overlay [H, W, 3]
    """
    img_copy = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position
    H, W = image.shape[:2]
    padding = 10
    
    if position == "top":
        text_x = padding
        text_y = text_height + padding
        bg_y1 = 0
        bg_y2 = text_height + 2 * padding
    else:  # bottom
        text_x = padding
        text_y = H - padding
        bg_y1 = H - text_height - 2 * padding
        bg_y2 = H
    
    # Draw background rectangle
    cv2.rectangle(img_copy, (0, bg_y1), (W, bg_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(img_copy, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return img_copy


def create_colorbar(height: int, width: int, vmin: float, vmax: float, colormap: str = "viridis") -> np.ndarray:
    """
    Create a colorbar image showing the uncertainty scale.
    
    Args:
        height: Height of colorbar in pixels
        width: Width of colorbar in pixels (typically 60-100)
        vmin: Minimum uncertainty value
        vmax: Maximum uncertainty value
        colormap: Matplotlib colormap name
    
    Returns:
        RGB image of colorbar [height, width, 3]
    """
    # Create figure with colorbar
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.subplots_adjust(left=0.0, right=0.4, top=1.0, bottom=0.0)
    
    # Create colorbar
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add colorbar to the axis
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.set_label('Uncertainty', rotation=270, labelpad=20, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Convert to image
    fig.canvas.draw()
    # Use tobytes() instead of deprecated tostring_rgb()
    colorbar_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    colorbar_img = colorbar_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    colorbar_img = colorbar_img[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)
    
    # Resize to exact dimensions
    colorbar_img = cv2.resize(colorbar_img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return colorbar_img


def create_visualization_grid(
    agentview_frame: np.ndarray,
    wrist_frame: np.ndarray,
    agentview_map: np.ndarray,
    wrist_map: np.ndarray,
    agentview_score: float,
    wrist_score: float,
    overall_score: float,
    timestep: int,
    colormap: str = "viridis",
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Create 2x2 grid visualization with overlays, heatmaps, and colorbar.
    
    Layout:
    ┌─────────────────────┬─────────────────────┬──────┐
    │   Agentview         │   Wrist             │      │
    │   + Overlay         │   + Overlay         │ C    │
    │   Score: X.XXX      │   Score: Y.YYY      │ O    │
    ├─────────────────────┼─────────────────────┤ L    │
    │   Agentview         │   Wrist             │ O    │
    │   Pure Heatmap      │   Pure Heatmap      │ R    │
    └─────────────────────┴─────────────────────┴──────┘
           Overall: Z.ZZZ | Timestep: T
    
    Args:
        agentview_frame: RGB image [H, W, 3]
        wrist_frame: RGB image [H, W, 3]
        agentview_map: Uncertainty map [16, 16]
        wrist_map: Uncertainty map [16, 16]
        agentview_score: Agentview uncertainty score
        wrist_score: Wrist uncertainty score
        overall_score: Overall uncertainty score
        timestep: Timestep number
        colormap: Matplotlib colormap
        alpha: Overlay transparency
    
    Returns:
        Grid image with colorbar [2*H + margin, 2*W + colorbar_width + margin, 3]
    """
    H, W, _ = agentview_frame.shape
    
    # Calculate uncertainty range for colorbar (across both cameras)
    vmin = min(agentview_map.min(), wrist_map.min())
    vmax = max(agentview_map.max(), wrist_map.max())
    
    # Create heatmaps
    ag_heatmap = create_uncertainty_heatmap(agentview_map, H, colormap)
    wrist_heatmap = create_uncertainty_heatmap(wrist_map, H, colormap)
    
    # Create overlays (without text - keep images clean for alignment)
    ag_overlay = overlay_heatmap_on_image(agentview_frame, ag_heatmap, alpha)
    wrist_overlay = overlay_heatmap_on_image(wrist_frame, wrist_heatmap, alpha)
    
    # Convert pure heatmaps to RGB (remove alpha channel)
    ag_heatmap_rgb = ag_heatmap[:, :, :3]
    wrist_heatmap_rgb = wrist_heatmap[:, :, :3]
    
    # Create grid without any text overlays (perfect alignment)
    top_row = np.hstack([ag_overlay, wrist_overlay])
    bottom_row = np.hstack([ag_heatmap_rgb, wrist_heatmap_rgb])
    
    # Add label margin at the top for row titles
    label_height = 30
    label_width = 2 * W  # Full width of concatenated images
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = (0, 0, 0)  # Black text
    
    # Top label: camera names and scores
    top_label = np.ones((label_height, label_width, 3), dtype=np.uint8) * 240  # Light gray
    
    # Agentview label (left half)
    ag_text = f"Agentview ({agentview_score:.4f})"
    (text_width, text_height), _ = cv2.getTextSize(ag_text, font, font_scale, thickness)
    text_x = (W - text_width) // 2
    text_y = (label_height + text_height) // 2
    cv2.putText(top_label, ag_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Wrist label (right half)
    wrist_text = f"Wrist ({wrist_score:.4f})"
    (text_width, text_height), _ = cv2.getTextSize(wrist_text, font, font_scale, thickness)
    text_x = W + (W - text_width) // 2
    text_y = (label_height + text_height) // 2
    cv2.putText(top_label, wrist_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Middle label: overall uncertainty and timestep (same height as top label)
    middle_label = np.ones((label_height, label_width, 3), dtype=np.uint8) * 240  # Light gray
    
    # Overall info centered across full width
    overall_text = f"Overall: {overall_score:.4f} | Timestep: {timestep}"
    (text_width, text_height), _ = cv2.getTextSize(overall_text, font, font_scale, thickness)
    text_x = (label_width - text_width) // 2
    text_y = (label_height + text_height) // 2
    cv2.putText(middle_label, overall_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Stack: top label + top row + middle label + bottom row
    grid_with_labels = np.vstack([top_label, top_row, middle_label, bottom_row])
    
    # Create and add colorbar on the right
    colorbar_width = 80
    grid_height = grid_with_labels.shape[0]
    colorbar = create_colorbar(grid_height, colorbar_width, vmin, vmax, colormap)
    
    # Add white margin between grid and colorbar
    margin_width = 10
    margin = np.ones((grid_height, margin_width, 3), dtype=np.uint8) * 255
    
    # Combine grid, margin, and colorbar
    grid_with_colorbar = np.hstack([grid_with_labels, margin, colorbar])
    
    return grid_with_colorbar


def main():
    parser = argparse.ArgumentParser(description="Verify uncertainty heatmap and video alignment")
    parser.add_argument("--uncertainty_path", type=str, required=True,
                        help="Path to episode_XXXXX_uncertainty.pt file")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to corresponding .mp4 video file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for verification images")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 50, 100],
                        help="Timesteps to visualize (e.g., 0 10 20 30)")
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap (viridis, plasma, hot, coolwarm, etc.)")
    parser.add_argument("--alpha", type=float, default=0.55,
                        help="Overlay transparency (0.0=invisible, 1.0=opaque)")
    parser.add_argument("--episode_id", type=int, default=None,
                        help="Episode ID for organizing outputs")
    
    args = parser.parse_args()
    
    # Create paths
    uncertainty_path = Path(args.uncertainty_path)
    video_path = Path(args.video_path)
    output_path = Path(args.output_path)
    
    # Organize output by episode
    if args.episode_id is not None:
        output_path = output_path / f"episode_{args.episode_id:05d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading uncertainty data from: {uncertainty_path}")
    print(f"Loading video from: {video_path}")
    print(f"Output directory: {output_path}")
    print(f"Timesteps to visualize: {args.timesteps}")
    print("")
    
    # Process each timestep
    for timestep in args.timesteps:
        try:
            print(f"Processing timestep {timestep}...")
            
            # Load uncertainty data
            uncertainty_data = load_uncertainty_data(uncertainty_path, timestep)
            
            # Extract scores and spatial maps
            overall_score = uncertainty_data['overall']
            agentview_score = uncertainty_data['agentview']
            wrist_score = uncertainty_data['wrist']
            agentview_map = extract_spatial_map(uncertainty_data, 'agentview')
            wrist_map = extract_spatial_map(uncertainty_data, 'wrist')
            
            # Load video frame
            video_frame = load_video_frame(video_path, timestep)
            agentview_frame, wrist_frame = split_concatenated_frame(video_frame)
            
            # Create visualization grid
            grid = create_visualization_grid(
                agentview_frame=agentview_frame,
                wrist_frame=wrist_frame,
                agentview_map=agentview_map,
                wrist_map=wrist_map,
                agentview_score=agentview_score,
                wrist_score=wrist_score,
                overall_score=overall_score,
                timestep=timestep,
                colormap=args.colormap,
                alpha=args.alpha,
            )
            
            # Save visualization
            output_file = output_path / f"timestep_{timestep:03d}_grid.png"
            grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), grid_bgr)
            
            print(f"  ✓ Saved: {output_file}")
            print(f"    Overall: {overall_score:.4f} | Agentview: {agentview_score:.4f} | Wrist: {wrist_score:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error at timestep {timestep}: {e}")
            continue
    
    print("")
    print(f"✓ Verification complete! Outputs saved to: {output_path}")
    print(f"  Total timesteps processed: {len(args.timesteps)}")


if __name__ == "__main__":
    main()
