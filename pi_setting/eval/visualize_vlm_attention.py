#!/usr/bin/env python
"""
VLM Attention Visualization for Pi0.5

This script visualizes task instruction → image attention from saved VLM attention maps.
It shows which image regions the model attends to when processing the task command.

Key Features:
- Extract task→image attention from VLM attention maps
- Aggregate across task tokens (or select specific tokens)
- Create heatmaps overlaid on video frames
- Handle coordinate transformations (LiberoProcessor flips)
- Support both agentview and wrist cameras

Usage:
    python visualize_vlm_attention.py \\
        --attention_file eval_logs/quick_test/vlm_attention/libero_object_0/episode_00000_vlm_attention.pt \\
        --video_file eval_logs/quick_test/videos/libero_object_0/eval_episode_00000.mp4 \\
        --output_dir eval_logs/quick_test/vlm_attention/libero_object_0/viz_episode_00000 \\
        --rollout_steps 0 10 20 30 \\
        --layer 17 \\
        --task_text "pick up the alphabet soup and place it in the basket"
"""

import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.ndimage import zoom

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from token_boundary_helper import TokenBoundaryHelper

# For getting LIBERO task instructions
try:
    from libero.libero import benchmark
    LIBERO_TASK_SUITE_TO_CLASS = benchmark.get_benchmark_dict()
except ImportError:
    LIBERO_TASK_SUITE_TO_CLASS = None


def get_libero_task_instruction(task_name: str, task_id: int) -> str:
    """
    Get the actual task instruction from LIBERO task suite.
    
    Args:
        task_name: Task suite name (e.g., 'libero_object', 'libero_spatial')
        task_id: Task ID (0-9)
    
    Returns:
        Task instruction string (e.g., 'Pick the alphabet soup and place it in the basket')
    """
    if LIBERO_TASK_SUITE_TO_CLASS is None:
        raise ImportError("Cannot import LIBERO. Please install lerobot[libero]")
    
    # Get task suite class
    task_suite_class = LIBERO_TASK_SUITE_TO_CLASS.get(task_name)
    if task_suite_class is None:
        raise ValueError(f"Unknown task suite: {task_name}. Available: {list(LIBERO_TASK_SUITE_TO_CLASS.keys())}")
    
    # Create task suite and get task
    task_suite = task_suite_class()
    if task_id >= len(task_suite.tasks):
        raise ValueError(f"Task ID {task_id} out of range for {task_name} (max: {len(task_suite.tasks) - 1})")
    
    task = task_suite.tasks[task_id]
    return task.language


def load_vlm_attention_data(attention_path: Path, rollout_step: int, layer: int = 17):
    """
    Load VLM attention weights from .pt file for a specific rollout step.
    
    Args:
        attention_path: Path to episode_XXXXX_vlm_attention.pt file
        rollout_step: Which rollout step to load (0-N)
        layer: Which transformer layer to use (default: 17, last layer)
    
    Returns:
        Tuple of (attention_weights, prefix_len):
        - attention_weights: tensor of shape [1, 8, prefix_len, prefix_len]
        - prefix_len: int (typically 968 = 768 image + 200 language)
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
    
    # Extract VLM attention
    vlm_attention = rollout_data['vlm_attention']
    prefix_len = vlm_attention['prefix_len']
    attention_weights = vlm_attention['attention_weights']
    
    if layer not in attention_weights:
        available_layers = list(attention_weights.keys())
        raise ValueError(f"Layer {layer} not found. Available: {available_layers}")
    
    return attention_weights[layer], prefix_len


def extract_task_to_img_attention(
    attention_weights: torch.Tensor,
    task_token_range: Tuple[int, int],
    camera_index: int,
    num_img_patches: int = 256,
    num_cameras: int = 3,
    head_aggregation: str = "mean",
    token_aggregation: str = "mean",
    specific_token_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Extract attention from task tokens to image patches of a specific camera.
    
    Args:
        attention_weights: [B, 8, prefix_len, prefix_len] or [8, prefix_len, prefix_len]
        task_token_range: (start, end) indices of task tokens in sequence
        camera_index: Which camera (0=agentview, 1=wrist, 2=empty)
        num_img_patches: Patches per camera (256 for 16x16 grid)
        num_cameras: Total cameras (3 for LIBERO)
        head_aggregation: "mean", "max", or "sum" across attention heads
        token_aggregation: "mean", "max", or "sum" across task tokens
    
    Returns:
        2D attention map of shape [16, 16]
    """
    # Handle batch dimension if present
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # [8, prefix_len, prefix_len]
    
    # Calculate image patch range for this camera
    img_start = camera_index * num_img_patches
    img_end = img_start + num_img_patches
    
    # Extract task tokens → image patches attention
    # [8, prefix_len, prefix_len] → [8, task_tokens, img_patches]
    task_start, task_end = task_token_range
    task_to_img = attention_weights[:, task_start:task_end, img_start:img_end]
    
    # If specific token requested, use only that token
    if specific_token_idx is not None:
        # Select specific token: [8, task_tokens, img_patches] → [8, img_patches]
        relative_idx = specific_token_idx - task_start
        if relative_idx < 0 or relative_idx >= task_to_img.shape[1]:
            raise ValueError(f"Token index {specific_token_idx} out of task token range [{task_start}, {task_end})")
        task_to_img = task_to_img[:, relative_idx, :]  # [8, img_patches]
    else:
        # Aggregate across task tokens
        # [8, task_tokens, img_patches] → [8, img_patches]
        if token_aggregation == "mean":
            task_to_img = task_to_img.mean(dim=1)
        elif token_aggregation == "max":
            task_to_img = task_to_img.max(dim=1)[0]
        elif token_aggregation == "min":
            task_to_img = task_to_img.min(dim=1)[0]
        elif token_aggregation == "sum":
            task_to_img = task_to_img.sum(dim=1)
        else:
            raise ValueError(f"Unknown token_aggregation: {token_aggregation}")
    
    # Aggregate across attention heads
    # [8, img_patches] → [img_patches]
    if head_aggregation == "mean":
        task_to_img = task_to_img.mean(dim=0)
    elif head_aggregation == "max":
        task_to_img = task_to_img.max(dim=0)[0]
    elif head_aggregation == "min":
        task_to_img = task_to_img.min(dim=0)[0]
    elif head_aggregation == "sum":
        task_to_img = task_to_img.sum(dim=0)
    else:
        raise ValueError(f"Unknown head_aggregation: {head_aggregation}")
    
    # Reshape to 2D grid
    grid_size = int(np.sqrt(num_img_patches))
    task_to_img = task_to_img.reshape(grid_size, grid_size)  # [16, 16]
    
    return task_to_img.to(torch.float32).cpu().numpy()


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


def visualize_vlm_attention_for_timestep(
    attention_path: Path,
    video_path: Path,
    output_dir: Path,
    rollout_step: int,
    task_text: str,
    layer: int = 17,
    head_aggregation: str = "mean",
    token_aggregation: str = "mean",
    alpha: float = 0.5,
    colormap: str = "hot",
    specific_token_idx: int = None,
):
    """
    Create VLM attention visualizations for a single timestep.
    
    Args:
        attention_path: Path to episode_XXXXX_vlm_attention.pt
        video_path: Path to eval_episode_XXXXX.mp4
        output_dir: Where to save visualization images
        rollout_step: Which timestep to visualize
        task_text: Task instruction text (to identify token boundaries)
        layer: Which transformer layer (default: 17)
        head_aggregation: How to aggregate across heads
        token_aggregation: How to aggregate across task tokens
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VLM attention
    print(f"  Loading VLM attention for rollout step {rollout_step}...")
    attention_weights, prefix_len = load_vlm_attention_data(attention_path, rollout_step, layer)
    
    # Find task token boundaries
    print(f"  Finding task token boundaries...")
    # Reconstruct full text (this is a simplification - in practice, get from processor)
    full_text = f"Task: {task_text}, State: " + " ".join(["128"] * 32) + ";\nAction: "
    
    helper = TokenBoundaryHelper()
    boundaries = helper.find_token_boundaries(full_text, num_img_tokens=768)
    task_token_range = boundaries['task']
    
    print(f"    Task tokens: [{task_token_range[0]}, {task_token_range[1]}) ({task_token_range[1] - task_token_range[0]} tokens)")
    
    # Load video frame
    print(f"  Loading video frame {rollout_step}...")
    frame = load_video_frame(video_path, rollout_step)
    agentview, wrist = split_concatenated_frame(frame)
    img_size = agentview.shape[0]
    
    # Process each camera
    cameras = {
        'agentview': (0, agentview, False),  # camera_index=0, frame, no flip needed
        'wrist': (1, wrist, True),           # camera_index=1, frame, need to undo processor flip
    }
    
    for camera_name, (camera_idx, camera_frame, apply_flip) in cameras.items():
        print(f"  Processing {camera_name}...")
        
        # Extract task→image attention
        attention_map = extract_task_to_img_attention(
            attention_weights,
            task_token_range,
            camera_idx,
            num_img_patches=256,
            num_cameras=3,
            head_aggregation=head_aggregation,
            token_aggregation=token_aggregation,
            specific_token_idx=specific_token_idx,
        )
        
        # COORDINATE TRANSFORMATION:
        # - Agentview: MODEL SPACE (flipped by LiberoProcessor) → VIDEO SPACE (also flipped in render())
        #              → No transformation needed (both flipped)
        # - Wrist: MODEL SPACE (flipped by LiberoProcessor) → VIDEO SPACE (NOT flipped in render())
        #          → Need to undo LiberoProcessor flip to align with video
        if apply_flip:
            attention_map = np.flip(attention_map, axis=(0, 1))  # Undo LiberoProcessor flip
        
        # Create heatmap
        heatmap = create_attention_heatmap(attention_map, img_size, colormap)
        
        # Create side-by-side visualization (overlay | pure heatmap)
        combined = create_sidebyside_with_overlay(
            camera_frame,
            heatmap,
            alpha=alpha,
            title=f"{camera_name.capitalize()} | Timestep {rollout_step}"
        )
        
        # Save PNG
        if specific_token_idx is not None:
            output_path = output_dir / f"timestep_{rollout_step:03d}_{camera_name}_token{specific_token_idx}.png"
        else:
            output_path = output_dir / f"timestep_{rollout_step:03d}_{camera_name}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        print(f"    Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VLM attention (task→image) from saved attention maps"
    )
    parser.add_argument(
        "--attention_file",
        type=Path,
        required=True,
        help="Path to episode_XXXXX_vlm_attention.pt file"
    )
    parser.add_argument(
        "--video_file",
        type=Path,
        required=True,
        help="Path to eval_episode_XXXXX.mp4 file"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save visualization images"
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        nargs='+',
        default=[0, 10, 20, 30],
        help="Which rollout steps to visualize (default: 0 10 20 30)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=17,
        help="Which transformer layer to visualize (default: 17 = last layer)"
    )
    parser.add_argument(
        "--task_text",
        type=str,
        default=None,
        help="Task instruction text (e.g., 'Pick the alphabet soup and place it in the basket'). If not provided, will auto-detect from task_name and task_id."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="LIBERO task suite name (e.g., 'libero_object'). Used with --task_id to auto-detect task text."
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="LIBERO task ID (0-9). Used with --task_name to auto-detect task text."
    )
    parser.add_argument(
        "--head_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "sum", "min"],
        help="How to aggregate across attention heads (default: mean)"
    )
    parser.add_argument(
        "--token_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "sum", "min"],
        help="How to aggregate across task tokens (default: mean)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay transparency (0=transparent, 1=opaque, default: 0.5)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="hot",
        help="Matplotlib colormap name (default: hot)"
    )
    parser.add_argument(
        "--specific_token_idx",
        type=int,
        default=None,
        help="Visualize attention from a specific token index (e.g., 773 for 'alphabet'). When set, token_aggregation is ignored."
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not args.attention_file.exists():
        raise FileNotFoundError(f"Attention file not found: {args.attention_file}")
    if not args.video_file.exists():
        raise FileNotFoundError(f"Video file not found: {args.video_file}")
    
    # Auto-detect task text if not provided
    if args.task_text is None:
        if args.task_name is None or args.task_id is None:
            raise ValueError("Must provide either --task_text OR both --task_name and --task_id")
        args.task_text = get_libero_task_instruction(args.task_name, args.task_id)
        print(f"Auto-detected task text: \"{args.task_text}\"")
    
    print("=" * 80)
    print("VLM Attention Visualization")
    print("=" * 80)
    print(f"Attention file: {args.attention_file}")
    print(f"Video file: {args.video_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Layer: {args.layer}")
    print(f"Task text: {args.task_text}")
    print(f"Specific token: {args.specific_token_idx if args.specific_token_idx else 'None (aggregate all)'}")
    print(f"Aggregation: heads={args.head_aggregation}, tokens={args.token_aggregation if args.specific_token_idx is None else 'N/A'}")
    print("=" * 80)
    
    # Process each rollout step
    for rollout_step in args.rollout_steps:
        print(f"\nProcessing rollout step {rollout_step}...")
        try:
            visualize_vlm_attention_for_timestep(
                attention_path=args.attention_file,
                video_path=args.video_file,
                output_dir=args.output_dir,
                rollout_step=rollout_step,
                task_text=args.task_text,
                layer=args.layer,
                head_aggregation=args.head_aggregation,
                token_aggregation=args.token_aggregation,
                alpha=args.alpha,
                colormap=args.colormap,
                specific_token_idx=args.specific_token_idx,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print(f"  Output saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
