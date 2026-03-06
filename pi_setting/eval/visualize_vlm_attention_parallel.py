#!/usr/bin/env python
"""
OPTIMIZED Parallel VLM Attention Visualization (v2)

Key optimization: Load each 7GB attention file ONCE, then parallelize inner loop
(layers/tokens/heads) using shared memory multiprocessing.

Reduces I/O from ~4.5TB to ~70GB for typical runs while utilizing all CPU cores.

Architecture:
- Outer loop (sequential): Iterate episodes, load .pt file once
- Inner loop (parallel): Process all layer/token/head combos with shared memory
"""

import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import sys
import time
import logging
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import contextlib
import os
import traceback

# Suppress verbose LIBERO logging
logging.getLogger('libero').setLevel(logging.WARNING)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from token_boundary_helper import TokenBoundaryHelper

# For getting LIBERO task instructions
try:
    from libero.libero import benchmark
    LIBERO_TASK_SUITE_TO_CLASS = benchmark.get_benchmark_dict()
except ImportError:
    LIBERO_TASK_SUITE_TO_CLASS = None


def get_libero_task_instruction(task_name: str, task_id: int) -> str:
    """Get the actual task instruction from LIBERO task suite."""
    if LIBERO_TASK_SUITE_TO_CLASS is None:
        raise ImportError("Cannot import LIBERO. Please install lerobot[libero]")
    
    task_suite_class = LIBERO_TASK_SUITE_TO_CLASS.get(task_name)
    if task_suite_class is None:
        raise ValueError(f"Unknown task suite: {task_name}")
    
    # Suppress stdout during task suite creation to avoid repetitive "[info]" messages
    import os
    import contextlib
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            task_suite = task_suite_class()
    
    if task_id >= len(task_suite.tasks):
        raise ValueError(f"Task ID {task_id} out of range")
    
    return task_suite.tasks[task_id].language


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
        raise ValueError(f"Failed to load frame {frame_idx} from {video_path}")
    
    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def split_concatenated_frame(frame: np.ndarray) -> tuple:
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


def load_attention_file_once(attention_path: Path, attention_type: str) -> Dict[str, Any]:
    """
    Load entire attention file into memory ONCE.
    Returns dict with all rollout steps ready for parallel processing.
    """
    print(f"  Loading attention file ({attention_path.stat().st_size / 1e9:.1f} GB)...", flush=True)
    
    data = torch.load(attention_path, map_location='cpu')
    
    # Structure: data['rollout_steps'] is a list of dicts, each with 'rollout_step' and 'vlm_attention'
    if 'rollout_steps' not in data:
        raise KeyError(f"Missing 'rollout_steps' in attention file. Available keys: {list(data.keys())}")
    
    rollout_steps = data['rollout_steps']
    
    # Build a dict mapping rollout_step -> attention_weights for fast lookup
    attention_by_step = {}
    task_text = None
    
    for step_data in rollout_steps:
        rollout_step = step_data['rollout_step']
        
        # Extract VLM attention (key depends on attention_type)
        if attention_type == "task":
            if 'vlm_attention' not in step_data:
                raise KeyError(f"Missing 'vlm_attention' in rollout step {rollout_step}")
            vlm_attention = step_data['vlm_attention']
        elif attention_type == "general":
            if 'general_vlm_attention' not in step_data:
                raise KeyError(f"Missing 'general_vlm_attention' in rollout step {rollout_step}")
            vlm_attention = step_data['general_vlm_attention']
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        # Extract attention weights: dict mapping layer_idx -> tensor[1, heads, tokens, tokens]
        attention_weights = vlm_attention['attention_weights']
        
        # Share memory for each layer's tensor for efficient multiprocessing
        if isinstance(attention_weights, dict):
            for layer_idx in attention_weights:
                attention_weights[layer_idx].share_memory_()
        else:
            attention_weights.share_memory_()
        
        attention_by_step[rollout_step] = attention_weights
        
        # Capture task_text if available (may be None for task-specific)
        if task_text is None:
            task_text = vlm_attention.get('task_text', None)
    
    return {
        "attention": attention_by_step,
        "task_text": task_text,
    }


def process_one_visualization(args_tuple):
    """
    Worker function: Process one layer/token/head/timestep combination.
    Takes pre-loaded attention data from shared memory.
    
    Returns: (success: bool, identifier: str, error_msg: Optional[str])
    """
    (shared_attention, shared_task_text, video_path, output_dir,
     task_name, task_id, episode_id, rollout_step, layer,
     token_idx, head_idx, head_agg, alpha, colormap) = args_tuple
    
    identifier = f"Task{task_id}_Ep{episode_id}_L{layer}_Step{rollout_step}"
    if token_idx is not None:
        identifier += f"_T{token_idx}"
    if head_idx is not None:
        identifier += f"_H{head_idx}"
    
    try:
        # Get attention for this rollout step
        if rollout_step not in shared_attention:
            return (False, identifier, f"Rollout step {rollout_step} not in attention data")
        
        # attention_weights is a dict: {layer_idx: tensor[1, heads, tokens, tokens]}
        attention_weights = shared_attention[rollout_step]
        
        # Validate layer
        if layer not in attention_weights:
            available_layers = list(attention_weights.keys())
            return (False, identifier, f"Layer {layer} not found. Available: {available_layers}")
        
        # Extract specific layer: [1, heads, tokens, tokens] or [heads, tokens, tokens]
        layer_attn = attention_weights[layer]
        
        # Remove batch dimension if present
        if layer_attn.dim() == 4:
            layer_attn = layer_attn[0]  # [heads, tokens, tokens]
        
        num_heads, num_tokens, _ = layer_attn.shape
        
        # Get task token boundaries
        task_text = get_libero_task_instruction(task_name.replace("_variants", ""), task_id)
        helper = TokenBoundaryHelper()
        
        if shared_task_text is None:
            full_text = f"Task: {task_text}, State: " + " ".join(["128"] * 7) + " 1.0"
        else:
            full_text = shared_task_text
        
        boundaries = helper.find_token_boundaries(full_text, num_img_tokens=768)
        task_start, task_end = boundaries['task']
        
        # Validate token index
        if token_idx is not None:
            if token_idx < task_start or token_idx >= task_end:
                return (False, identifier, f"Token {token_idx} outside task range [{task_start}, {task_end})")
        
        # Validate head index
        if head_idx is not None:
            if head_idx >= num_heads:
                return (False, identifier, f"Head {head_idx} out of range (max: {num_heads-1})")
        
        # Load video frame and split into cameras
        frame = load_video_frame(video_path, rollout_step)
        agentview, wrist = split_concatenated_frame(frame)
        
        # Process both cameras
        cameras = {
            "agentview": (0, agentview),
            "wrist": (1, wrist)
        }
        
        num_img_patches = 256  # 16x16 grid
        num_cameras = 3
        
        for cam_name, (cam_index, cam_frame) in cameras.items():
            # Calculate image patch range for this camera
            img_start = cam_index * num_img_patches
            img_end = img_start + num_img_patches
            
            # Extract task tokens → image patches attention
            # [heads, tokens, tokens] → [heads, task_tokens, img_patches]
            task_to_img = layer_attn[:, task_start:task_end, img_start:img_end]
            
            # Handle specific token or aggregate
            if token_idx is not None:
                relative_idx = token_idx - task_start
                task_to_img = task_to_img[:, relative_idx, :]  # [heads, img_patches]
            else:
                # Aggregate across task tokens
                if head_agg == "mean":
                    task_to_img = task_to_img.mean(dim=1)
                elif head_agg == "max":
                    task_to_img = task_to_img.max(dim=1)[0]
                elif head_agg == "sum":
                    task_to_img = task_to_img.sum(dim=1)
                else:  # min
                    task_to_img = task_to_img.min(dim=1)[0]
            
            # Handle specific head or aggregate
            if head_idx is not None:
                task_to_img = task_to_img[head_idx]  # [img_patches]
            else:
                if head_agg == "mean":
                    task_to_img = task_to_img.mean(dim=0)
                elif head_agg == "max":
                    task_to_img = task_to_img.max(dim=0)[0]
                elif head_agg == "sum":
                    task_to_img = task_to_img.sum(dim=0)
                else:  # min
                    task_to_img = task_to_img.min(dim=0)[0]
            
            # Reshape to 2D grid [16, 16]
            grid_size = int(np.sqrt(num_img_patches))
            attention_map = task_to_img.reshape(grid_size, grid_size).float().cpu().numpy()
            
            # Normalize to [0, 1]
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            
            # Create RGBA heatmap using matplotlib colormap
            img_size = cam_frame.shape[0]
            heatmap = create_attention_heatmap(attention_map, img_size, colormap)
            
            # Create side-by-side visualization with title
            title = f"{cam_name.capitalize()} | Timestep {rollout_step}"
            combined = create_sidebyside_with_overlay(cam_frame, heatmap, alpha, title)
            
            # Save
            filename = f"timestep_{rollout_step:03d}_{cam_name}_layer{layer}"
            if token_idx is not None:
                filename += f"_token{token_idx}"
            if head_idx is not None:
                filename += f"_head{head_idx}"
            filename += ".png"
            
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        return (True, identifier, None)
    
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (False, identifier, error_msg)


def process_one_episode(
    eval_folder: Path,
    task_name: str,
    task_id: int,
    episode_id: int,
    layers: List[int],
    tokens: List[int],
    heads: List[int],
    timesteps: List[int],
    attention_type: str,
    head_agg: str,
    alpha: float,
    colormap: str,
    num_workers: int,
    episode_num: int = None,
    total_episodes: int = None,
):
    """
    Process one episode: Load attention file once, parallelize all combos.
    
    Returns: (success_count, fail_count, errors)
    """
    episode_name = f"{task_name}_{task_id}/episode_{episode_id:05d}"
    
    # Show episode progress in outer loop
    if episode_num is not None and total_episodes is not None:
        print(f"\n{'='*80}")
        print(f"Episode {episode_num}/{total_episodes}: {episode_name}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"Processing: {episode_name}")
        print(f"{'='*80}")
    
    # Setup paths
    if attention_type == "task":
        attention_folder = "vlm_attention"
        attention_filename = f"episode_{episode_id:05d}_vlm_attention.pt"
    else:
        attention_folder = "general_vlm_attention"
        attention_filename = f"episode_{episode_id:05d}_general_vlm_attention.pt"
    
    task_name_id = f"{task_name}_{task_id}"
    attention_path = eval_folder / attention_folder / task_name_id / attention_filename
    video_path = eval_folder / "videos" / task_name_id / f"eval_episode_{episode_id:05d}.mp4"
    output_dir = eval_folder / attention_folder / task_name_id / f"viz_episode_{episode_id:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load attention file ONCE
    start_time = time.time()
    try:
        shared_data = load_attention_file_once(attention_path, attention_type)
        load_time = time.time() - start_time
        print(f"  ✓ Loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Failed to load attention file: {e}")
        return (0, 1, [f"Failed to load {attention_path}: {str(e)}"])
    
    # Generate all combinations
    token_list = tokens if tokens else [None]
    head_list = heads if heads else [None]
    
    combinations = []
    for layer in layers:
        for rollout_step in timesteps:
            for token_idx in token_list:
                for head_idx in head_list:
                    combinations.append((
                        shared_data["attention"],
                        shared_data["task_text"],
                        video_path,
                        output_dir,
                        task_name,
                        task_id,
                        episode_id,
                        rollout_step,
                        layer,
                        token_idx,
                        head_idx,
                        head_agg,
                        alpha,
                        colormap,
                    ))
    
    total_combos = len(combinations)
    print(f"  Processing {total_combos} visualizations with {num_workers} workers...")
    print(f"  (Errors will be shown immediately)\n")
    
    # Process in parallel
    success_count = 0
    fail_count = 0
    errors = []
    
    with mp.Pool(processes=num_workers) as pool:
        for i, (success, identifier, error_msg) in enumerate(
            pool.imap_unordered(process_one_visualization, combinations)
        ):
            if success:
                success_count += 1
                if (i + 1) % max(1, total_combos // 10) == 0:  # Progress every 10%
                    print(f"    [{i+1}/{total_combos}] {success_count} ✓, {fail_count} ✗")
            else:
                fail_count += 1
                print(f"\n  ✗ {identifier}")
                print(f"    ERROR: {error_msg}\n")
                errors.append(f"{identifier}: {error_msg}")
    
    process_time = time.time() - start_time
    print(f"\n  Summary: {success_count} ✓, {fail_count} ✗ ({process_time:.1f}s total)")
    
    return (success_count, fail_count, errors)


def main():
    parser = argparse.ArgumentParser(
        description="Optimized parallel VLM attention visualization (v2)"
    )
    parser.add_argument("--eval_folder", type=Path, required=True)
    parser.add_argument("--task_name", type=str, default="libero_object")
    parser.add_argument("--max_task_id", type=int, default=9)
    parser.add_argument("--max_episode_id", type=int, default=1)
    parser.add_argument("--attention_type", type=str, default="task", choices=["task", "general"])
    parser.add_argument("--layers", type=int, nargs='+', default=[17])
    parser.add_argument("--tokens", type=int, nargs='*', default=[])
    parser.add_argument("--heads", type=int, nargs='*', default=[])
    parser.add_argument("--timesteps", type=int, nargs='+', 
                       default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    parser.add_argument("--head_aggregation", type=str, default="mean", 
                       choices=["mean", "max", "sum", "min"])
    parser.add_argument("--token_aggregation", type=str, default="mean",
                       choices=["mean", "max", "sum", "min"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--colormap", type=str, default="hot")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Workers for INNER loop (default: CPU count)")
    
    args = parser.parse_args()
    
    # Auto-detect CPU count
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    
    # Calculate stats
    token_count = len(args.tokens) if args.tokens else 1
    head_count = len(args.heads) if args.heads else 1
    combos_per_episode = len(args.layers) * token_count * head_count * len(args.timesteps)
    total_episodes = (args.max_task_id + 1) * (args.max_episode_id + 1)
    
    print("=" * 80)
    print(f"OPTIMIZED Parallel VLM Attention Visualization v2")
    print("=" * 80)
    print(f"Strategy: Load each .pt file ONCE, parallelize inner loop")
    print(f"Attention type: {args.attention_type}")
    print(f"Eval folder: {args.eval_folder}")
    print(f"Tasks: 0-{args.max_task_id}, Episodes: 0-{args.max_episode_id}")
    print(f"Layers: {args.layers}")
    print(f"Tokens: {args.tokens if args.tokens else 'Aggregate all'}")
    print(f"Heads: {args.heads if args.heads else 'Aggregate all'}")
    print(f"Timesteps: {len(args.timesteps)} steps")
    print(f"Visualizations per episode: {combos_per_episode}")
    print(f"Total episodes: {total_episodes}")
    print(f"Inner loop workers: {args.num_workers} (utilizing all CPU cores!)")
    print("=" * 80)
    
    # Process episodes sequentially (could add 2-3 parallel episodes if needed)
    total_success = 0
    total_fail = 0
    all_errors = []
    episode_counter = 0
    
    for task_id in range(args.max_task_id + 1):
        for episode_id in range(args.max_episode_id + 1):
            episode_counter += 1
            success, fail, errors = process_one_episode(
                eval_folder=args.eval_folder,
                task_name=args.task_name,
                task_id=task_id,
                episode_id=episode_id,
                layers=args.layers,
                tokens=args.tokens,
                heads=args.heads,
                timesteps=args.timesteps,
                attention_type=args.attention_type,
                head_agg=args.head_aggregation,
                alpha=args.alpha,
                colormap=args.colormap,
                num_workers=args.num_workers,
                episode_num=episode_counter,
                total_episodes=total_episodes,
            )
            
            total_success += success
            total_fail += fail
            all_errors.extend(errors)
    
    # Final summary
    total_viz = total_success + total_fail
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {total_success} / {total_viz}")
    print(f"✗ Failed: {total_fail} / {total_viz}")
    print(f"Episodes processed: {total_episodes}")
    
    if total_fail > 0:
        print(f"\n⚠️  {total_fail} visualizations failed. See errors above.")
    
    print("=" * 80)
    
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    # Required for torch multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
