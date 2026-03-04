#!/usr/bin/env python
"""
Parallel VLM Attention Visualization

This script parallelizes VLM attention visualization across multiple CPU cores.
Supports both task-specific and general (baseline) VLM attention visualization.

Usage:
    # Task-specific VLM attention
    python visualize_vlm_attention_parallel.py \\
        --eval_folder uncertainty_quantification/eval_log/vlm_attention_object_all \\
        --task_name libero_object \\
        --max_task_id 9 \\
        --max_episode_id 1 \\
        --layers 15 16 17 \\
        --timesteps 0 10 20 30
    
    # General VLM attention (baseline)
    python visualize_vlm_attention_parallel.py \\
        --eval_folder uncertainty_quantification/eval_log/general_vlm_attention_object_all \\
        --task_name libero_object \\
        --max_task_id 9 \\
        --max_episode_id 1 \\
        --layers 15 16 17 \\
        --timesteps 0 10 20 30 \\
        --attention_type general
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
import subprocess
import sys


def visualize_one_combination(args_tuple):
    """
    Worker function to visualize one parameter combination.
    
    Args:
        args_tuple: (eval_folder, task_name, task_id, episode_id, rollout_steps,
                    layer, attention_type, head_agg, token_agg, 
                    alpha, colormap, specific_token_idx, specific_head_idx)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    (eval_folder, task_name, task_id, episode_id, rollout_steps,
     layer, attention_type, head_agg, token_agg, 
     alpha, colormap, specific_token_idx, specific_head_idx) = args_tuple
    
    # Build command
    cmd = [
        "python", "pi_setting/eval/visualize_vlm_attention.py",
        "--eval_folder", str(eval_folder),
        "--task_name", task_name,
        "--task_id", str(task_id),
        "--episode_id", str(episode_id),
        "--rollout_steps", *[str(s) for s in rollout_steps],
        "--layer", str(layer),
        "--attention_type", attention_type,
        "--head_aggregation", head_agg,
        "--alpha", str(alpha),
        "--colormap", colormap,
    ]
    
    # Add optional parameters
    if specific_token_idx is not None:
        cmd.extend(["--specific_token_idx", str(specific_token_idx)])
    else:
        cmd.extend(["--token_aggregation", token_agg])
    
    if specific_head_idx is not None:
        cmd.extend(["--specific_head_idx", str(specific_head_idx)])
    
    # Create identifier for logging
    identifier = f"Task{task_id}_Ep{episode_id}_L{layer}"
    if specific_token_idx is not None:
        identifier += f"_T{specific_token_idx}"
    if specific_head_idx is not None:
        identifier += f"_H{specific_head_idx}"
    
    print("run visualization for", identifier)
    try:
        # Run subprocess with suppressed output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per visualization
        )
        
        if result.returncode == 0:
            return (True, identifier)
        else:
            return (False, f"{identifier}: {result.stderr[:100]}")
    
    except subprocess.TimeoutExpired:
        return (False, f"{identifier}: Timeout")
    except Exception as e:
        return (False, f"{identifier}: {str(e)[:100]}")


def generate_all_combinations(
    eval_folder: Path,
    task_name: str,
    max_task_id: int,
    max_episode_id: int,
    layers: List[int],
    tokens: List[int],
    heads: List[int],
    timesteps: List[int],
    attention_type: str,
    head_agg: str,
    token_agg: str,
    alpha: float,
    colormap: str,
) -> List[Tuple]:
    """
    Generate all parameter combinations to visualize.
    
    Returns:
        List of argument tuples for visualize_one_combination()
    """
    combinations = []
    
    # Determine folder name based on attention_type
    if attention_type == "task":
        attention_folder = "vlm_attention"
        attention_filename = "episode_{}_vlm_attention.pt"
    else:  # general
        attention_folder = "general_vlm_attention"
        attention_filename = "episode_{}_general_vlm_attention.pt"
    
    for layer in layers:
        for task_id in range(max_task_id + 1):
            task_name_id = f"{task_name}_{task_id}"
            
            for episode_id in range(max_episode_id + 1):
                episode_num = f"{episode_id:05d}"
                
                # Check if files exist
                attention_file = eval_folder / attention_folder / task_name_id / attention_filename.format(episode_num)
                video_file = eval_folder / "videos" / task_name_id / f"eval_episode_{episode_num}.mp4"
                
                if not attention_file.exists() or not video_file.exists():
                    continue
                
                # Determine token iteration
                token_indices = tokens if tokens else [None]
                
                # Determine head iteration
                head_indices = heads if heads else [None]
                
                for token_idx in token_indices:
                    for head_idx in head_indices:
                        combinations.append((
                            eval_folder,
                            task_name,
                            task_id,
                            episode_id,
                            timesteps,
                            layer,
                            attention_type,
                            head_agg,
                            token_agg,
                            alpha,
                            colormap,
                            token_idx,
                            head_idx,
                        ))
    
    return combinations


def main():
    parser = argparse.ArgumentParser(
        description="Parallel VLM attention visualization"
    )
    parser.add_argument(
        "--eval_folder",
        type=Path,
        required=True,
        help="Evaluation output folder"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="libero_object",
        help="Task name (default: libero_object)"
    )
    parser.add_argument(
        "--max_task_id",
        type=int,
        default=9,
        help="Maximum task ID to process (default: 9)"
    )
    parser.add_argument(
        "--max_episode_id",
        type=int,
        default=1,
        help="Maximum episode ID to process (default: 1)"
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="task",
        choices=["task", "general"],
        help="Type of VLM attention: 'task' (task-specific) or 'general' (baseline, default: task)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        default=[17],
        help="Layers to visualize (default: 17)"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs='*',
        default=[],
        help="Specific token indices to visualize (empty = aggregate all)"
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs='*',
        default=[],
        help="Specific head indices to visualize (empty = aggregate all)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs='+',
        default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        help="Timesteps to visualize"
    )
    parser.add_argument(
        "--head_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "sum", "min"],
        help="Head aggregation method (default: mean)"
    )
    parser.add_argument(
        "--token_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "sum", "min"],
        help="Token aggregation method (default: mean)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay alpha (default: 0.5)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="hot",
        help="Colormap (default: hot)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect CPU count if not specified
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    
    print("=" * 80)
    print(f"Parallel VLM Attention Visualization ({args.attention_type.capitalize()})")
    print("=" * 80)
    print(f"Attention type: {args.attention_type}")
    print(f"Eval folder: {args.eval_folder}")
    print(f"Task range: {args.task_name}_0 to {args.task_name}_{args.max_task_id}")
    print(f"Episode range: 0 to {args.max_episode_id}")
    print(f"Layers: {args.layers}")
    print(f"Tokens: {args.tokens if args.tokens else 'Aggregate all'}")
    print(f"Heads: {args.heads if args.heads else 'Aggregate all'}")
    print(f"Timesteps: {len(args.timesteps)} steps")
    print(f"Workers: {args.num_workers}")
    print("=" * 80)
    
    # Generate all combinations
    print("\nGenerating parameter combinations...")
    combinations = generate_all_combinations(
        eval_folder=args.eval_folder,
        task_name=args.task_name,
        max_task_id=args.max_task_id,
        max_episode_id=args.max_episode_id,
        layers=args.layers,
        tokens=args.tokens,
        heads=args.heads,
        timesteps=args.timesteps,
        attention_type=args.attention_type,
        head_agg=args.head_aggregation,
        token_agg=args.token_aggregation,
        alpha=args.alpha,
        colormap=args.colormap,
    )
    
    total_combinations = len(combinations)
    print(f"Total combinations to process: {total_combinations}")
    
    if total_combinations == 0:
        print("No valid combinations found. Check file paths.")
        return
    
    # Process in parallel
    print(f"\nProcessing with {args.num_workers} workers...")
    print("Progress: ", end="", flush=True)
    
    with mp.Pool(processes=args.num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(visualize_one_combination, combinations)):
            results.append(result)
            
            # Progress indicator (every 10%)
            if (i + 1) % max(1, total_combinations // 10) == 0:
                progress = (i + 1) / total_combinations * 100
                print(f"{progress:.0f}% ", end="", flush=True)
    
    print("\n")
    
    # Summary
    successes = sum(1 for success, _ in results if success)
    failures = sum(1 for success, _ in results if not success)
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Successful: {successes} / {total_combinations}")
    print(f"✗ Failed: {failures} / {total_combinations}")
    
    if failures > 0:
        print("\nFailed combinations (first 10):")
        failed_messages = [msg for success, msg in results if not success]
        for msg in failed_messages[:10]:
            print(f"  - {msg}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
