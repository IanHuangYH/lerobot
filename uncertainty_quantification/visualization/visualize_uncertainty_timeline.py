#!/usr/bin/env python
"""
Timeline visualization for uncertainty scores over episode duration.

This script creates line plots showing how uncertainty evolves throughout
an episode, helping identify uncertainty spikes and patterns over time.

Outputs:
1. Overall uncertainty timeline (single line)
2. Camera comparison (agentview vs wrist)
3. Optional: Multi-episode comparison (successful vs failed)
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def load_episode_uncertainties(uncertainty_path: Path) -> Dict[str, List[float]]:
    """
    Load all uncertainty scores from an episode.
    
    Args:
        uncertainty_path: Path to episode_XXXXX_uncertainty.pt file
    
    Returns:
        Dictionary containing lists of scores:
        {
            'timesteps': [0, 1, 2, ...],
            'overall': [0.12, 0.15, ...],
            'agentview': [0.10, 0.14, ...],
            'wrist': [0.08, 0.11, ...]
        }
    """
    data = torch.load(uncertainty_path, map_location='cpu')
    rollout_steps = data['rollout_steps']
    
    timesteps = []
    overall_scores = []
    agentview_scores = []
    wrist_scores = []
    
    for step_data in rollout_steps:
        timestep = step_data['step']
        uncertainty = step_data['uncertainty']
        
        timesteps.append(timestep)
        overall_scores.append(uncertainty['overall'])
        agentview_scores.append(uncertainty['agentview'])
        wrist_scores.append(uncertainty['wrist'])
    
    return {
        'timesteps': timesteps,
        'overall': overall_scores,
        'agentview': agentview_scores,
        'wrist': wrist_scores,
    }


def plot_uncertainty_timeline(
    uncertainties: Dict[str, List[float]],
    output_path: Path,
    title: str = "Uncertainty Timeline",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot overall uncertainty timeline for a single episode.
    
    Args:
        uncertainties: Dictionary from load_episode_uncertainties()
        output_path: Path to save the plot (PNG)
        title: Plot title
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    timesteps = uncertainties['timesteps']
    overall = uncertainties['overall']
    
    # Plot overall uncertainty
    ax.plot(timesteps, overall, linewidth=2, color='#1f77b4', marker='o', 
            markersize=3, label='Overall Uncertainty')
    
    # Add statistics
    mean_unc = np.mean(overall)
    max_unc = np.max(overall)
    max_idx = np.argmax(overall)
    
    ax.axhline(mean_unc, color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_unc:.4f}', alpha=0.7)
    ax.scatter(timesteps[max_idx], max_unc, color='red', s=100, zorder=5,
               label=f'Max: {max_unc:.4f} @ t={timesteps[max_idx]}')
    
    # Styling
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Uncertainty Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved overall timeline: {output_path}")


def plot_camera_comparison(
    uncertainties: Dict[str, List[float]],
    output_path: Path,
    title: str = "Camera-wise Uncertainty Comparison",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot agentview vs wrist uncertainty comparison.
    
    Args:
        uncertainties: Dictionary from load_episode_uncertainties()
        output_path: Path to save the plot (PNG)
        title: Plot title
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    timesteps = uncertainties['timesteps']
    overall = uncertainties['overall']
    agentview = uncertainties['agentview']
    wrist = uncertainties['wrist']
    
    # Plot all three lines
    ax.plot(timesteps, overall, linewidth=2, color='black', marker='o', 
            markersize=3, label='Overall', alpha=0.8)
    ax.plot(timesteps, agentview, linewidth=2, color='#ff7f0e', marker='s', 
            markersize=3, label='Agentview', alpha=0.8)
    ax.plot(timesteps, wrist, linewidth=2, color='#2ca02c', marker='^', 
            markersize=3, label='Wrist', alpha=0.8)
    
    # Add statistics
    mean_overall = np.mean(overall)
    mean_ag = np.mean(agentview)
    mean_wrist = np.mean(wrist)
    
    ax.axhline(mean_overall, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(mean_ag, color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(mean_wrist, color='#2ca02c', linestyle='--', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Uncertainty Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Enhanced legend with means
    legend_labels = [
        f'Overall (mean: {mean_overall:.4f})',
        f'Agentview (mean: {mean_ag:.4f})',
        f'Wrist (mean: {mean_wrist:.4f})',
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc='best', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved camera comparison: {output_path}")


def plot_multi_episode_comparison(
    episode_data: List[Dict],
    output_path: Path,
    title: str = "Multi-Episode Uncertainty Comparison",
    figsize: Tuple[int, int] = (14, 7),
) -> None:
    """
    Plot multiple episodes on the same timeline for comparison.
    
    Args:
        episode_data: List of dictionaries with 'name', 'uncertainties' keys
        output_path: Path to save the plot (PNG)
        title: Plot title
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(episode_data)))
    
    for idx, ep_data in enumerate(episode_data):
        name = ep_data['name']
        uncertainties = ep_data['uncertainties']
        
        timesteps = uncertainties['timesteps']
        overall = uncertainties['overall']
        agentview = uncertainties['agentview']
        wrist = uncertainties['wrist']
        
        color = colors[idx]
        
        # Plot overall in top subplot
        ax1.plot(timesteps, overall, linewidth=2, color=color, marker='o', 
                markersize=2, label=name, alpha=0.8)
        
        # Plot cameras in bottom subplot
        ax2.plot(timesteps, agentview, linewidth=1.5, color=color, linestyle='-',
                marker='s', markersize=2, label=f'{name} (agentview)', alpha=0.7)
        ax2.plot(timesteps, wrist, linewidth=1.5, color=color, linestyle='--',
                marker='^', markersize=2, label=f'{name} (wrist)', alpha=0.7)
    
    # Styling for top subplot
    ax1.set_ylabel('Overall Uncertainty', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    
    # Styling for bottom subplot
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Camera Uncertainty', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=8, ncol=2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved multi-episode comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize uncertainty timelines")
    parser.add_argument("--uncertainty_paths", type=str, nargs="+", required=True,
                        help="Path(s) to episode_XXXXX_uncertainty.pt file(s)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for timeline plots")
    parser.add_argument("--episode_names", type=str, nargs="+", default=None,
                        help="Names for episodes (for multi-episode comparison)")
    parser.add_argument("--plot_type", type=str, default="all",
                        choices=["overall", "camera", "all"],
                        help="Which plots to generate: overall, camera, or all")
    parser.add_argument("--compare_episodes", action="store_true",
                        help="Create multi-episode comparison plot")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading uncertainty data from {len(args.uncertainty_paths)} file(s)...")
    print(f"Output directory: {output_dir}")
    print("")
    
    # Load all episodes
    all_episodes = []
    for idx, unc_path in enumerate(args.uncertainty_paths):
        unc_path = Path(unc_path)
        
        if not unc_path.exists():
            print(f"⚠ Warning: {unc_path} not found, skipping...")
            continue
        
        print(f"Loading: {unc_path.name}")
        uncertainties = load_episode_uncertainties(unc_path)
        
        # Determine episode name
        if args.episode_names and idx < len(args.episode_names):
            name = args.episode_names[idx]
        else:
            # Extract episode number from filename
            episode_num = unc_path.stem.split('_')[-2]  # e.g., episode_00003_uncertainty -> 00003
            name = f"Episode {episode_num}"
        
        all_episodes.append({
            'name': name,
            'uncertainties': uncertainties,
            'path': unc_path,
        })
        
        # Generate individual episode plots
        if args.plot_type in ["overall", "all"]:
            output_file = output_dir / f"{unc_path.stem}_timeline.png"
            plot_uncertainty_timeline(
                uncertainties,
                output_file,
                title=f"Uncertainty Timeline - {name}",
            )
        
        if args.plot_type in ["camera", "all"]:
            output_file = output_dir / f"{unc_path.stem}_camera_comparison.png"
            plot_camera_comparison(
                uncertainties,
                output_file,
                title=f"Camera Uncertainty - {name}",
            )
        
        print("")
    
    # Generate multi-episode comparison if requested
    if args.compare_episodes and len(all_episodes) > 1:
        print(f"Creating multi-episode comparison with {len(all_episodes)} episodes...")
        output_file = output_dir / "multi_episode_comparison.png"
        plot_multi_episode_comparison(
            all_episodes,
            output_file,
            title=f"Uncertainty Comparison ({len(all_episodes)} Episodes)",
        )
        print("")
    
    print("✓ Timeline visualization complete!")
    print(f"  Total plots generated: {output_dir}")


if __name__ == "__main__":
    main()
