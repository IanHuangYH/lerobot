"""
CLI script to collect RND training data from LIBERO demonstrations.

Usage:
    python -m uncertainty_quantification.scripts.collect_rnd_training_data \
        --task-type spatial \
        --num-episodes 500 \
        --tokens-per-frame 64
"""

import argparse
import logging
from pathlib import Path

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Collect RND training data from LIBERO demonstrations"
    )
    
    # Task configuration
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=["spatial", "object", "goal", "long"],
        help="LIBERO task type to process"
    )
    
    # Data collection parameters
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to use (default: None = use all available ~500 per task)"
    )
    parser.add_argument(
        "--tokens-per-frame",
        type=int,
        default=64,
        help="Number of tokens to sample per camera per frame (default: 64)"
    )
    parser.add_argument(
        "--save-full-tokens",
        action="store_true",
        help="Save all 256 tokens instead of sampling (overrides --tokens-per-frame)"
    )
    
    # Paths
    parser.add_argument(
        "--libero-dataset-dir",
        type=Path,
        default=Path("third_party/LIBERO/datasets"),
        help="Path to LIBERO datasets directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("uncertainty_quantification/rnd_dataset"),
        help="Output directory for collected datasets"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="lerobot/pi05_libero_finetuned",
        help="Path or HuggingFace ID of Pi0.5 policy checkpoint"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on. Examples: 'cuda', 'cuda:0', 'cuda:1', 'cpu' (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid circular dependency
    from uncertainty_quantification.dataset import collect_rnd_dataset
    
    logger.info("="*60)
    logger.info("RND Training Data Collection")
    logger.info("="*60)
    logger.info(f"Task type: {args.task_type}")
    logger.info(f"Number of episodes: {args.num_episodes if args.num_episodes else 'ALL (~500)'}")
    logger.info(f"Tokens per frame: {256 if args.save_full_tokens else args.tokens_per_frame}")
    logger.info(f"Device: {args.device}")
    
    # Show GPU information if using CUDA
    if "cuda" in args.device.lower():
        if torch.cuda.is_available():
            device_id = args.device.split(":")[-1] if ":" in args.device else "0"
            if device_id.isdigit():
                gpu_id = int(device_id)
                if gpu_id < torch.cuda.device_count():
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    logger.info(f"GPU: {gpu_name} (ID: {gpu_id})")
                else:
                    logger.warning(f"GPU {gpu_id} requested but only {torch.cuda.device_count()} GPUs available")
        else:
            logger.warning("CUDA requested but not available, will use CPU")
            args.device = "cpu"
    
    logger.info(f"LIBERO dataset: {args.libero_dataset_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*60)
    
    # Load Pi0.5 policy
    logger.info(f"Loading Pi0.5 policy from: {args.policy_path}")
    policy = PI05Policy.from_pretrained(args.policy_path)
    
    # Log the expected image feature keys from the policy config
    expected_image_keys = list(policy.config.image_features.keys())
    logger.info(f"Policy expects image keys: {expected_image_keys}")
    
    policy.eval()
    policy.to(args.device)
    
    # Freeze policy (no gradient computation needed)
    for param in policy.parameters():
        param.requires_grad = False
    
    logger.info("Policy loaded and frozen")
    
    # Collect dataset
    stats = collect_rnd_dataset(
        policy=policy,
        task_type=args.task_type,
        libero_dataset_dir=args.libero_dataset_dir,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        tokens_per_frame=args.tokens_per_frame,
        device=args.device,
        save_full_tokens=args.save_full_tokens,
    )
    
    logger.info("="*60)
    logger.info("Collection Complete!")
    logger.info(f"  Total episodes: {stats['total_episodes']}")
    logger.info(f"  Total frames: {stats['total_frames']}")
    logger.info(f"  Total tokens (agentview): {stats['total_tokens_agentview']:,}")
    logger.info(f"  Total tokens (wrist): {stats['total_tokens_wrist']:,}")
    
    # Calculate dataset sizes
    agentview_size_gb = stats['total_tokens_agentview'] * 2048 * 4 / (1024**3)
    wrist_size_gb = stats['total_tokens_wrist'] * 2048 * 4 / (1024**3)
    total_size_gb = agentview_size_gb + wrist_size_gb
    
    logger.info(f"  Dataset size: {total_size_gb:.2f} GB")
    logger.info(f"    - Agentview: {agentview_size_gb:.2f} GB")
    logger.info(f"    - Wrist: {wrist_size_gb:.2f} GB")
    logger.info("="*60)


if __name__ == "__main__":
    main()
