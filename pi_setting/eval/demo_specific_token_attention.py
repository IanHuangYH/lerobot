#!/usr/bin/env python
"""
Demo script showing how to visualize attention from specific tokens.

This addresses the question: "How do we see which specific objects the model attends to?"

Instead of averaging across all task tokens (which dilutes the signal), this script
shows how to visualize attention from specific meaningful tokens like "alphabet", "soup", "basket".

Usage:
    python demo_specific_token_attention.py \\
        --attention_file eval_logs/quick_test_vlm_attention_1/vlm_attention/libero_object_0/episode_00000_vlm_attention.pt \\
        --rollout_step 10 \\
        --task_name libero_object \\
        --task_id 0 \\
        --tokens_of_interest alphabet soup basket
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "pi_setting" / "eval"))
from visualize_vlm_attention import get_libero_task_instruction, load_vlm_attention_data
from token_boundary_helper import TokenBoundaryHelper


def analyze_token_attention(
    attention_file: Path,
    rollout_step: int,
    task_name: str,
    task_id: int,
    tokens_of_interest: list = None,
    layer: int = 17,
):
    """
    Analyze and visualize attention from specific tokens.
    
    Args:
        attention_file: Path to VLM attention .pt file
        rollout_step: Which timestep to analyze
        task_name: LIBERO task suite name
        task_id: LIBERO task ID
        tokens_of_interest: List of words to focus on (e.g., ['alphabet', 'soup', 'basket'])
        layer: Which transformer layer
    """
    # Get task text
    task_text = get_libero_task_instruction(task_name, task_id)
    print(f"Task: \"{task_text}\"")
    print()
    
    # Load attention
    attention_weights, prefix_len = load_vlm_attention_data(attention_file, rollout_step, layer)
    print(f"Loaded attention: shape={attention_weights.shape}, prefix_len={prefix_len}")
    
    # Tokenize to find boundaries
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    
    full_text = f"Task: {task_text}, State: " + " ".join(["128"] * 32) + ";\nAction: "
    encoding = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
    
    # Find task token boundaries
    helper = TokenBoundaryHelper()
    boundaries = helper.find_token_boundaries(full_text, num_img_tokens=768)
    task_start, task_end = boundaries['task']
    
    print(f"\nTask spans tokens [{task_start}, {task_end})")
    print(f"Task has {task_end - task_start} tokens")
    print()
    
    # Show all tokens in task
    print("=" * 80)
    print("ALL TASK TOKENS:")
    print("=" * 80)
    
    # Get token IDs for task portion
    task_token_ids = encoding['input_ids'][1:task_end-task_start+1]  # Skip <bos>
    task_offsets = encoding['offset_mapping'][1:task_end-task_start+1]
    
    for i, (token_id, (start, end)) in enumerate(zip(task_token_ids, task_offsets)):
        token_text = tokenizer.decode([token_id])
        actual_idx = task_start + i
        
        # Highlight tokens of interest
        if tokens_of_interest:
            is_interesting = any(word.lower() in token_text.lower() for word in tokens_of_interest)
            marker = " ★" if is_interesting else ""
        else:
            marker = ""
        
        print(f"  Token {actual_idx:3d}: '{token_text:15s}' (chars {start:3d}-{end:3d}){marker}")
    
    print()
    print("=" * 80)
    
    # Calculate average attention to images for each token
    print("ATTENTION TO IMAGES (averaged across all image patches):")
    print("=" * 80)
    
    attention = attention_weights[0]  # Remove batch dim: [8, 968, 968]
    
    # For each task token, compute mean attention to all image patches
    img_start, img_end = boundaries['images']  # (0, 768)
    
    token_to_img_attention = []
    for i in range(task_end - task_start):
        token_idx = task_start + i
        # Average across heads and image patches
        att_to_imgs = attention[:, token_idx, img_start:img_end].mean().item()
        token_to_img_attention.append(att_to_imgs)
    
    # Show top tokens by image attention
    token_att_pairs = list(zip(range(len(token_to_img_attention)), token_to_img_attention))
    token_att_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 tokens by average attention to images:")
    for rank, (i, att) in enumerate(token_att_pairs[:10], 1):
        token_id = task_token_ids[i]
        token_text = tokenizer.decode([token_id])
        actual_idx = task_start + i
        print(f"  {rank:2d}. Token {actual_idx:3d}: '{token_text:15s}' - attention={att:.6f}")
    
    print()
    print("=" * 80)
    print("HOW TO VISUALIZE SPECIFIC TOKENS:")
    print("=" * 80)
    print()
    print("ATTENTION INTERPRETATION:")
    print("  In VLM self-attention: attention[i, j] = 'token i attends to token j'")
    print("  We extract: attention[text_token, image_patches]")
    print("  This shows: 'When reading this word, which image regions does the model look at?'")
    print()
    print("EXAMPLE: Visualize 'soup' token (e.g., token 774):")
    print()
    print("Using bash script:")
    print("  # Edit run_visualize_vlm_attention.sh:")
    print("  SPECIFIC_TOKEN_IDX=774")
    print("  ./pi_setting/eval/run_visualize_vlm_attention.sh")
    print()
    print("Or directly:")
    print("  python pi_setting/eval/visualize_vlm_attention.py \\")
    print("      --attention_file eval_logs/.../episode_00000_vlm_attention.pt \\")
    print("      --video_file eval_logs/.../eval_episode_00000.mp4 \\")
    print("      --output_dir viz_output \\")
    print("      --task_name libero_object --task_id 0 \\")
    print("      --rollout_steps 10 \\")
    print("      --specific_token_idx 774")
    print()
    print("This will show which image regions 'soup' attends to!")
    print()
    
    if tokens_of_interest:
        print("=" * 80)
        print(f"TOKENS MATCHING YOUR KEYWORDS: {tokens_of_interest}")
        print("=" * 80)
        
        for i, (token_id, (start, end)) in enumerate(zip(task_token_ids, task_offsets)):
            token_text = tokenizer.decode([token_id])
            actual_idx = task_start + i
            
            is_interesting = any(word.lower() in token_text.lower() for word in tokens_of_interest)
            if is_interesting:
                att = token_to_img_attention[i]
                print(f"  Token {actual_idx:3d}: '{token_text:15s}' - attention to images: {att:.6f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze attention from specific tokens")
    parser.add_argument("--attention_file", type=Path, required=True)
    parser.add_argument("--rollout_step", type=int, default=0)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--tokens_of_interest", type=str, nargs='+', default=['alphabet', 'soup', 'basket'])
    parser.add_argument("--layer", type=int, default=17)
    
    args = parser.parse_args()
    
    analyze_token_attention(
        args.attention_file,
        args.rollout_step,
        args.task_name,
        args.task_id,
        args.tokens_of_interest,
        args.layer,
    )


if __name__ == "__main__":
    main()
