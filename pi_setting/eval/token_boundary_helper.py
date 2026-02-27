#!/usr/bin/env python
"""
Token Boundary Helper for VLM Attention Analysis

This module helps identify boundaries between different token types in the Pi0.5 model:
- Image tokens (from SigLIP vision encoder)
- Task instruction tokens (natural language)
- Robot state tokens (discretized joint/gripper states)

Based on debug_token_boundaries.py analysis.
"""

import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer


class TokenBoundaryHelper:
    """Helper class to identify token boundaries in Pi0.5 VLM prefix."""
    
    def __init__(self, tokenizer_name: str = "google/paligemma-3b-pt-224"):
        """
        Initialize the helper with a tokenizer.
        
        Args:
            tokenizer_name: HuggingFace model name for tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def find_token_boundaries(
        self,
        full_text: str,
        num_img_tokens: int = 768,  # 256 patches × 3 cameras
    ) -> Dict[str, Tuple[int, int]]:
        """
        Find token boundaries for different parts of the input sequence.
        
        Args:
            full_text: Complete text prompt (e.g., "Task: ..., State: ...; Action: ")
            num_img_tokens: Number of image tokens from vision encoder
        
        Returns:
            Dictionary with token ranges (inclusive start, exclusive end):
            {
                'images': (0, 768),              # Image patches from 3 cameras
                'task': (768, X),                # Task instruction text
                'state': (X, Y),                 # Discretized robot state
                'action_prefix': (Y, total_len)  # "\\nAction: "
            }
        """
        # Tokenize the full text with special tokens
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']
        
        # Image tokens come first (before <bos> in attention space)
        # But in the actual sequence after <bos>, so we adjust indices
        img_start = 0
        img_end = num_img_tokens
        
        # Find character positions of key markers
        state_start_char = full_text.find("State:")
        state_end_char = full_text.find(";") + 1  # Include semicolon
        action_start_char = full_text.find("\nAction:")
        
        # Find corresponding token indices using character offsets
        state_start_token = None
        state_end_token = None
        action_start_token = None
        
        for i, (start, end) in enumerate(offsets):
            # Adjust for image tokens that come before text
            token_idx = i + num_img_tokens - 1  # -1 because offsets include <bos>
            
            if start <= state_start_char < end and state_start_token is None:
                state_start_token = token_idx
            if start < state_end_char <= end and state_end_token is None:
                state_end_token = token_idx + 1  # Exclusive end
            if start <= action_start_char < end and action_start_token is None:
                action_start_token = token_idx
        
        # Task tokens: from end of images to start of state
        task_start = img_end
        task_end = state_start_token
        
        # Total sequence length
        total_len = len(tokens) + num_img_tokens - 1  # -1 for <bos>
        
        return {
            'images': (img_start, img_end),
            'task': (task_start, task_end),
            'state': (state_start_token, state_end_token),
            'action_prefix': (action_start_token, total_len),
            'total_len': total_len
        }
    
    def create_token_masks(
        self,
        boundaries: Dict[str, Tuple[int, int]],
        total_len: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Create boolean masks for different token types.
        
        Args:
            boundaries: Output from find_token_boundaries()
            total_len: Total sequence length (optional, inferred from boundaries)
        
        Returns:
            Dictionary of boolean masks:
            {
                'images': tensor([True, True, ..., False, ...]),
                'task': tensor([False, ..., True, True, ..., False, ...]),
                'state': tensor([False, ..., True, True, ...]),
                'language': tensor([False, ..., True, True, ...])  # task + state
            }
        """
        if total_len is None:
            if 'total_len' in boundaries:
                total_len = boundaries['total_len']
            else:
                # Find max end value from all tuples
                total_len = max(end for key, value in boundaries.items() 
                              if isinstance(value, tuple) for _, end in [value])
        
        # Create masks
        img_mask = torch.zeros(total_len, dtype=torch.bool)
        task_mask = torch.zeros(total_len, dtype=torch.bool)
        state_mask = torch.zeros(total_len, dtype=torch.bool)
        
        # Fill masks
        if 'images' in boundaries and isinstance(boundaries['images'], tuple):
            start, end = boundaries['images']
            img_mask[start:end] = True
        
        if 'task' in boundaries and isinstance(boundaries['task'], tuple):
            start, end = boundaries['task']
            task_mask[start:end] = True
        
        if 'state' in boundaries and isinstance(boundaries['state'], tuple):
            start, end = boundaries['state']
            state_mask[start:end] = True
        
        # Language = task + state
        language_mask = task_mask | state_mask
        
        return {
            'images': img_mask,
            'task': task_mask,
            'state': state_mask,
            'language': language_mask,
        }
    
    def get_task_token_indices(self, boundaries: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Get the start and end indices of task instruction tokens.
        
        Args:
            boundaries: Output from find_token_boundaries()
        
        Returns:
            Tuple of (start_idx, end_idx) for task tokens
        """
        return boundaries['task']
    
    def print_summary(self, boundaries: Dict[str, Tuple[int, int]]):
        """
        Print a human-readable summary of token boundaries.
        
        Args:
            boundaries: Output from find_token_boundaries()
        """
        print("Token Boundary Summary:")
        print("=" * 60)
        
        for key in boundaries.keys():
            if key == 'total_len':
                continue
            start, end = boundaries[key]
            count = end - start
            print(f"  {key:15s}: tokens [{start:4d}, {end:4d}) - {count:3d} tokens")
        
        print("=" * 60)
        print(f"  Total sequence length: {boundaries.get('total_len', 'N/A')} tokens")


def example_usage():
    """Example usage of TokenBoundaryHelper."""
    
    # Example text from LIBERO evaluation
    full_text = "Task: pick up the alphabet soup and place it in the basket, State: -1 93 -1 190 158 141 234 23 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128;\nAction: "
    
    helper = TokenBoundaryHelper()
    
    # Find boundaries
    boundaries = helper.find_token_boundaries(full_text, num_img_tokens=768)
    helper.print_summary(boundaries)
    
    # Create masks
    masks = helper.create_token_masks(boundaries)
    
    print("\nMask Statistics:")
    print(f"  Image tokens: {masks['images'].sum().item()} True values")
    print(f"  Task tokens: {masks['task'].sum().item()} True values")
    print(f"  State tokens: {masks['state'].sum().item()} True values")
    print(f"  Language tokens: {masks['language'].sum().item()} True values")
    
    # Example: Extract task token indices for attention analysis
    task_start, task_end = helper.get_task_token_indices(boundaries)
    print(f"\nTask instruction spans tokens [{task_start}, {task_end})")
    print("  Use this range to extract task→image attention:")
    print(f"    attention[:, :, {task_start}:{task_end}, :768]")


if __name__ == "__main__":
    example_usage()
