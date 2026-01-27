# Attention Map Saving for Pi0.5 Evaluation

This guide explains how to save and analyze attention maps during Pi0.5 policy evaluation on LIBERO tasks.

## Quick Reference

### How to Enable
```bash
lerobot-eval \
  --eval.save_attention_maps=true \
  --policy.compile_model=false \  # REQUIRED!
  --policy.path=lerobot/pi05_libero_finetuned \
  ...
```

### What Gets Saved
- **One file per episode**: `episode_00000_attention.pt` (~2.8 GB each)
- **Location**: `{output_dir}/attention/{task_group}_{task_id}/`
- **Content**: All action predictions × all denoising steps × last 2 layers

### Data Structure
```python
{
    'rollout_steps': [  # One per action in episode (~145 total)
        {
            'rollout_step': 0,
            'attention_maps': {  # All 10 denoising steps (t=1.0 → t=0.0)
                0: {'time': 1.0, 'attention_weights': {16: tensor, 17: tensor}, ...},
                ...
                9: {'time': 0.0, 'attention_weights': {16: tensor, 17: tensor}, ...}
            }
        },
        ...
    ],
    'metadata': {'episode_index': 0, 'num_rollout_steps': 145, 'num_denoising_steps': 10}
}
```

## Overview

When evaluating the Pi0.5 policy, you can optionally save attention weights from the transformer layers during each denoising step. This is useful for:

- Understanding how the model attends to visual vs. language tokens
- Debugging model behavior on specific tasks
- Analyzing which image regions influence action prediction
- Visualizing attention patterns across time

## Quick Start

### 1. Enable Attention Saving in Evaluation Script

Add `--eval.save_attention_maps=true` to your `lerobot-eval` command:

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-eval \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.save_attention_maps=true \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.device=cuda:0 \
  --output_dir=./eval_logs/my_eval \
  --env.task_ids='[0]'
```

### 2. Run the Evaluation

The script `pi_setting/eval/eval_libero_quick_test.sh` already has this flag enabled:

```bash
bash pi_setting/eval/eval_libero_quick_test.sh
```

### 3. Check Saved Attention Maps

After evaluation completes, attention maps are saved in:
```
eval_logs/quick_test_object/
├── attention/          # <-- Attention maps saved here
│   └── libero_object_0/
│       ├── episode_00000_attention.pt
│       ├── episode_00001_attention.pt
│       └── ...
├── videos/             # Rendered episode videos
│   └── libero_object_0/
└── eval_info.json      # Evaluation metrics
```

### 4. Inspect Saved Files

Use the inspection script to view attention map structure:

```bash
python pi_setting/eval/inspect_saved_attention.py eval_logs/quick_test_object/attention/libero_object_0/
```

Output example:
```
Found 1 attention map files in eval_logs/quick_test_object/attention/libero_object_0

================================================================================
File: episode_00000_attention.pt
================================================================================

📋 Metadata:
   Episode Index: 0
   Number of rollout steps: 145
   Number of denoising steps per action: 10

🔍 Structure:
   Total actions in episode: 145
   Denoising steps per action: 10
   
   Example (Rollout Step 0, Denoising Step 9):
      Time: t≈0.000 (final refinement)
      Prefix length (visual+language tokens): 276
      Suffix length (action tokens): 7
      Layers saved: 16, 17 (last 2 layers)

      Layer 17 attention shape: torch.Size([1, 8, 283, 283])
         (batch=1, heads=8, total_seq=283, total_seq=283)
         where total_seq = prefix_len + suffix_len = 276 + 7

💾 File Size: ~2.8 GB (145 rollout steps × 10 denoising steps × 2 layers)
```

## File Format

Each `.pt` file contains **all rollout steps for one episode**:

```python
{
    'rollout_steps': [  # List of attention maps, one per action prediction
        {
            'rollout_step': int,      # Which action step in the episode (0, 1, 2, ...)
            'attention_maps': dict,   # Attention from all denoising steps (see below)
        },
        ...
    ],
    'metadata': {
        'episode_index': int,         # Global episode number
        'num_rollout_steps': int,     # Total actions taken in episode (e.g., 145)
        'num_denoising_steps': int,   # Flow matching steps per action (10)
    }
}
```

### Attention Maps Structure (Per Rollout Step)

Each rollout step contains attention from **all denoising iterations**:

```python
attention_maps = {
    0: {  # First denoising step (t=1.0, pure noise)
        'time': 1.0,               # Flow matching timestep
        'prefix_len': 276,         # Visual + language tokens
        'suffix_len': 7,           # Action tokens
        'attention_weights': {
            16: Tensor,            # Layer 16: shape (batch, heads, seq, seq)
            17: Tensor,            # Layer 17: shape (batch, heads, seq, seq)
        }
    },
    1: {  # Second denoising step (t≈0.9)
        'time': 0.9,
        ...
    },
    ...
    9: {  # Final denoising step (t≈0.0, clean action)
        'time': 0.0,
        ...
    }
}
```

## Accessing Saved Attention Maps

### Basic Access Pattern

```python
import torch

# Load episode data
data = torch.load("eval_logs/quick_test_object/attention/libero_object_0/episode_00000_attention.pt")

# Get metadata
num_rollout_steps = data['metadata']['num_rollout_steps']  # e.g., 145
num_denoising_steps = data['metadata']['num_denoising_steps']  # 10

# Access specific rollout step (e.g., 15th action prediction)
rollout_step_15 = data['rollout_steps'][15]
step_index = rollout_step_15['rollout_step']  # Should be 15*run_chunk_step
attention_maps = rollout_step_15['attention_maps']  # All 10 denoising steps

# Access final denoising step (most refined action)
final_denoise = attention_maps[9]  # t≈0.0
time_value = final_denoise['time']
prefix_len = final_denoise['prefix_len']  # e.g., 968
suffix_len = final_denoise['suffix_len']  # e.g., 50

# Get attention from last layer
layer_17_attention = final_denoise['attention_weights'][17]
# Shape: (1, 8, 50, 1018) = (batch, heads, query_len, key_len)
# action_chunk = 50 (1: time embedding + 49:action component)
# 1018 = 768 (3 cameras (third person view + wrist view + dummy view) × 256 patches) + 200 (language) + 50 (suffix)
print(f"Attention shape: {layer_17_attention.shape}")

```

### Extracting Action → Visual Attention

```python
from visualize_pi05_attention import extract_action_to_visual_attention

# Get attention from a specific rollout step and denoising iteration
rollout_idx = 15  # Which action in the episode
denoise_idx = 9   # Final denoising step (t≈0)
layer_idx = 17    # Last layer

# Extract attention
step_data = data['rollout_steps'][rollout_idx]
att_maps = step_data['attention_maps'][denoise_idx]
att_weights = att_maps['attention_weights'][layer_idx]

# Extract action → visual attention
action_to_visual = extract_action_to_visual_attention(
    att_weights,
    prefix_len=att_maps['prefix_len'],
    suffix_len=att_maps['suffix_len'],
    num_img_tokens=256,  # SigLIP visual tokens
)
# Shape: (batch=1, heads=8, action_tokens=7, visual_tokens=256)

print(f"Average attention from actions to visual: {action_to_visual.mean():.4f}")
```

## Storage Considerations

### File Sizes (Current Implementation)

By default, the last **2 layers** are saved (layers 16-17 out of 18 total):

- **Per episode**: ~2-3 GB 
  - 145 rollout steps × 10 denoising steps × 2 layers × ~1MB per layer
- **Per 10 episodes**: ~20-30 GB
- **Warning**: Storage grows quickly!

### Reducing Storage

Modify in [modeling_pi05.py](../../src/lerobot/policies/pi05/modeling_pi05.py) line 638:

```python
def get_attention_maps(self, last_n_layers: int = 2):  # Change this number
```

**Options:**
1. **Save 1 layer only**: `last_n_layers=1` → ~1.5 GB per episode
2. **Save all 18 layers**: `last_n_layers=18` → ~27 GB per episode
3. **Save specific episodes**: Add filtering logic in `lerobot_eval.py` to skip saving for certain episodes
4. **Save only final denoising step**: Modify rollout() to only collect `attention_maps[9]`

## Implementation Details

### Modified Files

1. **`src/lerobot/policies/pi05/modeling_pi05.py`**
   - Added `save_attention_maps` flag and methods: `enable_attention_map_saving()`, `get_attention_maps()`, `clear_attention_maps()`
   - Modified `denoise_step()` to collect attention at each denoising iteration
   - **Key fix**: Modified `PaliGemmaWithExpertModel.forward()` suffix-only branch to use `output_attentions=True`

2. **`src/lerobot/scripts/lerobot_eval.py`**
   - Added `save_attention_maps` parameter to `rollout()` function
   - Collects attention after each `select_action()` call
   - Saves per-episode `.pt` files with all rollout steps

3. **`src/lerobot/configs/default.py`**
   - Added `save_attention_maps: bool = False` to `EvalConfig`

### Critical Bug Fix

**Problem**: Attention weights were always `None` during inference.

**Root cause**: The `PaliGemmaWithExpertModel.forward()` method only collected attention when both prefix and suffix embeddings were provided together. During denoising, only suffix is passed (`inputs_embeds=[None, suffix_embs]`), so attention was never captured.

**Solution**: Modified the suffix-only branch to call `self.gemma_expert.model.forward(..., output_attentions=True)` and extract attention from the transformers output.

### Why Compilation Must Be Disabled

`torch.compile()` optimizes away unused return values. Since `attention_weights` aren't used in the forward pass, compilation removes them. Must use `--policy.compile_model=false` when saving attention.

## Troubleshooting

### "Policy does not support attention map saving"

Only Pi0.5 policy supports this feature. Verify:
- Using `--policy.path=lerobot/pi05_libero_finetuned`
- Policy has `enable_attention_map_saving()` method

### "Attention maps are None or empty"

**✅ FIXED**: This issue has been resolved. The model now correctly collects attention weights from the suffix-only forward pass during denoising by using `output_attentions=True` in the transformers forward call.

### File sizes too large

Each episode saves ~2-3 GB with current settings. Options:
- **Reduce layers**: Change `last_n_layers=1` in `get_attention_maps()` 
- **Save specific episodes only**: Add filtering in `rollout()` to skip certain episodes
- **Save only critical denoising steps**: Modify to save only final step (t≈0) instead of all 10

## Verification

Test that attention saving works:
```bash
bash pi_setting/eval/eval_libero_quick_test.sh

# Check saved file
ls -lh eval_logs/quick_test_object/attention/libero_object_0/

# Inspect structure
python pi_setting/eval/inspect_saved_attention.py \
    eval_logs/quick_test_object/attention/libero_object_0/
```

**Expected output:**
- File exists: `episode_00000_attention.pt`
- Size: ~2.8 GB (for 145 rollout steps)
- No warning about empty attention maps
- Task completes successfully

## Limitations

1. **Large files**: 2-3 GB per episode with default settings
2. **Requires uncompiled model**: ~20-50% slower inference
3. **Memory intensive**: Loading full episode requires ~3 GB RAM
4. **Only Pi0.5**: Other policies not supported yet

## Visualizing Attention Maps

### Verifying Alignment with Video

The verification script overlays attention heatmaps on video frames to check alignment:

```bash
# Quick start - uses default paths
./pi_setting/eval/run_verify_attention_video_alignment.sh

# Or customize:
python pi_setting/eval/verify_attention_video_alignment.py \
    --attention_path eval_logs/quick_test_object/attention/libero_object_0/episode_00000_attention.pt \
    --video_path eval_logs/quick_test_object/videos/libero_object_0/eval_episode_0.mp4 \
    --output_path eval_logs/quick_test_object/verification \
    --timesteps 0 5 10 13 \
    --action_idx 0 \
    --alpha 0.5  # Transparency (0.0=invisible, 1.0=opaque)
```

**Output format:**
```
┌─────────────────────────────────────────────────┐
│         Camera Name | Timestep | Action         │
├────────────────────┬────────────────────────────┤
│                    │                            │
│  Video + Overlay   │    Pure Attention Map      │
│  (with alpha=0.5)  │    (no video background)   │
│                    │                            │
└────────────────────┴────────────────────────────┘
```

**Parameters:**
- `--alpha`: Controls transparency (default: 0.5)
  - Recommended: 0.4-0.6 for best visibility
- `--timesteps`: Which rollout steps to visualize
- `--action_idx`: Which action token to visualize (0-48)
- `--layer`: Transformer layer (default: 17, last layer)
- `--denoising_step`: Denoising iteration (default: 9, final step)

**Timestep Alignment:**
- ✅ `rollout_step = 0` in attention file → Frame 0 in MP4
- ✅ `rollout_step = 10` in attention file → Frame 10 in MP4
- Perfectly aligned! No offset needed.

**Coordinate System Handling:**

| Component | Agentview | Wrist |
|-----------|-----------|-------|
| **LiberoProcessor** | Flipped [::-1, ::-1] | Flipped [::-1, ::-1] |
| **MP4 render()** | Flipped [::-1, ::-1] | NOT flipped |
| **Transformation needed** | ❌ No | ✅ Yes (auto-applied) |

Both scripts automatically handle the correct coordinate transformations to align with MP4 video frames!
