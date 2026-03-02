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

Understanding the coordinate transformations is crucial for correct visualization. There are **three coordinate spaces**:

1. **RAW CAMERA SPACE**: Original camera output (before any processing)
2. **MODEL SPACE**: What the model sees after LiberoProcessor preprocessing
3. **VIDEO SPACE**: What appears in the MP4 file from `render()`

**Attention maps are saved in MODEL SPACE**, so we need transformations to align with VIDEO SPACE.

### Transformation Flow

**Important**: LiberoProcessor and `render()` are **independent operations** that both start from RAW CAMERA!

**AGENTVIEW:**
```
                    ┌─ LiberoProcessor [::-1, ::-1] → MODEL SPACE (flipped)
                    │
RAW CAMERA (original)
                    │
                    └─ render() [::-1, ::-1] → VIDEO SPACE (flipped)
```
- LiberoProcessor: Takes raw camera, flips `[::-1, ::-1]` → MODEL sees **FLIPPED**
- render(): Takes raw camera, flips `[::-1, ::-1]` (line 226) → VIDEO shows **FLIPPED**
- **Result**: Both apply the SAME transformation to raw camera → ✅ **MODEL = VIDEO, aligned!**

**WRIST:**
```
                    ┌─ LiberoProcessor [::-1, ::-1] → MODEL SPACE (flipped)
                    │
RAW CAMERA (original)
                    │
                    └─ render() (no flip) → VIDEO SPACE (original)
```
- LiberoProcessor: Takes raw camera, flips `[::-1, ::-1]` → MODEL sees **FLIPPED**
- render(): Takes raw camera, does NOT flip (line 232) → VIDEO shows **ORIGINAL**
- **Result**: Different transformations applied → ❌ **MODEL ≠ VIDEO, need transformation!**

### Summary

| Component | Agentview | Wrist |
|-----------|-----------|-------|
| **LiberoProcessor (MODEL)** | Flipped [::-1, ::-1] | Flipped [::-1, ::-1] |
| **MP4 render() (VIDEO)** | Flipped [::-1, ::-1] | NOT flipped |
| **Spaces aligned?** | ✅ Yes | ❌ No |
| **Transformation needed** | ❌ No | ✅ Yes (undo flip) |

**Key takeaway**: Attention maps are in MODEL SPACE. The visualization scripts automatically apply the correct transformations to align with VIDEO SPACE for each camera.

---

## VLM Attention Saving (Task → Image Attention)

In addition to action attention (which shows how action predictions attend to images), Pi0.5 also supports saving **VLM prefix attention** to analyze how the model's task instruction tokens attend to visual features.

### What is VLM Attention?

**VLM (Vision-Language Model) attention** captures the self-attention within the prefix encoding phase:
- **Captured when**: During prefix encoding (before action generation starts)
- **Captured once per**: Each action prediction (1 capture vs 10 for action attention)
- **Contains**: Image↔Image, Image↔Language, Language↔Language, Language↔Image attention
- **Use case**: Understand how task instructions guide visual grounding

**Key differences from action attention:**

| Aspect | Action Attention | VLM Attention |
|--------|-----------------|---------------|
| **When captured** | During denoising (10 steps per action) | During prefix encoding (once per action) |
| **What it shows** | Action tokens → Images | Task tokens ↔ Images |
| **Frequency** | 10× per action | 1× per action |
| **File size** | ~2.8 GB/episode | ~280 MB/episode |
| **Sequence length** | [50 actions, 1018 prefix] | [968 prefix, 968 prefix] |
| **Purpose** | "How does action attend to visual context?" | "How does task command attend to objects?" |

### Understanding VLM Self-Attention

#### Basic Concept

In the VLM (Vision-Language Model), all tokens (images + text) are in a **single unified sequence**:

```
Sequence: [img_patch_0, img_patch_1, ..., img_patch_767, text_token_0, text_token_1, ..., text_token_N]
          └─────────────── 768 image patches ──────────────┘ └──────── language tokens ────────┘
```

Self-attention computes: `attention[i, j]` = **"how much token i (query) attends to token j (key/value)"**

#### What We Visualize

When we extract `attention[text_token_idx, image_patch_idx]`, we answer:

> **"When the model processes this text token (word), which image regions does it look at?"**

This is the **text-to-image attention pattern**, showing visual grounding of language.

#### Attention Matrix Structure

```python
attention_weights.shape = [batch, heads, sequence_len, sequence_len]
                        = [1, 8, 968, 968]

# Where sequence = [768 image patches | 200 language tokens]
# For LIBERO task "pick up the alphabet soup and place it in the basket":
#   - Tokens [0:768]: image patches from 3 cameras (256 each)
#   - Tokens [768:782]: task tokens ("Task:", "pick", "up", "the", "alphabet", "soup", ...)
#   - Tokens [782:~907]: state tokens (robot joint positions, gripper state)
#   - Tokens [~907:968]: action prefix tokens
```

#### Extracting Text→Image Attention

```python
# Shape: [heads, text_tokens, image_patches]
task_to_img = attention[:, task_start:task_end, img_start:img_end]
             = attention[:, 768:782, 0:768]

# For specific token (e.g., token 774 = "soup"):
specific_token_to_img = attention[:, 774, 0:768]  # [heads, 768]
```

#### Index Selection for Specific Tokens

**Question**: "What index should I use for specific tokens?"

**Answer**: Use the **absolute token index** in the full sequence.

From the tokenization output:
```
Token 768: 'Task'       → Use index 768 to see what 'Task' attends to
Token 773: ' alphabet'  → Use index 773 to see what 'alphabet' attends to  
Token 774: ' soup'      → Use index 774 to see what 'soup' attends to
Token 780: ' basket'    → Use index 780 to see what 'basket' attends to
```

#### Common Questions About VLM Attention

**Q1: Why does "Task" token have highest attention?**

**A**: The token "Task:" appears at position 768, right after image patches. Due to positional proximity and attention patterns, it often has strong attention. This is expected and doesn't indicate incorrect behavior.

**Q2: Should I use Q from text and KV from images?**

**A**: No need to think about Q/K/V separately. Self-attention already computes this:
- Text tokens are queries (Q)
- Image patches are keys (K) and values (V)
- `attention[text_idx, img_idx]` gives you the result

**Q3: How to interpret low attention values?**

**A**: Attention values are **relative within each query token**. Focus on:
1. **Spatial patterns**: Where attention concentrates (not absolute values)
2. **Comparisons**: Does "soup" attend to soup can more than other regions?
3. **Rankings**: Which tokens have highest image attention (see demo script)

**Q4: Why [768, 782) instead of [0, 14)?**

**A**: Token indices are **absolute positions** in the full sequence. The task starts at position 768 (after 768 image patches) and spans 14 tokens, so [768, 782)

### How to Enable VLM Attention Saving

Add `--eval.save_vlm_attention_maps=true` to your evaluation command:

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-eval \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.save_vlm_attention_maps=true \  # Enable VLM attention saving
  --policy.compile_model=false \          # REQUIRED! (same as action attention)
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.device=cuda:0 \
  --output_dir=./eval_logs/my_eval
```

**Quick test script:**

Edit `pi_setting/eval/eval_libero_quick_test.sh` and uncomment the VLM attention flag:

```bash
# Change this line:
# --eval.save_vlm_attention_maps=false \

# To:
--eval.save_vlm_attention_maps=true \
```

Then run:
```bash
bash pi_setting/eval/eval_libero_quick_test.sh
```

### VLM Attention Data Structure

VLM attention files are saved as `episode_XXXXX_vlm_attention.pt` with this structure:

```python
{
    'episode_index': 0,
    'batch_index': 0,
    'rollout_steps': [  # One per action in episode (~145 total)
        {
            'rollout_step': 0,
            'vlm_attention': {
                'prefix_len': 968,  # 768 image + 200 language tokens
                'attention_weights': {
                    0: tensor([1, 8, 968, 968]),   # Layer 0: [batch, heads, seq, seq]
                    1: tensor([1, 8, 968, 968]),   # Layer 1
                    ...
                    17: tensor([1, 8, 968, 968]),  # Layer 17 (last layer)
                }
            }
        },
        ...
    ],
    'metadata': {
        'num_rollout_steps': 145
    }
}
```

**Attention dimensions:**
- **Shape**: `[1, 8, 968, 968]`
- **Batch**: 1 (single environment)
- **Heads**: 8 attention heads
- **Sequence**: 968 = 768 (images) + 200 (language)
  - **Image tokens**: indices 0-767 (256 patches × 3 cameras)
  - **Language tokens**: indices 768-967 (task + state + action prefix)

### Token Boundaries

To analyze specific attention patterns, you need to know which tokens correspond to which content:

```python
from pi_setting.eval.token_boundary_helper import TokenBoundaryHelper

# Example task text
task_text = "pick up the alphabet soup and place it in the basket"
full_text = f"Task: {task_text}, State: <state_tokens>;\nAction: "

# Find boundaries
helper = TokenBoundaryHelper()
boundaries = helper.find_token_boundaries(full_text, num_img_tokens=768)

# Output:
# {
#     'images': (0, 768),      # 256 patches × 3 cameras
#     'task': (768, X),        # Task instruction (natural language)
#     'state': (X, Y),         # Robot state (discretized)
#     'action_prefix': (Y, 968) # "\nAction: "
# }
```

**Create masks for analysis:**

```python
masks = helper.create_token_masks(boundaries)

# Extract task→image attention
task_start, task_end = boundaries['task']
img_start, img_end = boundaries['images']

# attention shape: [1, 8, 968, 968]
task_to_img_attention = attention[0, :, task_start:task_end, img_start:img_end]
# Result: [8 heads, N task tokens, 768 image patches]
```

### Visualizing VLM Attention

Use `visualize_vlm_attention.py` to create heatmap overlays showing task→image attention.

#### Usage Examples

**Example 1: Aggregate All Task Tokens (Default)**

Shows overall task→image attention (averaged across all words):

```bash
python pi_setting/eval/visualize_vlm_attention.py \
    --attention_file eval_logs/.../episode_00000_vlm_attention.pt \
    --video_file eval_logs/.../eval_episode_00000.mp4 \
    --output_dir viz_output \
    --task_name libero_object --task_id 0 \
    --rollout_steps 10 20 30
```

**Output**: `timestep_010_agentview.png`, `timestep_010_wrist.png`

**Interpretation**: "On average, which image regions do all task words look at?"

**Example 2: Specific Token Attention**

Shows attention from a single word:

```bash
python pi_setting/eval/visualize_vlm_attention.py \
    --attention_file eval_logs/.../episode_00000_vlm_attention.pt \
    --video_file eval_logs/.../eval_episode_00000.mp4 \
    --output_dir viz_output \
    --task_name libero_object --task_id 0 \
    --rollout_steps 10 \
    --specific_token_idx 774  # "soup" token
```

**Output**: `timestep_010_agentview_token774.png`, `timestep_010_wrist_token774.png`

**Interpretation**: "When the model reads 'soup', which image regions does it look at?"

**Example 3: Compare Multiple Tokens**

To understand object grounding, visualize multiple object-related tokens:

```bash
# Token 773: "alphabet"
python ... --specific_token_idx 773 --output_dir viz_alphabet

# Token 774: "soup"  
python ... --specific_token_idx 774 --output_dir viz_soup

# Token 780: "basket"
python ... --specific_token_idx 780 --output_dir viz_basket
```

**Expected Behavior**:
- Token 773 ("alphabet") should attend to alphabet soup can
- Token 774 ("soup") should attend to alphabet soup can  
- Token 780 ("basket") should attend to basket/container

#### Full Command Line Options

```bash
python pi_setting/eval/visualize_vlm_attention.py \
    --attention_file eval_logs/quick_test/vlm_attention/libero_object_0/episode_00000_vlm_attention.pt \
    --video_file eval_logs/quick_test/videos/libero_object_0/eval_episode_00000.mp4 \
    --output_dir eval_logs/quick_test/vlm_attention/libero_object_0/viz_episode_00000 \
    --rollout_steps 0 10 20 30 \
    --layer 17 \
    --task_name libero_object \
    --task_id 0 \
    --head_aggregation mean \
    --token_aggregation mean \
    --alpha 0.5 \
    --specific_token_idx 774  # Optional: visualize specific token
```

**Parameters:**
- `--task_name`, `--task_id`: Auto-detect task text from LIBERO (recommended)
- `--task_text`: Manual task text (alternative to auto-detection)
- `--specific_token_idx`: Visualize attention from a specific token (e.g., 774 for "soup")
  - When set, `--token_aggregation` is ignored
- `--head_aggregation`: How to combine attention heads (`mean`/`max`/`sum`)
- `--token_aggregation`: How to combine task tokens (`mean`/`max`/`sum`)
  - `mean`: Average attention across all task tokens (default)
  - `max`: Take maximum attention for each image patch
  - `sum`: Sum attention (useful for total attention magnitude)
- `--layer`: Which transformer layer (0-17, default: 17 = last)
- `--alpha`: Overlay transparency (0=invisible, 1=opaque)

**Batch processing:**

Process multiple episodes at once using the batch runner:

```bash
bash pi_setting/eval/run_visualize_vlm_attention.sh
```

Edit the script to configure:
- `EVAL_FOLDER`: Which evaluation run to visualize
- `TIMESTEPS`: Which rollout steps to visualize
- `SPECIFIC_TOKEN_IDX`: Specific token to visualize (leave empty for aggregated view)
- `TOKEN_AGG`: Aggregation strategy (mean/max/sum) - ignored if SPECIFIC_TOKEN_IDX is set

**Output structure:**

```
eval_logs/quick_test/vlm_attention/libero_object_0/
├── episode_00000_vlm_attention.pt          # Raw VLM attention data
├── viz_episode_00000/                       # Visualizations
│   ├── timestep_000_agentview.png          # Side-by-side: overlay | heatmap
│   ├── timestep_000_wrist.png              # Side-by-side: overlay | heatmap
│   ├── timestep_010_agentview_token774.png # Specific token visualization
│   └── ...
└── episode_00001_vlm_attention.pt
```

**Visualization format:**

Each PNG file contains:
```
┌─────────────────────────────────────────────────┐
│         Camera Name | Timestep N                │  ← White title bar
├────────────────────┬────────────────────────────┤
│                    │                            │
│  Video + Overlay   │    Pure Attention Map      │  ← Side-by-side
│  (with alpha=0.5)  │    (no video background)   │
│                    │                            │
└────────────────────┴────────────────────────────┘
```

### Analysis Workflow for VLM Attention

#### Step 1: Identify Important Tokens

Use the demo script to find which tokens have highest attention to images:

**Quick start with batch script:**

```bash
# Edit configuration in the script
bash pi_setting/eval/run_demo_specific_token_attention.sh
```

Edit `run_demo_specific_token_attention.sh` to configure:
- `EVAL_FOLDER`: Which evaluation run to analyze
- `TASK_NAME`, `TASK_ID`: Which task to analyze
- `EPISODE_ID`: Which episode to analyze
- `ROLLOUT_STEP`: Which timestep to analyze
- `TOKENS_OF_INTEREST`: Keywords to highlight (e.g., "alphabet soup basket")

**Or run directly:**

```bash
python demo_specific_token_attention.py \
    --attention_file eval_logs/.../episode_00000_vlm_attention.pt \
    --rollout_step 10 \
    --task_name libero_object --task_id 0 \
    --tokens_of_interest alphabet soup basket
```

**Command line options:**
- `--attention_file`: Path to VLM attention .pt file (required)
- `--rollout_step`: Which timestep to analyze (default: 0)
- `--task_name`: LIBERO task suite name (required)
- `--task_id`: LIBERO task ID (required)
- `--tokens_of_interest`: Space-separated keywords to highlight
- `--layer`: Transformer layer to analyze (default: 17)

**Output shows:**
- All task tokens with indices (e.g., Token 773: ' alphabet')
- Which tokens have highest image attention (ranked)
- Highlighted tokens matching your keywords (★ marker)

Example output:
```
Task: "pick up the alphabet soup and place it in the basket"

ALL TASK TOKENS:
  Token 768: 'Task           ' (chars   0-  4)
  Token 769: ':              ' (chars   4-  5)
  Token 770: ' pick          ' (chars   5- 10)
  Token 771: ' up            ' (chars  10- 13)
  Token 772: ' the           ' (chars  13- 17)
  Token 773: ' alphabet      ' (chars  17- 26) ★
  Token 774: ' soup          ' (chars  26- 31) ★
  Token 775: ' and           ' (chars  31- 35)
  Token 776: ' place         ' (chars  35- 41)
  Token 777: ' it            ' (chars  41- 44)
  Token 778: ' in            ' (chars  44- 47)
  Token 779: ' the           ' (chars  47- 51)
  Token 780: ' basket        ' (chars  51- 58) ★
  Token 781: ',              ' (chars  58- 59)

Top 10 tokens by average attention to images:
   1. Token 768: 'Task           ' - attention=0.001183
   2. Token 774: ' soup          ' - attention=0.000488
   3. Token 773: ' alphabet      ' - attention=0.000096
   ...

TOKENS MATCHING YOUR KEYWORDS: ['alphabet', 'soup', 'basket']
  Token 773: ' alphabet      ' - attention to images: 0.000096
  Token 774: ' soup          ' - attention to images: 0.000488
  Token 780: ' basket        ' - attention to images: 0.000058
```

**Interpretation:**
- Token 774 ("soup") has highest attention among object words
- Token 768 ("Task") has highest overall (due to positional effects)
- Use these indices for Step 2 visualization

#### Step 2: Visualize Specific Tokens

Based on Step 1 output, visualize interesting tokens:

```bash
# In run_visualize_vlm_attention.sh:
SPECIFIC_TOKEN_IDX=774  # "soup" token

./pi_setting/eval/run_visualize_vlm_attention.sh
```

Or directly:
```bash
python pi_setting/eval/visualize_vlm_attention.py \
    --attention_file eval_logs/.../episode_00000_vlm_attention.pt \
    --video_file eval_logs/.../eval_episode_00000.mp4 \
    --output_dir viz_soup \
    --task_name libero_object --task_id 0 \
    --rollout_steps 10 20 30 \
    --specific_token_idx 774
```

#### Step 3: Analyze Results

Compare visualizations:
1. Does object word attend to correct object?
2. Does action word ("pick", "place") attend to relevant objects?
3. Does spatial word ("in") attend to container/target location?

This reveals whether the VLM correctly grounds language in visual perception.

### Coordinate Transformations

VLM attention maps follow the same coordinate system as action attention:

| Camera | Model Space | Video Space | Transformation Needed |
|--------|-------------|-------------|----------------------|
| **Agentview** | Flipped | Flipped | ✅ No (aligned) |
| **Wrist** | Flipped | Not flipped | ✅ Yes (undo flip) |

The visualization scripts automatically handle these transformations by setting `apply_flip=True` for wrist camera.

### Example Analysis Workflow

**1. Save VLM attention during evaluation:**
```bash
bash pi_setting/eval/eval_libero_quick_test.sh  # With save_vlm_attention_maps=true
```

**2. Inspect saved data:**
```python
import torch

data = torch.load("eval_logs/quick_test/vlm_attention/libero_object_0/episode_00000_vlm_attention.pt")

print(f"Episode has {len(data['rollout_steps'])} rollout steps")
print(f"Prefix length: {data['rollout_steps'][0]['vlm_attention']['prefix_len']}")
print(f"Layers saved: {list(data['rollout_steps'][0]['vlm_attention']['attention_weights'].keys())}")

# Check attention shape
att = data['rollout_steps'][0]['vlm_attention']['attention_weights'][17]
print(f"Attention shape: {att.shape}")  # [1, 8, 968, 968]
```

**3. Visualize task→image attention:**
```bash
# Single episode
python pi_setting/eval/visualize_vlm_attention.py \
    --attention_file eval_logs/quick_test/vlm_attention/libero_object_0/episode_00000_vlm_attention.pt \
    --video_file eval_logs/quick_test/videos/libero_object_0/eval_episode_00000.mp4 \
    --output_dir eval_logs/quick_test/vlm_attention/libero_object_0/viz_episode_00000 \
    --rollout_steps 0 20 40 60 \
    --task_text "pick up the alphabet soup and place it in the basket"

# Or batch process
bash pi_setting/eval/run_visualize_vlm_attention.sh
```

**4. Analyze attention patterns:**
```python
from pi_setting.eval.token_boundary_helper import TokenBoundaryHelper

# Load attention
data = torch.load("episode_00000_vlm_attention.pt")
attention = data['rollout_steps'][0]['vlm_attention']['attention_weights'][17]  # [1, 8, 968, 968]

# Find token boundaries
task_text = "pick up the alphabet soup and place it in the basket"
helper = TokenBoundaryHelper()
boundaries = helper.find_token_boundaries(f"Task: {task_text}, State: ...", num_img_tokens=768)

# Extract task→image attention
task_start, task_end = boundaries['task']
task_to_img = attention[0, :, task_start:task_end, :768]  # [8 heads, N task tokens, 768 images]

# Average across heads and tokens
task_to_img_mean = task_to_img.mean(dim=0).mean(dim=0)  # [768]

# Find which image patches have highest attention
top_k = 10
top_patches = task_to_img_mean.topk(top_k)
print(f"Top {top_k} attended patches: {top_patches.indices}")
print(f"Attention values: {top_patches.values}")
```

### File Size Comparison

VLM attention files are **10× smaller** than action attention files:

```
episode_00000_attention.pt          ~2.8 GB  (10 denoising steps per action)
episode_00000_vlm_attention.pt      ~280 MB  (1 prefix encoding per action)
```

This is because:
- VLM attention captured once per action (vs 10× for denoising)
- But saves all 18 layers (vs 2 layers for action attention)

**Storage requirements:**
- 10 episodes with action attention: ~28 GB
- 10 episodes with VLM attention: ~2.8 GB
- 10 episodes with both: ~31 GB

### Creating Attention Grid Visualizations

After generating individual attention visualizations with `run_visualize_vlm_attention.sh`, you can combine them into comprehensive grid layouts to analyze how attention patterns evolve across transformer layers and attention heads.

#### What is an Attention Grid?

An **attention grid** is a single image showing multiple attention visualizations arranged in a matrix:
- **Rows**: Different transformer layers (0-17, showing attention evolution through network depth)
- **Columns**: Different attention heads (0-7, showing head specialization)
- **Each cell**: Attention overlay for one layer-head combination

This enables systematic analysis:
- **Layer evolution**: How attention patterns change from early (layer 0) to late (layer 17) layers
- **Head specialization**: Whether different heads focus on different aspects (objects, spatial relations, etc.)
- **Comprehensive view**: All combinations in one image for easy comparison

#### Grid Layout Structure

```
                    Head 0      Head 1      Head 2  ...  Head 7
        Layer 0     [image]     [image]     [image] ...  [image]
        Layer 1     [image]     [image]     [image] ...  [image]
        Layer 2     [image]     [image]     [image] ...  [image]
        ...
        Layer 17    [image]     [image]     [image] ...  [image]
```

**Features:**
- Title bar at top with metadata (timestep, camera, token, layer/head ranges)
- Layer labels on left margin ("Layer 0", "Layer 1", ...)
- Head labels on top margin ("Head 0", "Head 1", ...)
- Small padding between subplots (5 pixels) for visual separation
- Each subplot shows only the attention overlay (no individual title bars)

#### Quick Start

**Step 1: Generate individual visualizations**

First, create individual attention maps using `run_visualize_vlm_attention.sh`:

```bash
# Configure to save all layers and heads
# Edit run_visualize_vlm_attention.sh:
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)
SPECIFIC_HEAD_IDXS=(0 1 2 3 4 5 6 7)
SPECIFIC_TOKEN_IDXS=(774)  # "soup" token

# Run visualization
bash pi_setting/eval/run_visualize_vlm_attention.sh
```

This creates individual PNGs like:
- `timestep_000_agentview_layer0_token774_head0.png`
- `timestep_000_agentview_layer0_token774_head1.png`
- ... (18 layers × 8 heads = 144 images per timestep/camera/token)

**Step 2: Create grid visualization**

```bash
# Run the grid creation script
bash pi_setting/eval/create_attention_grid.sh
```

This combines all individual images into grids like:
- `timestep_000_agentview_token774_layer0to17_head0to7.png` (8×18 grid)
- `timestep_010_agentview_token774_layer0to17_head0to7.png`
- `timestep_020_wrist_token774_layer0to17_head0to7.png`

#### Configuration

Edit `create_attention_grid.sh` to customize:

```bash
# Source data
EVAL_FOLDER="quick_test_vlm_attention_1"
TASK_NAME="libero_object_0"
EPISODE_NUM="00000"

# Which visualizations to combine
TIMESTEPS=(0 10 20)                    # Which rollout steps
CAMERAS=("agentview" "wrist")          # Which cameras
SPECIFIC_TOKEN_IDXS=(770 773 774 775 776 780)  # Which tokens

# Grid configuration
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)  # All layers
HEADS=(0 1 2 3 4 5 6 7)                                # All heads

# Visual settings
PADDING=5           # Pixels between subplots
LABEL_FONT_SIZE=12  # Font size for head/layer labels
TITLE_FONT_SIZE=16  # Font size for title bar
```

#### Output

**File naming:**
```
timestep_{step:03d}_{camera}_token{token}_layer{min}to{max}_head{min}to{max}.png
```

**Examples:**
- `timestep_000_agentview_token774_layer0to17_head0to7.png` - Full 8×18 grid
- `timestep_010_wrist_token780_layer0to17_head0to7.png` - Wrist camera, "basket" token

**Output location:**
```
eval_logs/{EVAL_FOLDER}/vlm_attention/{TASK_NAME}/viz_episode_{N}/grids/
├── timestep_000_agentview_token774_layer0to17_head0to7.png
├── timestep_010_agentview_token774_layer0to17_head0to7.png
├── timestep_020_agentview_token774_layer0to17_head0to7.png
├── timestep_000_wrist_token774_layer0to17_head0to7.png
└── ...
```

#### Image Dimensions

**For full 8×18 grid:**
- Each subplot: ~256×256 pixels (attention overlay only, title bar removed)
- Grid width: 80 (left margin) + 8 × 256 + 7 × 5 (padding) ≈ **2133 pixels**
- Grid height: 130 (top margin + title) + 18 × 256 + 17 × 5 (padding) ≈ **4823 pixels**

**Total**: ~2133 × 4823 pixels (~10 MB PNG file)

This is large but allows zooming in to see details while maintaining overview.

#### Advanced Usage

**Create partial grids (faster, smaller files):**

```bash
# Edit create_attention_grid.sh:
LAYERS=(0 5 10 15 17)  # Sample 5 layers instead of all 18
HEADS=(0 2 4 6)        # Sample 4 heads instead of all 8

# Creates 4×5 grid instead of 8×18
```

**Direct Python invocation:**

```bash
python pi_setting/eval/create_attention_grid.py \
    --input_dir eval_logs/.../viz_episode_00000 \
    --output_dir eval_logs/.../viz_episode_00000/grids \
    --timestep 10 \
    --camera agentview \
    --token_idx 774 \
    --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17" \
    --heads "0,1,2,3,4,5,6,7" \
    --padding 5 \
    --label_font_size 12 \
    --title_font_size 16
```

#### Analysis Workflow

**Complete workflow for attention analysis:**

```bash
# Step 1: Save VLM attention during evaluation
bash pi_setting/eval/eval_libero_quick_test.sh  # With save_vlm_attention_maps=true

# Step 2: Identify interesting tokens
bash pi_setting/eval/run_demo_specific_token_attention.sh

# Step 3: Generate individual visualizations for all layers/heads
# Edit run_visualize_vlm_attention.sh:
#   LAYERS=(0 1 2 ... 17)  # All layers
#   SPECIFIC_HEAD_IDXS=(0 1 ... 7)  # All heads
#   SPECIFIC_TOKEN_IDXS=(774)  # From Step 2
bash pi_setting/eval/run_visualize_vlm_attention.sh

# Step 4: Create grid visualizations
bash pi_setting/eval/create_attention_grid.sh

# Step 5: Analyze results
# Open grids and look for patterns:
# - Which layers show object-specific attention?
# - Do different heads specialize?
# - How does attention evolve across layers?
```

#### What to Look For

**Layer progression (vertical):**
- **Early layers (0-5)**: Low-level features, edges, textures
- **Middle layers (6-12)**: Object parts, spatial relationships
- **Late layers (13-17)**: Semantic understanding, task-relevant objects

**Head specialization (horizontal):**
- Do some heads focus on specific objects?
- Do some heads attend to spatial relationships?
- Do some heads show broader context attention?

**Consistency:**
- Does attention to task-relevant objects increase in later layers?
- Are certain heads consistently more selective?
- Does attention match expected object locations?

#### Example Interpretation

For task "pick up the alphabet soup and place it in the basket" with token 774 ("soup"):

**Expected patterns:**
- **Layer 0-3**: Distributed attention (no clear object focus)
- **Layer 4-8**: Attention starts concentrating on soup can
- **Layer 9-17**: Strong, focused attention on soup can
- **Some heads**: Sharp focus on soup can (object-specific)
- **Other heads**: Broader attention including basket (context)

**What this reveals:**
- The model progressively refines understanding through layers
- Different heads capture different aspects (object vs. context)
- Late layers show clear task-relevant object grounding

#### Troubleshooting

**Q: "Missing" gray boxes in grid**

**A:** Some individual visualizations weren't created. Check:
1. Did `run_visualize_vlm_attention.sh` complete successfully?
2. Are all layer/head combinations in `LAYERS` and `SPECIFIC_HEAD_IDXS`?
3. Check input directory for missing PNG files

**Q: Subplots still have title bars**

**A:** The extraction isn't working correctly. This was fixed by adjusting the white detection threshold from 240 to 200. Make sure you have the latest version of `create_attention_grid.py`.

**Q: Grid image too large to view**

**A:** Options:
1. Create partial grids with fewer layers/heads
2. Use image viewer with zoom (e.g., `eog`, `feh`)
3. View on high-resolution display
4. Create separate grids for layer groups (0-8, 9-17)

**Q: File not found errors**

**A:** Make sure:
1. Input directory path is correct in `create_attention_grid.sh`
2. Individual visualizations exist (check with `ls`)
3. File naming matches expected pattern (layer/token/head indices)

### Troubleshooting VLM Attention

**Q: "VLM attention saving was enabled but no VLM attention maps were collected"**

**A:** This was a bug in early versions. Make sure you have the latest code where `PaliGemmaWithExpertModel.forward()` passes `output_attentions=True` in the prefix-only branch (around line 365).

**Q: Attention maps look random/noisy**

**A:** Try different aggregation strategies:
- Use `--token_aggregation=max` to highlight strongest attention
- Try different layers: `--layer 10` (middle) or `--layer 0` (early)
- Check specific attention heads instead of averaging

**Q: Heatmaps don't align with video**

**A:** Make sure:
- Wrist camera transformations are correct (`apply_flip=True`)
- Rollout step matches video frame index
- Using the correct evaluation run (attention + video from same run)

**Q: Error "Token index 774 out of task token range [768, 772)"**

**Cause**: Task text mismatch. The code is using wrong/incomplete task text.

**Solution**: Always use `--task_name` and `--task_id` for auto-detection:
```bash
--task_name libero_object --task_id 0
```
Instead of manually providing `--task_text`.

**Q: Error "Must provide either --task_text OR both --task_name and --task_id"**

**Solution**: Choose one:
- Option A: `--task_text "pick up the alphabet soup and place it in the basket"`
- Option B: `--task_name libero_object --task_id 0` (recommended - auto-detects from LIBERO)

**Q: AttributeError: 'NoneType' object has no attribute 'shape'**

**A:** The model's attention implementation must be set to "eager" mode (not "sdpa"). This is automatically handled during evaluation, but for debugging:

```python
model.paligemma.language_model.config._attn_implementation = "eager"
```

