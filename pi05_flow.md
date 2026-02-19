# PI-0.5 VLA Data Flow Documentation

This document traces the complete data flow through the PI-0.5 Vision-Language-Action (VLA) policy, from raw environment observations to predicted actions.

---

## Overview

PI-0.5 processes **three modalities**:
1. **Vision**: Camera images (224×224 RGB)
2. **Language**: Task instruction + discretized robot state
3. **Action**: Robot control commands

The model uses **PaliGemma** (SigLIP vision encoder + Gemma-2B language model) + a 300M action expert.

---

## Pipeline Steps

### Step 1: Environment Observation
**File**: `lerobot/scripts/lerobot_eval.py` (line ~254 in `rollout()`)

**Goal**: Get raw observation from simulation/real environment

**Code Location**:
```python
observation, reward, terminated, truncated, info = env.step(action)
```

**Example Output**:
```python
observation = {
    'observation.images.image': torch.Tensor,  # Shape: [1, 3, 256, 256], dtype=uint8, range=[0, 255] (1st camera)
    'observation.images.image2': torch.Tensor,  # Shape: [1, 3, 256, 256], dtype=uint8, range=[0, 255] (2nd camera)
    'observation.robot_state.eef': np.ndarray,    # mat [1,3,3], pos [1,3], quat [1,4]
    'observation.robot_state.gripper': np.ndarray,    # qpos [1,2], qvel [1,2]
    'observation.robot_state.joint': np.ndarray,    # pos [1,7], vel [1,7]
    # ... other sensor data
}
```

**Key Details**:
- Images are raw camera frames (256×256 RGB), for pi05 libero, there are 2 cameras and 1 pad camera
- Robot state contains continuous values (joint angles, gripper position, etc.)

---

### Step 2: Add Task Instruction
**File**: `lerobot/envs/base.py` → `add_envs_task_to_observation()`

**Goal**: Inject natural language task instruction into observation

**Example Output**:
```python
observation['task'] = "pick up the alphabet soup and place it in the basket"
```

**Shape**: String (variable length, ~10-100 characters)

---

### Step 3: Transform Robot State (Environment Preprocessor)
**File**: `lerobot/scripts/lerobot_eval.py` → `observation = env_preprocessor(observation)`  
**Processor**: `lerobot/processor/env_processor.py` → `LiberoProcessorStep`

**Goal**: Convert complex nested LIBERO robot state into a simplified 8-dimensional state vector suitable for the policy

**What it does**:
1. **Flatten robot state** from nested dictionaries into a single vector:
   - End-effector position (3D): `robot_state.eef.pos` → [x, y, z]
   - End-effector orientation (3D): `robot_state.eef.quat` (4D quaternion) → converted to axis-angle (3D)
   - Gripper position (2D): `robot_state.gripper.qpos` → [gripper_joint_1, gripper_joint_2]
   - **Total: 3 + 3 + 2 = 8 dimensions**

2. **Rotate images** by 180° (flip H and W dimensions) to account for LIBERO camera orientation convention

**Example Input**:
```python
observation = {
    'observation.images.image': torch.Tensor,  # Shape: [1, 3, 256, 256], dtype=uint8, range=[0, 255] (1st camera)
    'observation.images.image2': torch.Tensor,  # Shape: [1, 3, 256, 256], dtype=uint8, range=[0, 255] (2nd camera)
    'observation.robot_state.eef': np.ndarray,    # mat [1,3,3], pos [1,3], quat [1,4]
    'observation.robot_state.gripper': np.ndarray,    # qpos [1,2], qvel [1,2]
    'observation.robot_state.joint': np.ndarray,    # pos [1,7], vel [1,7]
    # ... other sensor data
}
```

**Example Output**:
```python
observation = {
    'observation.images.image': torch.Tensor,   # Shape: [1, 3, 256, 256] (rotated 180°)
    'observation.images.image2': torch.Tensor,  # Shape: [1, 3, 256, 256] (rotated 180°)
    'observation.state': torch.Tensor,          # Shape: [1, 8], dtype=float32
    # robot_state dictionary removed (flattened into observation.state)
}
```

**State Vector Breakdown** (8 dimensions):
```python
observation.state = [
    eef_x, eef_y, eef_z,           # End-effector position (3D)
    eef_axis_x, eef_axis_y, eef_axis_z,  # End-effector orientation as axis-angle (3D)
    gripper_joint_1, gripper_joint_2     # Gripper joint positions (2D)
]
# Example values: [0.45, -0.12, 0.35, 0.1, -0.05, 0.8, 0.02, 0.02]
```

**Key Details**:
- **Why 8D?** The policy only needs the essential control-relevant state: where the end-effector is (position), how it's oriented (axis-angle), and gripper state
- **Quaternion → Axis-angle conversion**: LIBERO provides orientation as quaternion (4D: x,y,z,w), but it's converted to axis-angle (3D) which is more compact and avoids quaternion redundancy
- **Simplification**: Removes less critical state info like joint velocities, end-effector velocity, etc., keeping only what's needed for control

---

### Step 4: Robot State Discretization 
**File**: `observation = preprocessor(observation)` →  `lerobot/policies/pi05/processor_pi05.py` → `Pi05PrepareStateTokenizerProcessorStep`

**Goal**: Convert continuous robot state to discrete bins and format as text template

**Code Location**:
```python
class Pi05PrepareStateTokenizerProcessorStep:
    def observation(self, observation):
        # Discretize state: continuous → 256 bins
        state_discretized = np.digitize(state, bins) - 1  # Range: [0, 255]
        
        # Format template
        task_text = f"Task: {instruction}, State: {state_str};\nAction: "
```

**Example Input**:
```python
observation['observation.state'] = [0.123, -0.456, 0.789, ...]  # 8 floats
```

**Example Output**:
```python
observation['task'] = [
    'Task: pick up the alphabet soup and place it in the basket, '
    'State: -1 93 -1 190 158 141 234 23 128 128 128 128 128 128 128 128 '
    '128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128;\n'
    'Action: '
]
```

**Shape**: List of strings, length = batch_size (usually 1)

**Key Details**:
- State discretized into 256 bins (0-255, plus -1 for invalid/masked values)
- Maximum 32 state dimensions (padded with 128 if fewer)
- Numbers like "128" appear frequently as padding values
- Template format: `"Task: {instruction}, State: {numbers};\nAction: "`

---

### Step 5: Tokenization on full task (language, robot state)
**File**: `observation = preprocessor(observation)` →  `lerobot/common/processors/tokenizer_processor.py` → `TokenizerProcessorStep`

**Goal**: Convert text to token IDs using PaliGemma tokenizer

**Code Location**:
```python
class TokenizerProcessorStep:
    def _tokenize_text(self, text):
        encoding = self.tokenizer(
            text,
            max_length=200,
            padding='max_length',
            return_tensors='pt'
        )
```

**Example Input**:
```python
text = 'Task: pick up the alphabet soup and place it in the basket, State: -1 93 -1 190 158 141 234 23 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128;\nAction: '
```

**Example Output**:
```python
observation[OBS_LANGUAGE_TOKENS] = torch.Tensor([
    2, 2366, 235292, 728, 235274, 235248, 235315, 235304, ...  # 144 real tokens
    0, 0, 0, 0, 0, 0, ...  # 56 padding tokens
])
observation[OBS_LANGUAGE_ATTENTION_MASK] = torch.Tensor([
    True, True, True, ...,   # 144 True values
    False, False, False, ... # 56 False values
])
```

**Shape**: 
- `tokens`: [batch_size, 200] = [1, 200]
- `attention_mask`: [batch_size, 200] = [1, 200]

**Important Token Breakdown** (144 real tokens):
```
Token   0: <bos>                                # 1 token (beginning of sequence)
Tokens  1-14: "Task: ... basket, State: "       # ~14 tokens (task instruction)
Tokens 15-139: "... 128 128 128;"               # ~125 tokens (discretized state)
Tokens 140-143: "\nAction: "                    # ~4 tokens (action prompt)
Tokens 144-199: <pad>                           # 56 padding tokens
```

**Why 32 numbers ≠ 32 tokens?**
- Each digit is tokenized separately!
- "128" = 4 tokens: `'▁'` (space) + `'1'` + `'2'` + `'8'`
- "-1" = 2 tokens: `'-'` + `'1'`
- Result: 32 numbers → ~125 tokens

**Decoded Output**:
```python
tokenizer.decode(tokens[0]) = 
    "<bos>Task: pick up the alphabet soup and place it in the basket, "
    "State: -1 93 -1 190 158 141 234 23 128 128 128 128 128 128 128 128 "
    "128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128;\n"
    "Action: <pad><pad><pad>...<pad>"
```

---

### Step 6: Image Preprocessing (Inside Policy)
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PI05Policy._preprocess_images()`

**Goal**: Resize and normalize images for SigLIP vision encoder

**Code Location**:
```python
def _preprocess_images(self, batch):
    # Resize 256×256 → 224×224 with aspect ratio preserved (to fit the pi0.5 model input size)
    img = resize_with_pad_torch(img, 224, 224)
    
    # Normalize [0, 1] → [-1, 1] for SigLIP
    img = img * 2.0 - 1.0
```

**Example Input**:
```python
batch['observation.image']: [1, 3, 256, 256], dtype=float32, range=[0, 1]
```

**Example Output**:
```python
images = [
    torch.Tensor,  # Shape: [1, 3, 224, 224], dtype=float32, range=[-1, 1]
]
img_masks = [
    torch.Tensor   # Shape: [1], dtype=bool, all True (real image)
]
```

**Key Details**:
- `resize_with_pad_torch()`: Maintains aspect ratio, pads with -1 (black)
- Final size: 224×224 (required by SigLIP)
- Normalization: [-1, 1] range for SigLIP
- Images stay in **pixel space** (not tokenized!)

---

### Step 7: Embedding (Multimodal Fusion)
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PI05Pytorch.embed_prefix()`

**Goal**: Convert images and text tokens to embedding vectors and concatenate

**Code Location**:
```python
def embed_prefix(self, images, img_masks, tokens, masks):
    # Process images through SigLIP vision encoder
    for img in images:
        img_emb = self.paligemma_with_expert.embed_image(img)  # Vision encoder
        embs.append(img_emb)  # Shape: [B, 256, 2048]
    
    # Process language tokens through embedding layer
    lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])  # Scale
    embs.append(lang_emb)  # Shape: [B, 200, 2048]
    
    # Concatenate in embedding space
    embs = torch.cat(embs, dim=1)  # Shape: [B, 256*3+200, 2048], where 3 is 2 valid camera + 1 empty camera
```

**Example Output**:
```python
prefix_embs:      [1, 968, 2048]  # 256*3 image tokens + 200 text tokens
prefix_pad_masks: [1, 968]        # True for real tokens, False for padding
prefix_att_masks: [1, 968]        # 0 for all (bidirectional attention)
```

**Key Details**:
- **Images**: 224×224 image → SigLIP → 256 image embedding tokens
- **Text**: 200 token IDs → embedding layer → 200 text embedding vectors
- **Fusion**: Concatenated in sequence dimension (not summed!)
- Embedding dimension: 2048 (PaliGemma hidden size)
- At this point, images and text are in the same embedding space

---

#### Understanding Embedding Dimensions

**Why these specific dimensions?**

1. **256 (number of image tokens)**
   - **NOT** the image size (224×224)!
   - SigLIP vision encoder divides the 224×224 image into **patches**
   - Typical patch size: 14×14 pixels (configurable)
   - Number of patches: (224÷14) × (224÷14) = **16 × 16 = 256 patches**
   - Each patch becomes one embedding token
   - **Flow**: `(1, 3, 224, 224)` pixels → SigLIP processes 256 patches → `(1, 256, 2048)` embeddings

2. **200 (max sequence length for language tokens)**
   - Maximum allowed token sequence length set in the tokenizer config
   - Actual usage may be less (e.g., 144 real tokens in our example)
   - Remaining positions padded with `<pad>` tokens
   - **Why 200?** Balance between:
     - Long enough for task description + discretized state (144 tokens)
     - Short enough for memory efficiency

3. **2048 (embedding/hidden dimension)**
   - The **hidden size** of the Gemma-2B language model
   - Set in config: `vlm_config_hf.vision_config.projection_dim = 2048`
   - **Critical requirement**: All modalities must share the same embedding dimension
   - SigLIP vision encoder projects patches to 2048-D
   - Gemma embedding layer maps token IDs to 2048-D
   - This allows multimodal fusion via simple concatenation

**Summary:**
```
Images:  (1, 3, 224, 224) → [SigLIP: 256 patches] → (1, 256, 2048)
                             Each patch = 14×14 pixels = 1 token

Tokens:  (1, 200) → [Embedding layer] → (1, 200, 2048)
                     Token ID → 2048-D learned vector

Result:  (1, 256×3 + 200, 2048) = (1, 968, 2048)
         768 image tokens (3 cameras) + 200 text tokens
```

---

#### Understanding Masks

**prefix_pad_masks: [1, 968]**
- **Type**: Boolean tensor
- **Purpose**: Distinguishes **real tokens** (True) from **padding** (False)
- **Construction**:
  ```python
  # For each camera image (256 tokens each)
  pad_masks.append(img_mask[:, None].expand(bsize, 256))  # e.g., [True]*256
  
  # For language tokens (200 tokens)
  pad_masks.append(masks)  # e.g., [True]*144 + [False]*56
  
  # Concatenate all
  pad_masks = torch.cat(pad_masks, dim=1)  # Shape: [1, 968]
  ```
- **Example values**:
  ```
  Position   0-255: True  (camera 1 - real image)
  Position 256-511: True  (camera 2 - real image)
  Position 512-767: False (camera 3 - empty/padded)
  Position 768-911: True  (real language tokens: <bos> + task + state + action prompt)
  Position 912-967: False (padding tokens)
  ```
- **Usage**: Prevents the model from attending to padding positions

**prefix_att_masks: [1, 968]**
- **Type**: Boolean/integer tensor (also called "autoregressive mask" or "mask_ar")
- **Purpose**: Controls **attention pattern** - bidirectional vs causal
- **Construction**:
  ```python
  # For all image tokens
  att_masks += [0] * 256  # Per camera
  
  # For all language tokens
  att_masks += [0] * 200
  
  # All zeros for prefix!
  att_masks = torch.tensor(att_masks)  # Shape: [1, 968]
  ```
- **Meaning of values**:
  - `0` = **Bidirectional attention** (token can see all other tokens with `0`)
  - `1` = **Attention boundary** (creates causal/autoregressive split)
  
- **Example values**:
  ```
  Position   0-967: 0  (all prefix tokens use bidirectional attention)
  ```
  
- **How it creates attention patterns** (via `make_att_2d_masks()`):
  ```python
  cumsum = torch.cumsum(att_masks, dim=1)
  att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
  ```
  - With all `0`s: cumsum is all `0`s → all positions can attend to each other
  - With `[0,0,1,0,0]`: cumsum is `[0,0,1,1,1]` → positions 3-5 can't attend back to 0-1

**Attention Pattern for Prefix:**
```
Since all prefix_att_masks = 0, we get FULL BIDIRECTIONAL attention:

              Img1(256)  Img2(256)  Img3(256)  Text(200)
Img1(256)         ✓          ✓          ✓          ✓
Img2(256)         ✓          ✓          ✓          ✓
Img3(256)         ✓          ✓          ✓          ✓
Text(200)         ✓          ✓          ✓          ✓

✓ = Can attend (all tokens see all other prefix tokens)
```

**Combined Effect:**
```python
# Final 2D attention mask combines both:
att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

# Result: [batch_size, 968, 968] boolean mask where:
# - True  = can attend (valid token, no attention boundary)
# - False = cannot attend (padding or blocked by causal mask)
```

---

**Structure of Concatenated Sequence**:
```
Position   0-767:   Image embeddings (256 tokens from 224×224 image)
Position 768-967:   Text embeddings (200 tokens, 144 real + 56 padding)
                    ├─ 256: <bos>
                    ├─ 257-270: Task instruction tokens
                    ├─ 271-395: State number tokens  
                    ├─ 396-399: "\nAction: " tokens
                    └─ 400-455: <pad> tokens
```

**How to Identify Token Boundaries** (for attention analysis):
```python
# Use helper class to find where task text vs state starts
from token_boundary_helper import TokenBoundaryTracker
tracker = TokenBoundaryTracker()
boundaries = tracker.get_token_boundaries(batch['task'][0])

# Returns:
# {
#   'task_range': (1, 15),      # Pure task instruction tokens (in text portion)
#   'state_range': (15, 140),   # Discretized state number tokens
#   'action_range': (140, 144), # "\nAction: " tokens
# }

# Adjust for full sequence (after image embeddings):
image_offset = 256
task_tokens_in_full_seq = range(256 + 1, 256 + 15)     # Positions 257-270
state_tokens_in_full_seq = range(256 + 15, 256 + 140)  # Positions 271-395
```

---

### Step 8: Prepare Attention Masks & Position Encoding
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PI05Pytorch.sample_actions()`

**Goal**: Convert 1D masks to 2D/4D attention masks and create position IDs for the transformer

**Code Location**:
```python
def sample_actions(self, images, img_masks, tokens, masks, ...):
    # After embedding prefix
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
    
    # Create 2D attention mask
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    
    # Create position IDs for RoPE
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    
    # Convert to 4D for transformer
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
```

**What happens**:

1. **Create 2D Attention Mask** `[batch, seq_len, seq_len]`
   - Input: `prefix_pad_masks` [1, 968], `prefix_att_masks` [1, 968]
   - Output: `prefix_att_2d_masks` [1, 968, 968]
   - Purpose: Define which tokens can attend to which other tokens

2. **Create Position IDs** `[batch, seq_len]`
   - Input: `prefix_pad_masks` [1, 968]
   - Output: `prefix_position_ids` [1, 968]
   - Purpose: Sequential position for each token (used by RoPE embeddings)

3. **Convert to 4D Float Mask** `[batch, 1, seq_len, seq_len]`
   - Input: `prefix_att_2d_masks` [1, 968, 968]
   - Output: `prefix_att_2d_masks_4d` [1, 1, 968, 968]
   - Purpose: Transformer-ready mask with added head dimension

---

#### Understanding the Masks

**1. prefix_att_2d_masks: [1, 968, 968] - Boolean Attention Matrix**

**Purpose**: Defines the exact attention pattern - which tokens can attend to which

**How it's created** (via `make_att_2d_masks()`):
```python
def make_att_2d_masks(pad_masks, att_masks):
    # Step 1: Cumulative sum creates attention boundaries
    cumsum = torch.cumsum(att_masks, dim=1)
    # For prefix: [0,0,0,...,0] → cumsum = [0,0,0,...,0]
    
    # Step 2: Compare cumsum values to create attention pattern
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    # Since all zeros: 0 <= 0 is always True → bidirectional attention!
    
    # Step 3: Combine with padding mask
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    # Blocks attention to/from padding positions
    
    return att_2d_masks & pad_2d_masks
```

**Example visualization**:
```
For prefix (all att_masks=0, some padding at end):

             Img1  Img2  Text  Pad
Img1 [256]    ✓     ✓     ✓     ✗
Img2 [256]    ✓     ✓     ✓     ✗
Text [200]    ✓     ✓     ✓     ✗
Pad  [56]     ✗     ✗     ✗     ✗

✓ = True (can attend)
✗ = False (cannot attend - padding)

Matrix shape: [1, 968, 968]
Values: mostly True (bidirectional), False only for padding
```

**Why bidirectional for prefix?**
- All `att_masks = 0` → all cumsum values are 0
- `0 <= 0` is always True → every token can attend to every other token
- This is **prefix-LM** style: condition tokens use full bidirectional attention

**Contrast with causal attention**:
```python
# If att_masks = [1, 1, 1, 1, 1] (pure causal)
# cumsum = [1, 2, 3, 4, 5]

#          Token0  Token1  Token2  Token3  Token4
# Token0     T       F       F       F       F     (only attends to self)
# Token1     T       T       F       F       F     (attends to 0,1)
# Token2     T       T       T       F       F     (attends to 0,1,2)
# Token3     T       T       T       T       F     (attends to 0,1,2,3)
# Token4     T       T       T       T       T     (attends to all)
```

---

**2. prefix_att_2d_masks_4d: [1, 1, 968, 968] - Float Mask for Transformer**

**Purpose**: Convert boolean mask to float values that transformers use in attention computation

**How it's created** (via `_prepare_attention_masks_4d()`):
```python
def _prepare_attention_masks_4d(self, att_2d_masks):
    # Add dimension for attention heads (broadcasts to all heads)
    att_2d_masks_4d = att_2d_masks[:, None, :, :]  # [1, 968, 968] → [1, 1, 968, 968]
    
    # Convert boolean to float:
    # True → 0.0 (can attend, no penalty)
    # False → -2.3819763e38 (cannot attend, huge negative number)
    return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
```

**Why these specific values?**
- `0.0`: No modification to attention scores
- `-2.38e38`: Very large negative number (close to float32 minimum)

**How it's used in transformer attention**:
```python
# Inside transformer layer:
attention_scores = (Q @ K.T) / sqrt(d_k)  # Raw scores, e.g., [-5.2, 3.1, -2.4, 8.7]
attention_scores += att_2d_masks_4d        # Add mask: [0, 0, -2e38, 0]
                                            # Result: [-5.2, 3.1, -inf, 8.7]
attention_weights = softmax(attention_scores)  # exp(-inf) ≈ 0.0
output = attention_weights @ V              # Masked positions contribute ~0
```

**Why add a dimension for heads?**
- Shape `[batch, 1, seq_len, seq_len]` broadcasts to `[batch, num_heads, seq_len, seq_len]`
- All 8 attention heads use the same mask pattern
- Efficient: store mask once, broadcast to all heads

---

**3. prefix_position_ids: [1, 968] - Position for RoPE Embeddings**

**Purpose**: Tell each token its position in the sequence for Rotary Position Embeddings (RoPE)

**How it's created**:
```python
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
```

**Example**:
```python
# If pad_masks = [T, T, T, T, T, F, F, T, T, F]
#                 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
# cumsum       = [1, 2, 3, 4, 5, 5, 5, 6, 7, 7]
# position_ids = [0, 1, 2, 3, 4, 4, 4, 5, 6, 6]  (minus 1)

# Real example for prefix:
# Positions 0-767: Image tokens (all valid) → position_ids = [0, 1, 2, ..., 767]
# Positions 768-911: Real text tokens → position_ids = [768, 769, ..., 911]
# Positions 912-967: Padding tokens → position_ids = [912, 912, ..., 912] (same as previous)
```

**Why cumsum minus 1?**
- `cumsum` counts how many valid tokens so far (1-indexed)
- Subtract 1 to get 0-indexed positions: [1,2,3,...] → [0,1,2,...]
- Padding tokens keep the same position_id as the previous valid token

**What is RoPE?**
- **Rotary Position Embedding**: Adds position info by rotating Q and K vectors
- Better than absolute position embeddings for:  
  - Relative position encoding (knows distance between tokens)
  - Length generalization (handles sequences longer than training)
- Position IDs tell RoPE how much rotation to apply

---

#### Summary Visualization

```
Input (from Step 7):
  prefix_pad_masks:  [1, 968]        Boolean: [T,T,...,T,F,F,...,F]
  prefix_att_masks:  [1, 968]        Integer: [0,0,...,0,0,0,...,0]

                            ↓

Step 8 Transformations:

  1. make_att_2d_masks()
     prefix_att_2d_masks:   [1, 968, 968]   Boolean 2D attention matrix
     
  2. torch.cumsum() - 1
     prefix_position_ids:   [1, 968]        Integer: [0,1,2,...,967]
     
  3. _prepare_attention_masks_4d()
     prefix_att_2d_masks_4d: [1, 1, 968, 968]  Float: 0.0 or -2e38

                            ↓

Ready for Transformer:
  ✓ Attention mask: Which tokens can attend to which
  ✓ Position IDs: Where each token is in the sequence
  ✓ Float format: Compatible with attention computation
```

---

### Step 9: Process Prefix & Create KV Cache
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PI05Pytorch.sample_actions()`

**Goal**: Process prefix (images + text) through PaliGemma transformer and cache the Key/Value tensors for efficient reuse during denoising

**Code Location**:
```python
def sample_actions(self, images, img_masks, tokens, masks, ...):
    # After preparing masks and position IDs (Step 8)
    
    # Process prefix through PaliGemma and cache K/V
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,  # [1, 1, 968, 968]
        position_ids=prefix_position_ids,        # [1, 968]
        past_key_values=None,                    # No cache yet
        inputs_embeds=[prefix_embs, None],       # Only process prefix
        use_cache=True,                          # Create cache!
    )

self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf) # vlm_config_hf is PaliGemmaConfig
self.model = PaliGemmaModel(config)
self.language_model = language_model = AutoModel.from_config(config=config.text_config) # config.text_config is config.text_config
# so self.language_model load GemmaModel from /opt/conda/envs/lerobot/lib/python3.10/site-packages/transformers/models/gemma/modeling_gemma.py


```

**What happens**:
1. Prefix embeddings go through 18 PaliGemma transformer layers
2. Each layer computes Query (Q), Key (K), Value (V) from prefix tokens
3. Self-attention is performed: Q attends to K/V within the prefix (images ↔ text)
4. The **output embeddings** (result after all layers) are discarded (we use `_`)
5. **Only** K and V tensors from each layer are saved to `past_key_values` for reuse

**Output**:
```python
past_key_values: tuple of 18 tuples, each containing:
    (key_cache, value_cache)
    
# Structure:
past_key_values = (
    (layer_0_keys, layer_0_values),   # Layer 0
    (layer_1_keys, layer_1_values),   # Layer 1
    ...
    (layer_17_keys, layer_17_values), # Layer 17 (18 total layers)
)

# Each key_cache and value_cache:
# Shape: [batch_size, num_kv_heads, seq_len, head_dim]
#      = [1, 1, 968, 256]
```

---

#### Understanding KV Cache

**Q1: What does the list of 18 mean? Why is each entry [1, 1, 968, 256]?**

**The list of 18 = 18 transformer layers**
- Gemma-2B has 18 transformer layers (configured as `depth=18`)
- Each layer needs its own K and V cache
- `past_key_values[i]` = cached K and V for layer `i`

**Shape breakdown: [1, 1, 968, 256]**
```python
Dimension 0: batch_size = 1
Dimension 1: num_kv_heads = 1  (Grouped Query Attention with 1 KV head)
Dimension 2: seq_len = 968     (256*3 image tokens + 200 text tokens)
Dimension 3: head_dim = 256    (each attention head has 256 dimensions)
```

**What is Grouped Query Attention (GQA)?**
- Standard attention: 8 query heads, 8 key heads, 8 value heads
- GQA (Gemma): 8 query heads, **1 key head**, **1 value head**
- The 1 KV head is shared across all 8 query heads
- Reduces memory: 8× less KV cache size!

---

**Q2: Is Q computed? Why cache only K and V?**

**Yes, Q is computed and used!**

**Transformer Attention Refresher:**
```python
# Inside each transformer layer during prefix processing:
Q = query_proj(prefix_hidden_states)   # Computed!
K = key_proj(prefix_hidden_states)     # Computed!
V = value_proj(prefix_hidden_states)   # Computed!

# Self-attention happens WITHIN the prefix:
attention_scores = Q @ K.T / sqrt(head_dim)
attention_weights = softmax(attention_scores + mask)
output = attention_weights @ V   # This is what gets passed to next layer

# After all 18 layers:
# - Final output embeddings → Discarded (the `_` variable)
# - K and V tensors from each layer → Saved to past_key_values
# - Q tensors → Not saved (were only needed for computation)
```

**What happens to each component:**
- **Q (Query)**: Computed in each layer, used for self-attention, then **discarded** (not cached)
- **K (Key)**: Computed in each layer, used for self-attention, then **cached** for reuse
- **V (Value)**: Computed in each layer, used for self-attention, then **cached** for reuse
- **Output**: Final result after all layers → **discarded** (the `_` variable)

**Why cache only K and V?**

The prefix (images + text) is **fixed context**:
- Will be attended to 64 times during the denoising loop
- K and V don't change, so we cache them
- Later processing will provide new Q to query this cached context

**Self-attention within prefix (images ↔ text) in Step 9:**
```python
# What actually happens in Step 9:
for layer in range(18):
    # Compute all three:
    prefix_Q = query_proj(hidden_states)
    prefix_K = key_proj(hidden_states)  
    prefix_V = value_proj(hidden_states)
    
    # Self-attention: prefix tokens attend to each other
    # Images ↔ Text bidirectional attention
    attention = softmax(prefix_Q @ prefix_K.T + mask) @ prefix_V
    hidden_states = mlp(attention)  # Process through rest of layer
    
    # Save K and V for later reuse:
    past_key_values[layer] = (prefix_K, prefix_V)
    # Q and output are not saved

# After all layers:
# _ (discarded) = final hidden_states
# past_key_values = [(layer0_K, layer0_V), (layer1_K, layer1_V), ...]
```

---

**Q3: Why ignore the first output from `paligemma_with_expert.forward()`?**

The forward pass returns: `(output_embeddings, kv_cache)`

```python
_, past_key_values = self.paligemma_with_expert.forward(...)
#^
#└── We discard this! (final output embeddings after all 18 layers)
```

**What is `_` (the discarded output)?**
- The **final hidden states** after processing through all 18 transformer layers
- Shape: `[1, 968, 2048]` - contextualized embeddings for each token
- Result of: Input → Layer0 → Layer1 → ... → Layer17 → Output

**Why discard the output embeddings?**

1. **We don't make predictions from the prefix directly**
   - The prefix (images + text) is just **conditioning context**
   - We're not generating output from observations alone
   - Predictions happen in later steps when processing the denoising loop

2. **Only K and V are needed for later processing**
   - Later steps will provide new queries (Q)
   - Those queries will attend to the cached prefix K and V
   - The cached K/V already contain the contextualized information
   
3. **The actual predictions come from subsequent processing**
   - The denoising loop will process noisy inputs
   - Those inputs will attend to this cached prefix context
   - Final predictions are made from that combined processing

**Analogy:**
```
Prefix processing = "Building an index of a reference book"
  → Read through the book (compute Q, K, V, self-attention)
  → Create index/table of contents (cache K and V)
  → Don't need to remember the exact reading process (discard output)
  → Later queries will look up the index when needed
```

---

#### Memory Efficiency of KV Cache

**Without KV cache** (recompute prefix every step):
```
Total prefix computations = 18 layers × 64 steps = 1,152 forward passes
Memory overhead: Minimal (no storage)
Time overhead: HUGE (1,152× computation)
```

**With KV cache** (compute prefix once):
```
Total prefix computations = 18 layers × 1 step = 18 forward passes
Memory overhead: 18 × 2 × [1,1,968,256] × 4 bytes ≈ 68 MB
Time overhead: Minimal (reuse cached K/V 64 times)

Speedup: ~64× faster inference!
```

**Storage breakdown per layer:**
```python
# Per layer:
key_cache:   [1, 1, 968, 256] × float32 = 1 × 1 × 968 × 256 × 4 bytes ≈ 0.99 MB
value_cache: [1, 1, 968, 256] × float32 = 1 × 1 × 968 × 256 × 4 bytes ≈ 0.99 MB

# Total for 18 layers:
total_kv_cache = 18 × (0.99 + 0.99) MB ≈ 35.6 MB

# Note: Q is NOT cached
# - Q was computed and used during self-attention
# - But discarded after use (not stored anywhere)
```

---

#### Complete Attention Flow

```
┌──────────────────────────────────────────────────────────────┐
│  Step 9: Process Prefix ONCE                                 │
│  Compute Q, K, V → Self-attention → Cache only K/V          │
└──────────────────────────────────────────────────────────────┘

Prefix Embeddings [1, 968, 2048]
         │
         ▼
┌─────────────────────┐
│  Layer 0            │
│  • Compute Q, K, V  │──┐ Cache K [1,1,968,256]
│  • Q @ K.T → V      │  │ Cache V [1,1,968,256]
│  • Self-attention   │  │ (Q is used then discarded)
│  • MLP              │  │
└─────────────────────┘  │
         │               │
         ▼               │
┌─────────────────────┐  │
│  Layer 1            │  │
│  • Compute Q, K, V  │──┤ Cache K, V for each layer
│  • Q @ K.T → V      │  │ (Q is used then discarded)
│  • Self-attention   │  │
│  • MLP              │  │
└─────────────────────┘  │
         │               │
        ...              │
         │               │
         ▼               │
┌─────────────────────┐  │
│  Layer 17           │  │
│  • Compute Q, K, V  │──┘
│  • Q @ K.T → V      │
│  • Self-attention   │
│  • MLP              │
└─────────────────────┘
         │
         ▼
   Output embeddings [1, 968, 2048]
   Discarded! (stored in `_`)
   
   ✓ Kept: past_key_values with K and V from each layer
   ✗ Discarded: Q (was used during computation) and output embeddings

┌──────────────────────────────────────────────────────────────┐
│  Step 12: Denoising Loop (64 iterations)                     │
│  New input provides Q, attends to cached prefix K/V          │
└──────────────────────────────────────────────────────────────┘

New Input Embeddings [1, 50, 1024]
         │
         ▼
┌─────────────────────┐
│  Layer 0            │
│  • Compute Q, K, V  ├───┐
│  (from new input)   │   │
└─────────────────────┘   │
         │                │
         │                │ Concatenate
  Cached Prefix K/V       │ K = [prefix_K; input_K]
  [1, 1, 968, 256] ───────┘ V = [prefix_V; input_V]
         │                  Full: [1, 1, 1018, 256]
         ▼
  Cross-attention:
  input_Q @ [prefix_K; input_K] → attend to prefix + new input
         │
         ▼
   Output [1, 50, 1024]
         │
         ▼
   Used for predictions
```

---

### Step 10: Action Embedding (Suffix)
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PI05Pytorch.embed_suffix()`

**Goal**: Embed noisy actions with timestep for diffusion denoising

**Code Location**:
```python
def embed_suffix(self, noisy_actions, timestep):
    # Create timestep embedding (sinusoidal)
    time_emb = create_sinusoidal_pos_embedding(timestep, dim=1024)
    time_emb = self.time_mlp(time_emb)  # Process through MLP
    
    # Embed noisy actions
    action_emb = self.action_in_proj(noisy_actions)  # [B, 50, 1024]
    
    # Combine for AdaRMS conditioning
    adarms_cond = time_emb  # Used to modulate expert Gemma layers
```

**Example Output**:
```python
suffix_embs:      [1, 50, 1024]  # Action expert hidden size
suffix_pad_masks: [1, 50]        # All True (50 action steps)
suffix_att_masks: [1, 50]        # [1, 0, 0, ..., 0] (causal for actions)
adarms_cond:      [1, 1024]      # Timestep conditioning
```

**Key Details**:
- Action chunk size: 50 timesteps
- Only first action token can attend to prefix (image + text)
- Remaining 49 action tokens use causal attention

---

### Step 11: Transformer Forward Pass (Denoising Loop)
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PaliGemmaWithExpertModel.forward()`

**Goal**: Process through dual-stream transformer (PaliGemma + Action Expert)

**Architecture**:
```
Prefix Stream (PaliGemma):          Suffix Stream (Action Expert):
[Image + Text Embeddings]           [Action Embeddings]
         ↓                                   ↓
    PaliGemma Layers                   Expert Gemma Layers
    (shared attention)                 (conditioned on time)
         ↓                                   ↓
    Cached as KV-cache    ←─────────→    Attends to prefix
```

**Code Location**:
```python
# Step 1: Process prefix (image + text) through PaliGemma
_, past_key_values = self.paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs, None],  # Only prefix
    use_cache=True,  # Cache for efficiency
)

# Step 2: Denoise loop - process suffix (actions) attending to cached prefix
for step in range(num_inference_steps):
    v_t = self.denoise_step(
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,  # Reuse cached prefix
        x_t=noisy_actions,
        timestep=current_time,
    )
    x_t = x_t + dt * v_t  # Update actions
```

**Attention Pattern**:
```
                Image (256)  Text (200)  Action (50)
Image (256)        ✓            ✓          ✗
Text (200)         ✓            ✓          ✗
Action[0]          ✓            ✓          ✓
Action[1]          ✓            ✓        ✓ ✓
Action[2]          ✓            ✓      ✓ ✓ ✓
...
Action[49]         ✓            ✓      ✓ ✓ ✓...✓

Legend:
✓ = Can attend (attention weight > 0)
✗ = Cannot attend (masked out)
```

---

### Step 12: Action Prediction (Denoising)
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PI05Pytorch.denoise_step()`

**Goal**: Predict denoised actions from model output

**Code Location**:
```python
def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
    # Process noisy actions through expert
    result = self.paligemma_with_expert.forward(
        inputs_embeds=[None, suffix_embs],  # Only suffix
        past_key_values=past_key_values,    # Attend to cached prefix
    )
    
    # Project to action space
    suffix_out = outputs_embeds[1][:, -chunk_size:]
    v_t = self.action_out_proj(suffix_out)  # Velocity prediction
    
    return v_t
```

**Example Output**:
```python
v_t: [1, 50, 7]  # Velocity for 50 timesteps, 7 DOF robot
```

**Denoising Process**:
```
Initial: x_0 = pure_noise              [1, 50, 7]
Step 1:  x_1 = x_0 + dt * v_0          [1, 50, 7]
Step 2:  x_2 = x_1 + dt * v_1          [1, 50, 7]
...
Step 64: x_64 = clean_actions          [1, 50, 7]  ← Final output
```

---

### Step 13: Action Unpadding and Return
**File**: `lerobot/policies/pi05/modeling_pi05.py` → `PI05Policy.predict_action_chunk()`

**Goal**: Remove padding and return final actions

**Code Location**:
```python
def predict_action_chunk(self, batch):
    # Get actions from model
    actions = self.model.sample_actions(images, img_masks, tokens, masks)
    # Shape: [1, 50, 32] (padded to max_action_dim=32)
    
    # Unpad to actual robot DOF
    original_action_dim = 7  # Example: 6 joints + 1 gripper
    actions = actions[:, :, :original_action_dim]
    # Shape: [1, 50, 7]
    
    return actions
```

**Final Output**:
```python
actions: [1, 50, 7]  # 50 future timesteps, 7 DOF
# Example values:
# [[[-0.05,  0.12, -0.03,  0.08, -0.01,  0.04,  1.0],  # t=0
#   [-0.04,  0.11, -0.02,  0.07,  0.00,  0.03,  1.0],  # t=1
#   ...
#   [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.0]]] # t=49
```

---

## Key Architectural Insights

### 1. **Images are NOT tokenized into the text sequence**
- Images and text are processed **separately** until embedding space
- Concatenated as embeddings, not as text tokens
- No special `<image>` tokens in the token sequence

### 2. **State is encoded as text, not as a separate modality**
- Continuous robot state → Discretized into 256 bins → Formatted as numbers in text
- Allows language model to leverage text understanding for state
- Trade-off: Numbers tokenize inefficiently (32 numbers → 125 tokens)

### 3. **Two-stream architecture**
- **PaliGemma stream**: Processes vision + language prefix
- **Action Expert stream**: Processes actions with AdaRMS time conditioning
- Streams share attention (action expert attends to PaliGemma outputs)

### 4. **Diffusion-based action prediction**
- Actions predicted via flow matching (continuous diffusion)
- 64 denoising steps (default)
- Each step: predict velocity, update action

### 5. **Attention masking strategy**
- Image + Text: **Bidirectional** (prefix-LM style)
- Actions: **Causal** (autoregressive)
- First action token acts as a "bridge" attending to all prefix tokens

---

## Shape Summary Table

| Stage | Data | Shape | Dtype | Range/Notes |
|-------|------|-------|-------|-------------|
| Raw Observation | Image | `[256, 256, 3]` | uint8 | [0, 255] |
| Raw Observation | State | `[8]` | float32 | Continuous |
| Discretized State | State bins | `[32]` | int | [0, 255] or -1 |
| Task Text | String | Variable | str | ~50-100 chars |
| Tokenized Task | Tokens | `[1, 200]` | int64 | 144 real + 56 pad |
| Tokenized Task | Attention Mask | `[1, 200]` | bool | 144 True + 56 False |
| Preprocessed Image | Image | `[1, 3, 224, 224]` | float32 | [-1, 1] |
| Image Embeddings | Vision tokens | `[1, 256, 2048]` | float32 | - |
| Text Embeddings | Language tokens | `[1, 200, 2048]` | float32 | - |
| Prefix Embeddings | Concatenated | `[1, 456, 2048]` | float32 | 256 img + 200 text |
| Action Embeddings | Suffix | `[1, 50, 1024]` | float32 | Action expert dim |
| Predicted Actions | Final output | `[1, 50, 7]` | float32 | Robot commands |

---

## Helper Tools

### Identifying Token Boundaries
Use `token_boundary_helper.py` to find where task text vs state appears in token sequence:

```python
from token_boundary_helper import TokenBoundaryTracker

tracker = TokenBoundaryTracker()
boundaries = tracker.get_token_boundaries(batch['task'][0])
masks = tracker.create_masks(batch['task'][0])

# Use for attention analysis
task_attention = attention_weights[:, :, :, boundaries['task_range'][0]:boundaries['task_range'][1]]
state_attention = attention_weights[:, :, :, boundaries['state_range'][0]:boundaries['state_range'][1]]
```

---

## File Reference

| Component | Primary File | Key Functions |
|-----------|-------------|---------------|
| Evaluation Loop | `lerobot/scripts/lerobot_eval.py` | `rollout()` |
| Environment | `lerobot/envs/base.py` | `add_envs_task_to_observation()` |
| State Discretization | `lerobot/policies/pi05/processor_pi05.py` | `Pi05PrepareStateTokenizerProcessorStep` |
| Tokenization | `lerobot/common/processors/tokenizer_processor.py` | `TokenizerProcessorStep._tokenize_text()` |
| Image Preprocessing | `lerobot/policies/pi05/modeling_pi05.py` | `PI05Policy._preprocess_images()` |
| Image Resizing | `lerobot/policies/pi05/modeling_pi05.py` | `resize_with_pad_torch()` |
| Embedding | `lerobot/policies/pi05/modeling_pi05.py` | `PI05Pytorch.embed_prefix()` |
| Model Architecture | `lerobot/policies/pi05/modeling_pi05.py` | `PaliGemmaWithExpertModel` |
| Action Prediction | `lerobot/policies/pi05/modeling_pi05.py` | `PI05Pytorch.sample_actions()` |
| Denoising | `lerobot/policies/pi05/modeling_pi05.py` | `PI05Pytorch.denoise_step()` |

---

## Questions & Answers

**Q: Why do 32 state numbers become 125 tokens?**  
A: Each digit is tokenized separately. "128" = 4 tokens: space + '1' + '2' + '8'. This is a limitation of treating numbers as text.

**Q: Are images included in the language tokens?**  
A: No! Images are processed separately through SigLIP and concatenated in embedding space, not token space.

**Q: How can I separate task text from state tokens for attention analysis?**  
A: Use `token_boundary_helper.py` to get token ranges, or tokenize substrings separately to find boundaries.

**Q: Why resize 256→224 instead of direct crop?**  
A: To preserve aspect ratio without distortion. SigLIP is trained on 224×224 images.

**Q: What's the difference between tokenizer and embedding layer?**  
A: Tokenizer = vocabulary lookup (text → integer IDs). Embedding layer = neural network (IDs → learned vectors).

---

## Visualizing Image-Language Attention

### Goal: See which image regions the model attends to for specific words

**Example**: For task "pick up the alphabet soup", visualize which image patches correspond to "alphabet soup" tokens.

### Step-by-Step Guide

#### Step 1: Enable Attention Weight Extraction

The model needs to return attention weights during the prefix processing. Modify the prefix forward pass:

```python
# In modeling_pi05.py, when processing prefix:

# Original code (Step 9):
_, past_key_values = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)

# Modified to get attention weights:
outputs, past_key_values, attention_weights = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
    return_attention_weights=True,  # Add this!
)

# attention_weights = {
#     layer_idx: tensor of shape [batch_size, num_heads, seq_len, seq_len]
# }
```

**Note**: The transformers library GemmaModel supports `output_attentions=True`, which returns attention weights. This flows through to the custom forward implementation.

---

#### Step 2: Identify Token Positions

You need to find which token positions correspond to your target phrase.

**Token Sequence Structure** (968 total tokens):
```
Position   0-255:   Camera 1 image patches (16×16 grid)
Position 256-511:   Camera 2 image patches (16×16 grid)
Position 512-767:   Camera 3 image patches (or padding)
Position 768-967:   Language tokens (task + state + padding)
```

**Find "alphabet soup" tokens**:

```python
from transformers import AutoTokenizer
from token_boundary_helper import TokenBoundaryTracker

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

# Your task text
task_text = "pick up the alphabet soup and place it in the basket"

# Tokenize just the task portion to find word positions
task_tokens = tokenizer.encode(task_text, add_special_tokens=False)
task_text_decoded = [tokenizer.decode([t]) for t in task_tokens]

# Find "alphabet soup" tokens
target_phrase = "alphabet soup"
phrase_tokens = tokenizer.encode(target_phrase, add_special_tokens=False)

# Find where these tokens appear in the full sequence
# (after <bos> + "Task: " prefix)
# Example: tokens might be at positions [5, 6, 7] in the task text
# In full sequence: add image offset (768) + task prefix tokens

# Let's say "alphabet" is token 5 and "soup" is token 7 in task text
# In full 968-token sequence:
image_offset = 768  # First 768 are images
task_start_offset = 1  # <bos> token
alphabet_token_pos = image_offset + task_start_offset + 5  # ~774
soup_token_pos = image_offset + task_start_offset + 7      # ~776
```

**Helper function**:
```python
def find_token_positions(full_text, target_phrase, tokenizer, num_image_tokens=768):
    """Find positions of a phrase in the full token sequence.
    
    Args:
        full_text: The full task text (e.g., "Task: pick up the alphabet soup...")
        target_phrase: The phrase to find (e.g., "alphabet soup")
        tokenizer: The tokenizer instance
        num_image_tokens: Number of image tokens before text (default 768 for 3 cameras)
    
    Returns:
        List of token positions in the full 968-token sequence
    """
    # Tokenize full text
    full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
    full_decoded = [tokenizer.decode([t]) for t in full_tokens]
    
    # Tokenize target phrase
    phrase_tokens = tokenizer.encode(target_phrase, add_special_tokens=False)
    phrase_decoded = [tokenizer.decode([t]) for t in phrase_tokens]
    
    # Find phrase in full sequence
    positions = []
    for i in range(len(full_tokens) - len(phrase_tokens) + 1):
        if full_tokens[i:i+len(phrase_tokens)] == phrase_tokens:
            # Found match! Add offset for image tokens
            positions.extend(range(num_image_tokens + i, num_image_tokens + i + len(phrase_tokens)))
            break
    
    return positions

# Usage:
soup_positions = find_token_positions(
    batch['task'][0],  # Full task text
    "alphabet soup",
    tokenizer,
    num_image_tokens=768
)
print(f"'alphabet soup' tokens at positions: {soup_positions}")
# Example output: [774, 775, 776]
```

---

#### Step 3: Extract Attention from Language to Image Tokens

```python
# Get attention weights from a specific layer (e.g., layer 10)
layer_idx = 10
attn_weights = attention_weights[layer_idx]  # Shape: [1, 8, 968, 968]
#                                                      [batch, heads, query, key]

# Extract attention from "alphabet soup" tokens to all image tokens
soup_to_image_attn = attn_weights[0, :, soup_positions, :768]
# Shape: [8, 3, 768]
#        [heads, soup_tokens, image_tokens]

# Average over attention heads and soup tokens
soup_to_image = soup_to_image_attn.mean(dim=0).mean(dim=0)  # Shape: [768]

# Split into separate cameras
camera1_attn = soup_to_image[:256]   # First camera
camera2_attn = soup_to_image[256:512]  # Second camera
camera3_attn = soup_to_image[512:768]  # Third camera (might be padding)
```

---

#### Step 4: Reshape Attention to 2D Spatial Map

Image patches are arranged in a **16×16 grid** (256 patches = 16×16).

```python
import torch
import numpy as np

def attention_to_heatmap(attention_weights, spatial_size=(16, 16)):
    """Convert 1D attention weights to 2D spatial heatmap.
    
    Args:
        attention_weights: Tensor of shape [256] (one per image patch)
        spatial_size: Tuple of (height, width) for reshaping (default 16×16)
    
    Returns:
        2D numpy array of shape [16, 16]
    """
    # Reshape from [256] to [16, 16]
    heatmap = attention_weights.reshape(spatial_size)
    return heatmap.cpu().numpy()

# Create heatmaps for each camera
heatmap_cam1 = attention_to_heatmap(camera1_attn)  # Shape: [16, 16]
heatmap_cam2 = attention_to_heatmap(camera2_attn)
```

---

#### Step 5: Visualize on Original Image

```python
import matplotlib.pyplot as plt
import cv2

def overlay_attention_on_image(image, attention_heatmap, alpha=0.5):
    """Overlay attention heatmap on original image.
    
    Args:
        image: Original image [224, 224, 3], range [0, 1] or [0, 255]
        attention_heatmap: Attention map [16, 16]
        alpha: Transparency of overlay (0=transparent, 1=opaque)
    
    Returns:
        Overlayed image
    """
    # Resize attention heatmap to match image size
    heatmap_resized = cv2.resize(attention_heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Convert to colormap (e.g., 'hot' or 'jet')
    import matplotlib.cm as cm
    heatmap_color = cm.jet(heatmap_norm)[:, :, :3]  # RGB, no alpha
    
    # Ensure image is in [0, 1] range
    if image.max() > 1:
        image = image / 255.0
    
    # Overlay
    overlayed = alpha * heatmap_color + (1 - alpha) * image
    return overlayed

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Get original images (from batch, after preprocessing)
img1 = batch['observation.images.image'][0].permute(1, 2, 0).cpu().numpy()  # [224, 224, 3]
img2 = batch['observation.images.image2'][0].permute(1, 2, 0).cpu().numpy()

# Denormalize from [-1, 1] to [0, 1]
img1 = (img1 + 1) / 2
img2 = (img2 + 1) / 2

# Overlay attention
overlay1 = overlay_attention_on_image(img1, heatmap_cam1, alpha=0.6)
overlay2 = overlay_attention_on_image(img2, heatmap_cam2, alpha=0.6)

# Plot
axes[0].imshow(img1)
axes[0].set_title("Camera 1 (Original)")
axes[0].axis('off')

axes[1].imshow(overlay1)
axes[1].set_title(f"Camera 1 + 'alphabet soup' Attention\n(Layer {layer_idx})")
axes[1].axis('off')

axes[2].imshow(overlay2)
axes[2].set_title(f"Camera 2 + 'alphabet soup' Attention\n(Layer {layer_idx})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('attention_visualization.png', dpi=150)
plt.show()
```

---

#### Step 6: Analyze Multiple Layers

Different layers capture different levels of abstraction. Compare early vs late layers:

```python
def visualize_attention_across_layers(attention_weights, soup_positions, images, layer_indices=[0, 5, 10, 15, 17]):
    """Visualize how attention evolves across transformer layers."""
    
    fig, axes = plt.subplots(len(layer_indices), 2, figsize=(10, len(layer_indices) * 3))
    
    for idx, layer_idx in enumerate(layer_indices):
        # Extract attention for this layer
        attn = attention_weights[layer_idx][0, :, soup_positions, :768].mean(dim=0).mean(dim=0)
        
        # Split cameras
        cam1_attn = attn[:256]
        cam2_attn = attn[256:512]
        
        # Create heatmaps
        heatmap1 = attention_to_heatmap(cam1_attn)
        heatmap2 = attention_to_heatmap(cam2_attn)
        
        # Overlay
        overlay1 = overlay_attention_on_image(images[0], heatmap1)
        overlay2 = overlay_attention_on_image(images[1], heatmap2)
        
        # Plot
        axes[idx, 0].imshow(overlay1)
        axes[idx, 0].set_title(f"Layer {layer_idx} - Camera 1")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(overlay2)
        axes[idx, 1].set_title(f"Layer {layer_idx} - Camera 2")
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_evolution.png', dpi=150)
    plt.show()

# Usage:
visualize_attention_across_layers(
    attention_weights, 
    soup_positions, 
    [img1, img2],
    layer_indices=[0, 5, 10, 15, 17]
)
```

---

### Complete Example Script

```python
import torch
from transformers import AutoTokenizer
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. Load model and data
policy = PI05Policy.from_pretrained("path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

# 2. Run inference with attention extraction
# (You'll need to modify the forward pass to return attention_weights)
observation = env.reset()
batch = policy.preprocess_observation(observation)

# Get attention from prefix processing (modify model code to support this)
# attention_weights = {...}  # Dict of layer_idx -> [1, 8, 968, 968]

# 3. Find target phrase tokens
task_text = batch['task'][0]  # "Task: pick up the alphabet soup..."
target_phrase = "alphabet soup"

def find_phrase_tokens(task_text, target_phrase, tokenizer, num_image_tokens=768):
    """Find positions of target phrase in full sequence."""
    tokens = tokenizer.encode(task_text, add_special_tokens=True)
    decoded = [tokenizer.decode([t]) for t in tokens]
    
    phrase_tokens = tokenizer.encode(target_phrase, add_special_tokens=False)
    
    # Simple substring search
    positions = []
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i:i+len(phrase_tokens)] == phrase_tokens:
            positions.extend(range(num_image_tokens + i, num_image_tokens + i + len(phrase_tokens)))
            break
    
    print(f"Found '{target_phrase}' at positions: {positions}")
    print(f"Tokens: {[decoded[p - num_image_tokens] for p in positions]}")
    return positions

phrase_positions = find_phrase_tokens(task_text, target_phrase, tokenizer)

# 4. Extract and visualize attention
layer_idx = 10  # Middle layer

attn_weights = attention_weights[layer_idx][0, :, phrase_positions, :768]
# Average over heads and phrase tokens
avg_attn = attn_weights.mean(dim=0).mean(dim=0)  # [768]

# Split by camera
cam1_attn = avg_attn[:256].reshape(16, 16).cpu().numpy()
cam2_attn = avg_attn[256:512].reshape(16, 16).cpu().numpy()

# 5. Overlay on images
img1 = batch['observation.images.image'][0].permute(1, 2, 0).cpu().numpy()
img2 = batch['observation.images.image2'][0].permute(1, 2, 0).cpu().numpy()

# Denormalize from [-1, 1] to [0, 1]
img1 = (img1 + 1) / 2
img2 = (img2 + 1) / 2

# Create overlays
heatmap1 = cv2.resize(cam1_attn, (224, 224))
heatmap2 = cv2.resize(cam2_attn, (224, 224))

heatmap1_norm = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min() + 1e-8)
heatmap2_norm = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min() + 1e-8)

import matplotlib.cm as cm
overlay1 = 0.6 * cm.jet(heatmap1_norm)[:, :, :3] + 0.4 * img1
overlay2 = 0.6 * cm.jet(heatmap2_norm)[:, :, :3] + 0.4 * img2

# 6. Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(overlay1)
axes[0].set_title(f"'alphabet soup' → Camera 1 (Layer {layer_idx})")
axes[0].axis('off')

axes[1].imshow(overlay2)
axes[1].set_title(f"'alphabet soup' → Camera 2 (Layer {layer_idx})")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('soup_attention.png', dpi=150)
plt.show()

print(f"Attention visualization saved to soup_attention.png")
print(f"Hot regions (red/yellow) = high attention from '{target_phrase}' tokens")
```

---

### Expected Results

**Interpretation**:
- **Red/Yellow regions**: High attention - where "alphabet soup" tokens look at the image
- **Blue/Purple regions**: Low attention - ignored by those tokens
- **Layer progression**: 
  - Early layers (0-5): More distributed, general features
  - Middle layers (8-12): Start focusing on relevant objects
  - Late layers (15-17): Highly focused on task-relevant regions

**What you might see**:
- Hotspots around the actual "alphabet soup" can in the image
- Attention to the "basket" when processing "basket" tokens
- Different cameras might show different attention patterns based on viewpoint

---

### Key Insights

1. **Bidirectional attention in prefix**: Language tokens CAN attend to image tokens (and vice versa)
2. **16×16 spatial resolution**: Limited by patch size (14×14 pixels per patch)
3. **Multi-head averaging**: Each head might focus on different aspects; averaging gives overall pattern
4. **Layer depth matters**: Different layers capture different levels of abstraction

This technique helps you understand:
- Whether the model correctly identifies task-relevant objects
- How language grounding works in the vision-language model
- Potential failure modes (e.g., attending to wrong objects)
