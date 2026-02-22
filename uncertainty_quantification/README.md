# Uncertainty Quantification for Pi0.5 VLA Policy

This module implements uncertainty prediction for the Pi0.5 Vision-Language-Action (VLA) policy using Random Network Distillation (RND) on visual embeddings, following the methodology from the FIPER paper (Römer et al., 2025).

---

## 📋 Overview

**Goal**: Predict failure at runtime for Pi0.5 policy on LIBERO tasks by detecting out-of-distribution (OOD) visual observations.

**Method**: Random Network Distillation (RND) on SigLIP vision encoder embeddings
- **Training**: RND learns to predict a random target network's outputs on in-distribution (successful) demonstrations
- **Inference**: High prediction error indicates OOD/novel observations → high uncertainty → potential failure

**Key Design Decisions**:
1. ✅ **Train on individual tokens** (not mean-pooled) to enable spatial uncertainty heatmaps
2. ✅ **Separate RND per camera** (agentview, wrist) - 2048-D input each
3. ✅ **Train 4 RND models per task type** (spatial, object, goal, long) for balanced generalization
4. ✅ **Sample 64 tokens per camera per frame** during training to reduce dataset size
5. ✅ **Batch process tokens during inference** for efficiency

---

## 🎯 Architecture

### **RND Model Structure**

```
Pi0.5 Vision Encoder (SigLIP)
    ↓
Image Embeddings: (batch=1, 256 tokens, 2048-D) per camera
    ↓
Individual Token RND per Camera:
    ├─ RND_agentview: (2048-D) → uncertainty_score
    └─ RND_wrist: (2048-D) → uncertainty_score
```

### **Training Data Flow**

```
LIBERO Demonstration Frame
    ↓
Pi0.5 SigLIP Encoder → (1, 256, 2048) per camera
    ↓
Random Sample 64 tokens per camera
    ↓
Save tokens: (2048,) embeddings
    ↓
Dataset: (N_frames × 64 × 2_cameras, 2048)
```

### **Inference Data Flow**

```
Current Observation
    ↓
Pi0.5 SigLIP Encoder → (1, 256, 2048) per camera
    ↓
Batch RND forward pass on all 256 tokens
    ↓
Uncertainty per token: (256,) scores
    ↓
├─ Overall uncertainty: mean(uncertainties)
└─ Spatial heatmap: reshape(16, 16) → upsample(224, 224)
```

---

## 📊 RND Design Options & Rationale

### **Three Possible Approaches**

We consider three alternative architectures for RND-based uncertainty quantification:

---

#### **Option 1: Position-Agnostic Token-Level RND** ✅ **(Chosen for Initial Implementation)**

**Architecture**:
- **RND Input**: Individual `2048-D` SigLIP token embeddings (pure semantic features)
- **Training Data**: Randomly sample 64 tokens per camera per frame
- **Inference**: Process all 256 tokens per camera → per-token uncertainty scores
- **Output**: Spatial uncertainty heatmap `(16, 16)` + overall score

**Key Characteristics**:
- ✅ Training/inference distributions match exactly (both use individual tokens)
- ✅ Enables spatial uncertainty heatmaps (which token regions are novel?)
- ✅ VLA-aligned: Leverages spatial generalization (doesn't penalize object rearrangements)
- ✅ Moderate dataset size with sampling: ~15.7 GB per task type

**When It Detects Uncertainty**:
- Novel objects (unseen textures, shapes, colors)
- New visual features (different materials, lighting conditions)
- Object appearance changes (not position changes)

**When It Misses**:
- Spatial layout anomalies (object in wrong position but same visual features)
- Positional constraint violations (e.g., object floating in impossible location)

---

#### **Option 2: Position-Aware Token-Level RND** (Future Work)

**Architecture**:
- **RND Input**: `2176-D` = `2048-D` token embedding + `128-D` positional encoding
- **Training Data**: Same sampling strategy, but append positional encoding to each token
- **Inference**: Same spatial heatmap capability
- **Output**: Spatial uncertainty heatmap + overall score

**Positional Encoding**:
```python
def add_positional_encoding(token_emb, token_idx):
    """
    Args:
        token_emb: (2048,) - SigLIP token embedding
        token_idx: int - Token position (0-255)
    Returns:
        position_aware_emb: (2176,) - [token_emb | pos_encoding]
    """
    row = token_idx // 16  # 0-15
    col = token_idx % 16   # 0-15
    
    # 2D sinusoidal positional encoding (128-D)
    pos_encoding = create_2d_sinusoidal_encoding(row, col, dim=128)
    
    return torch.cat([token_emb, pos_encoding], dim=0)  # (2176,)

def create_2d_sinusoidal_grid(height, width, emb_dim, device):
    """Create 2D sinusoidal positional encoding (Transformer-style)"""
    y_pos = torch.arange(height, dtype=torch.float32, device=device)
    x_pos = torch.arange(width, dtype=torch.float32, device=device)
    
    dim_t = torch.arange(emb_dim // 4, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (emb_dim // 4))
    
    pos_y = y_pos[:, None] / dim_t
    pos_x = x_pos[:, None] / dim_t
    
    pos_y = torch.stack([torch.sin(pos_y), torch.cos(pos_y)], dim=2).flatten(1)
    pos_x = torch.stack([torch.sin(pos_x), torch.cos(pos_x)], dim=2).flatten(1)
    
    pos_encoding = torch.zeros(height, width, emb_dim, device=device)
    pos_encoding[:, :, :emb_dim//2] = pos_y[:, None, :]
    pos_encoding[:, :, emb_dim//2:] = pos_x[None, :, :]
    
    return pos_encoding  # (16, 16, 128)
```

**Key Characteristics**:
- ✅ Detects both feature novelty AND spatial layout novelty
- ✅ Still enables spatial heatmaps
- ⚠️ May penalize valid scene rearrangements (false positives)
- ⚠️ Higher complexity (2176-D input)

**When It Detects Uncertainty**:
- Everything from Option 1 (novel objects/features)
- PLUS: Spatial anomalies (object in unusual position)
- PLUS: Layout violations (spatial structure differs from training)

**Trade-off vs Option 1**:
- **Gain**: Better detection of spatial anomalies
- **Cost**: Lower spatial generalization (VLA strength underutilized)

---

#### **Option 3: Mean-Pooled Embedding RND** (Future Work)

**Architecture**:
- **RND Input**: `4096-D` = concatenated mean-pooled embeddings from both cameras
  - Agentview: mean over 256 tokens → `2048-D`
  - Wrist: mean over 256 tokens → `2048-D`
  - Combined: `[mean_agentview | mean_wrist]` → `4096-D`
- **Training Data**: One `4096-D` vector per frame (no token sampling needed)
- **Inference**: Single uncertainty score per frame (no spatial heatmap)
- **Output**: Overall uncertainty only (cannot localize to image regions)

**Key Characteristics**:
- ✅ Smallest dataset size: ~100K frames × 4096-D = ~1.6 GB per task type
- ✅ Holistic scene representation (captures global context)
- ✅ Fastest inference (single forward pass per frame)
- ❌ No spatial uncertainty heatmaps (loses per-token granularity)
- ❌ Training/inference distribution mismatch if we want per-token scores later

**When It Detects Uncertainty**:
- Overall scene novelty (different room layout, lighting, object distribution)
- Global appearance shifts (different environment altogether)

**When It Misses**:
- Localized anomalies (one small object differs while rest of scene is normal)
- Fine-grained spatial details (mean-pooling smooths out local variations)

**Implementation**:
```python
# Data collection
def collect_mean_pooled_embeddings(img_embs_dict):
    """
    Args:
        img_embs_dict: {
            'agentview': (1, 256, 2048),
            'wrist': (1, 256, 2048)
        }
    Returns:
        mean_pooled: (4096,) - Combined mean-pooled embedding
    """
    agentview_mean = img_embs_dict['agentview'].mean(dim=1)  # (1, 2048)
    wrist_mean = img_embs_dict['wrist'].mean(dim=1)          # (1, 2048)
    
    combined = torch.cat([agentview_mean, wrist_mean], dim=1)  # (1, 4096)
    return combined.squeeze(0)  # (4096,)

# RND model
class RND_MeanPooled(nn.Module):
    def __init__(self, input_dim=4096, output_size=512):
        super().__init__()
        self.target_network = nn.Sequential(
            nn.Linear(4096, 2048), nn.LeakyReLU(),
            nn.Linear(2048, 4096), nn.LeakyReLU(),
            nn.Linear(4096, 2048), nn.LeakyReLU(),
            nn.Linear(2048, output_size)
        )
        # ... (similar for predictor)
```

---

### **Comprehensive Comparison**

| Aspect | Option 1: Position-Agnostic | Option 2: Position-Aware | Option 3: Mean-Pooled |
|--------|---------------------------|-------------------------|----------------------|
| **RND Input Dimension** | 2048-D | 2176-D | 4096-D |
| **Input Type** | Individual token | Token + position | Mean-pooled both cameras |
| **Spatial Heatmap** | ✅ Yes | ✅ Yes | ❌ No |
| **Dataset Size (per task)** | ~15.7 GB | ~17.0 GB | ~1.6 GB |
| **Training Samples** | ~960K tokens | ~960K tokens | ~100K frames |
| **Inference Speed** | Medium (batched tokens) | Medium (batched tokens) | Fast (single forward) |
| **Detects Feature Novelty** | ✅ Yes | ✅ Yes | ✅ Yes (global) |
| **Detects Spatial Novelty** | ❌ No | ✅ Yes | ⚠️ Coarse-grained |
| **Spatial Generalization** | ✅ High | ⚠️ Lower | ✅ High |
| **False Positive Risk** | Low | Higher | Low |
| **Localization Ability** | ✅ Per-token | ✅ Per-token | ❌ Frame-level only |
| **VLA Design Alignment** | ✅ Strong | ⚠️ Moderate | ✅ Strong |

---

### **Why Option 1 is Chosen for Initial Implementation**

1. **Distribution Matching**: Training and inference both use individual tokens → no distribution shift
2. **Spatial Heatmaps**: Enables visualization of *where* uncertainty is high (critical for debugging)
3. **VLA Alignment**: VLAs are designed for spatial generalization → penalizing position changes is counterproductive
4. **Practical Balance**: Reasonable dataset size with sampling, while preserving per-token granularity
5. **Extensibility**: Can aggregate token-level scores to frame-level (Option 3 behavior) but not vice versa

**Key Insight from Design Discussion**:
> "VLA suppose can do the task even the object is in different place... maybe [position-aware] will be too sensitive, e.g., even there are no novel object but just some object exchange the location"

Position-agnostic RND focuses on detecting novel visual features (what VLAs struggle with) rather than spatial rearrangements (what VLAs excel at).

---

### **When to Consider Alternative Options**

**Option 2 (Position-Aware)** might be better if:
- Experiments show spatial layout is critical for failure prediction
- Task involves strict spatial constraints (e.g., "put in top-left drawer")
- Position-agnostic RND misses important spatial anomalies
- Combining both approaches (Option 1 + Option 2) improves detection

**Option 3 (Mean-Pooled)** might be better if:
- Spatial heatmaps are not needed (only overall uncertainty)
- Storage/computation is highly constrained
- Holistic scene-level uncertainty is more predictive than local features
- Analysis shows per-token granularity doesn't improve failure prediction

---

### **Additional Design Decisions**

#### **Separate RND per Camera**

**Reasoning**:
- Each camera sees different viewpoints (agentview = third-person, wrist = egocentric)
- Separate models learn view-specific OOD patterns
- Enables camera-specific uncertainty debugging

**Note**: Option 3 combines cameras via concatenation, but Options 1-2 keep them separate.

#### **Train 4 RNDs per Task Type**

**LIBERO Structure**:
- 4 task types: spatial, object, goal, long
- 10 scenes per type = 40 total scenes

**Options Considered**:
| Approach | # Models | Generalization | Specificity | Choice |
|----------|----------|----------------|-------------|--------|
| All tasks (1 RND) | 1 | Highest | Lowest | Too general |
| **Per task type (4 RNDs)** | **4** | **Good** | **Good** | **✅ Chosen** |
| Per scene (40 RNDs) | 40 | Lowest | Highest | Overhead |

**Chosen**: 4 RND models (one per task type) × 2 cameras = **8 total models** for Option 1/2.

For Option 3: 4 RND models (one per task type, cameras combined) = **4 total models**.

#### **Sample 64 Tokens per Frame** (Option 1 & 2)

**Full collection**:
- 100 episodes × 150 frames × 256 tokens × 2 cameras = 7.68M tokens
- 7.68M × 2048 × 4 bytes ≈ **61 GB** per task type

**Sampled collection** (64 tokens):
- 100 episodes × 150 frames × 64 tokens × 2 cameras = 1.92M tokens
- 1.92M × 2048 × 4 bytes ≈ **15.7 GB** per task type

**Trade-off**: 75% storage reduction while preserving token-level variance

**Option 3 (no sampling needed)**:
- 100 episodes × 150 frames × 1 vector × 4096-D = 100K vectors
- 100K × 4096 × 4 bytes ≈ **1.6 GB** per task type (smallest)

---

## 🗂️ Data Structure

### **Training Dataset** (per task type)

```
lerobot/data/rnd_training/
├── spatial/
│   ├── agentview_tokens.pt       # Shape: (N, 2048)
│   └── wrist_tokens.pt           # Shape: (N, 2048)
├── object/
│   ├── agentview_tokens.pt
│   └── wrist_tokens.pt
├── goal/
│   ├── agentview_tokens.pt
│   └── wrist_tokens.pt
└── long/
    ├── agentview_tokens.pt
    └── wrist_tokens.pt

# N ≈ 100 episodes × 150 frames × 64 sampled tokens = 960K tokens per camera
# File size: ~7.8 GB per camera per task type
```

### **Trained RND Models**

```
lerobot/data/rnd_models/
├── spatial_agentview_rnd.ckpt
├── spatial_wrist_rnd.ckpt
├── object_agentview_rnd.ckpt
├── object_wrist_rnd.ckpt
├── goal_agentview_rnd.ckpt
├── goal_wrist_rnd.ckpt
├── long_agentview_rnd.ckpt
└── long_wrist_rnd.ckpt

# Total: 8 RND models (4 task types × 2 cameras)
```

### **Inference Outputs** (per episode)

```
eval_logs/{eval_name}/uncertainty/
└── libero_object_0/
    ├── episode_00000_uncertainty.pt
    ├── episode_00001_uncertainty.pt
    └── ...

# Format per file:
{
    'rollout_steps': [
        {
            'step': 0,
            'uncertainty': {
                'overall': 0.0234,           # Combined score
                'agentview': 0.0250,         # Camera-specific
                'wrist': 0.0218,
                'spatial_maps': {            # Optional: for visualization
                    'agentview': torch.Tensor,  # (16, 16)
                    'wrist': torch.Tensor,      # (16, 16)
                }
            }
        },
        ...
    ],
    'metadata': {'episode_index': 0, 'num_steps': 145}
}
```

---

## 📐 Technical Details

### **Shape Summary**

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| **Training Data Collection** |
| Vision encoder output | `img_embs` | `(1, 256, 2048)` | Per camera |
| Sample tokens | `sampled_tokens` | `(64, 2048)` | Random subset |
| Save to dataset | `token` | `(2048,)` | Individual embedding |
| Full dataset | `tokens` | `(960K, 2048)` | All sampled tokens |
| **RND Training** |
| RND input | `token_emb` | `(2048,)` | Single token |
| Target output | `target_feat` | `(512,)` | Random network output |
| Predictor output | `pred_feat` | `(512,)` | Learned prediction |
| Loss | `uncertainty` | `()` | MSE between pred and target |
| **Inference** |
| Vision encoder output | `img_embs` | `(1, 256, 2048)` | Per camera |
| All tokens (batched) | `tokens` | `(256, 2048)` | All patches |
| RND outputs (batched) | `uncertainties` | `(256,)` | Per-token scores |
| Spatial map | `uncertainty_map` | `(16, 16)` | Reshaped |
| Upsampled heatmap | `heatmap` | `(224, 224)` | For visualization |

### **RND Network Architecture** (from FIPER)

```python
# RND_OE (Observation Embedding)
# Input: (2048,) - individual token embedding

# Target Network (frozen)
nn.Sequential(
    nn.Linear(2048, 1024), nn.LeakyReLU(),
    nn.Linear(1024, 2048), nn.LeakyReLU(),
    nn.Linear(2048, 4096), nn.LeakyReLU(),
    nn.Linear(4096, 512)  # output_size
)

# Predictor Network (trainable)
nn.Sequential(
    nn.Linear(2048, 1024), nn.LeakyReLU(),
    nn.Linear(1024, 2048), nn.LeakyReLU(),
    nn.Linear(2048, 4096), nn.LeakyReLU(),
    nn.Linear(4096, 2048), nn.ReLU(),
    nn.Linear(2048, 1024), nn.ReLU(),
    nn.Linear(1024, 512)  # output_size
)

# Loss: MSE(predictor(token), target(token))
```

---

## 🚀 Implementation Phases

### **Phase 1: Data Collection for RND Training**

**Goal**: Extract and save token embeddings from LIBERO demonstrations

**Steps**:
1. Download LIBERO demonstration datasets (100 episodes per task type)
2. Create data collection script: `scripts/collect_rnd_training_data.py`
3. For each frame:
   - Load Pi0.5 policy (frozen, evaluation mode)
   - Extract SigLIP embeddings via `embed_prefix()`
   - Sample 64 random tokens per camera
   - Save as individual `(2048,)` vectors
4. Save datasets per task type and camera

**Output**:
- `data/rnd_training/{task_type}/{camera}_tokens.pt`
- ~15.7 GB per task type (4 task types → ~63 GB total)

**Files to Create**:
- `lerobot/uncertainty_quantification/data_collection.py` - Core extraction logic
- `lerobot/scripts/collect_rnd_training_data.py` - CLI script
- `lerobot/configs/rnd_data_collection.yaml` - Configuration

---

### **Phase 2: RND Model Training**

**Goal**: Train 8 RND models (4 task types × 2 cameras)

**Steps**:
1. Port RND model code from `fiper_template` to `lerobot/uncertainty_quantification/`
   - Adapt `rnd_models.py` (RND_OE class)
   - Adapt `rnd_trainer.py`
2. Create training script with PyTorch DataLoader
3. Train each RND model:
   - Input: `(2048,)` token embeddings
   - Output: `(512,)` features
   - Loss: MSE between predictor and frozen target
4. Save trained models as `.ckpt` files

**Hyperparameters** (from FIPER):
- Learning rate: 1e-4
- Batch size: 256
- Epochs: 50-100 (with early stopping)
- Optimizer: Adam
- Loss: MSE

**Output**:
- `data/rnd_models/{task_type}_{camera}_rnd.ckpt`
- 8 model files (each ~50-100 MB)

**Files to Create**:
- `lerobot/uncertainty_quantification/rnd/` - RND model code
- `lerobot/scripts/train_rnd.py` - Training script
- `lerobot/configs/rnd_training.yaml` - Training config

---

### **Phase 3: Inference Integration**

**Goal**: Integrate RND uncertainty prediction into `lerobot_eval.py`

**Steps**:
1. Modify `modeling_pi05.py`:
   - Add `enable_uncertainty_prediction()` method
   - Store latest embeddings during `embed_prefix()`
   - Add `get_uncertainty_scores()` method
2. Modify `lerobot_eval.py`:
   - Add `--eval.save_uncertainty_maps` flag
   - Load appropriate RND models based on task type
   - Call RND inference after each action prediction
   - Save uncertainty scores per rollout step
3. Implement batched token processing for efficiency

**Uncertainty Computation**:
```python
# In modeling_pi05.py
def get_uncertainty_scores(self):
    uncertainties = {}
    for camera in ['agentview', 'wrist']:
        tokens = self.latest_embeddings[camera]  # (256, 2048)
        rnd = self.rnd_models[camera]
        
        # Batch forward pass
        targets = rnd.target_network(tokens)  # (256, 512)
        preds = rnd.predictor_network(tokens)  # (256, 512)
        token_uncertainties = F.mse_loss(preds, targets, reduction='none').mean(dim=1)  # (256,)
        
        uncertainties[camera] = token_uncertainties.mean().item()  # Overall score
        uncertainties[f'{camera}_spatial'] = token_uncertainties.view(16, 16)  # Spatial map
    
    uncertainties['overall'] = (uncertainties['agentview'] + uncertainties['wrist']) / 2
    return uncertainties
```

**Output**:
- `eval_logs/{eval_name}/uncertainty/{task}/episode_XXXXX_uncertainty.pt`

**Files to Modify**:
- `lerobot/src/lerobot/policies/pi05/modeling_pi05.py`
- `lerobot/src/lerobot/scripts/lerobot_eval.py`
- `lerobot/src/lerobot/configs/default.py` (add `save_uncertainty_maps` flag)

---

### **Phase 4: Visualization**

**Goal**: Visualize uncertainty heatmaps overlaid on rollout videos

**Steps**:
1. Create verification script (similar to `verify_attention_video_alignment.py`)
2. For each timestep:
   - Load uncertainty spatial map `(16, 16)`
   - Upsample to image resolution `(224, 224)`
   - Overlay on video frame with colormap (red = high uncertainty)
3. Create side-by-side visualizations:
   - Left: agentview + uncertainty overlay
   - Right: wrist + uncertainty overlay
4. Generate timeline plots showing uncertainty over episode

**Visualization Types**:
1. **Heatmap overlay**: Uncertainty as transparent red overlay on video
2. **Pure heatmap**: Uncertainty map only (no video background)
3. **Timeline graph**: Uncertainty score vs. timestep
4. **Comparison**: Success vs. failure episodes

**Output**:
- `eval_logs/{eval_name}/uncertainty/{task}/verification/`
  - `episode_XXXXX/timestep_XXX_agentview_overlay.png`
  - `episode_XXXXX/timestep_XXX_wrist_overlay.png`
  - `episode_XXXXX/uncertainty_timeline.png`

**Files to Create**:
- `lerobot/uncertainty_quantification/visualization.py`
- `lerobot/pi_setting/eval/verify_uncertainty_video_alignment.py`
- `lerobot/pi_setting/eval/run_verify_uncertainty_video_alignment.sh`

---

## 🔧 File Structure (After Implementation)

```
lerobot/
├── uncertainty_quantification/
│   ├── README.md                    # This file
│   ├── __init__.py
│   ├── data_collection.py           # Phase 1: Extract embeddings
│   ├── visualization.py             # Phase 4: Create heatmaps
│   └── rnd/
│       ├── __init__.py
│       ├── rnd_models.py            # Adapted from fiper_template
│       ├── rnd_trainer.py           # Training logic
│       └── utils.py
├── scripts/
│   ├── collect_rnd_training_data.py # Phase 1 CLI
│   └── train_rnd.py                 # Phase 2 CLI
├── configs/
│   ├── rnd_data_collection.yaml
│   └── rnd_training.yaml
├── data/
│   ├── rnd_training/
│   │   ├── spatial/
│   │   ├── object/
│   │   ├── goal/
│   │   └── long/
│   └── rnd_models/
│       ├── spatial_agentview_rnd.ckpt
│       ├── spatial_wrist_rnd.ckpt
│       └── ...
├── pi_setting/eval/
│   ├── verify_uncertainty_video_alignment.py
│   └── run_verify_uncertainty_video_alignment.sh
└── src/lerobot/
    ├── policies/pi05/modeling_pi05.py  # Modified for RND
    ├── scripts/lerobot_eval.py         # Modified for uncertainty saving
    └── configs/default.py              # Add save_uncertainty_maps flag
```

---

## � Future Considerations: Position-Aware RND

### **Current Approach: Position-Agnostic**

The current implementation uses **position-agnostic** token-level RND:
- **Input**: Pure `2048-D` SigLIP token embeddings (semantic features only)
- **Rationale**: VLAs are designed for spatial generalization - they should succeed even when objects are repositioned
- **Benefit**: Avoids false positives from rearranged scenes (different object positions in same task type)
- **Detection focus**: Novel objects, textures, or features (not spatial layout changes)

### **Alternative Approach: Position-Aware RND**

Position-aware RND could be explored in future work to detect spatial novelty:

**Architecture**:
```python
# Add positional encoding to token embeddings
def add_positional_encoding(token_emb, token_idx):
    """
    Args:
        token_emb: (2048,) - SigLIP token embedding
        token_idx: int - Token position (0-255)
    
    Returns:
        position_aware_emb: (2176,) - [token_emb | pos_encoding]
    """
    # Convert token index to 2D grid position
    row = token_idx // 16  # 0-15
    col = token_idx % 16   # 0-15
    
    # Create learnable or fixed positional encoding (128-D)
    # Option 1: Sinusoidal (like Transformer)
    pos_encoding = create_2d_sinusoidal_encoding(row, col, dim=128)
    
    # Option 2: Learnable embedding
    # pos_encoding = learned_pos_embedding[row, col]  # (128,)
    
    # Concatenate: (2048,) + (128,) = (2176,)
    return torch.cat([token_emb, pos_encoding], dim=0)

# RND model input changes from 2048-D → 2176-D
```

**Shape Changes**:
| Component | Position-Agnostic | Position-Aware |
|-----------|------------------|----------------|
| Token embedding | `(2048,)` | `(2048,)` |
| Positional encoding | N/A | `(128,)` |
| **RND input** | **`(2048,)`** | **`(2176,)`** |
| RND output | `(512,)` | `(512,)` |

**When Position-Aware Might Be Useful**:

1. **Detecting spatial constraint violations**:
   - Example: Object appears in physically impossible location (e.g., cup floating in air)
   - Example: Robot gripper position deviates from expected trajectory

2. **Task-specific spatial priors**:
   - Tasks with strong spatial structure (e.g., "put item in top-left drawer")
   - Detecting when object positions violate learned spatial distributions

3. **Fine-grained OOD detection**:
   - Combine both approaches: position-agnostic RND + position-aware RND
   - Position-agnostic: "Is this object/texture novel?"
   - Position-aware: "Is this spatial layout novel?"
   - Aggregate both signals for comprehensive uncertainty

**Trade-offs**:

| Aspect | Position-Agnostic | Position-Aware |
|--------|------------------|----------------|
| **Spatial generalization** | ✅ Better | ⚠️ May penalize valid rearrangements |
| **Detect feature novelty** | ✅ Yes | ✅ Yes |
| **Detect spatial novelty** | ❌ No | ✅ Yes |
| **False positive risk** | Lower | Higher (scene variations) |
| **Model complexity** | Simpler (2048-D) | More complex (2176-D) |

**Implementation Sketch** (for future work):

```python
# In data_collection.py
def collect_position_aware_tokens(img_embs):
    """
    Args:
        img_embs: (1, 256, 2048) - SigLIP embeddings
    
    Returns:
        position_aware_tokens: (256, 2176) - With positional encodings
    """
    batch, num_tokens, emb_dim = img_embs.shape
    tokens = img_embs[0]  # (256, 2048)
    
    # Add 2D sinusoidal positional encoding
    pos_encodings = create_2d_sinusoidal_grid(
        height=16, width=16, emb_dim=128, device=tokens.device
    )  # (16, 16, 128)
    pos_encodings = pos_encodings.view(256, 128)  # (256, 128)
    
    # Concatenate: (256, 2048) + (256, 128) = (256, 2176)
    position_aware_tokens = torch.cat([tokens, pos_encodings], dim=1)
    
    return position_aware_tokens

def create_2d_sinusoidal_grid(height, width, emb_dim, device):
    """Create 2D sinusoidal positional encoding (Transformer-style)"""
    # Generate position indices
    y_pos = torch.arange(height, dtype=torch.float32, device=device)  # (16,)
    x_pos = torch.arange(width, dtype=torch.float32, device=device)   # (16,)
    
    # Frequency bands
    dim_t = torch.arange(emb_dim // 4, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (emb_dim // 4))
    
    # Encode y and x separately (each gets emb_dim//2 dimensions)
    pos_y = y_pos[:, None] / dim_t  # (16, emb_dim//4)
    pos_x = x_pos[:, None] / dim_t  # (16, emb_dim//4)
    
    # Apply sin/cos
    pos_y = torch.stack([torch.sin(pos_y), torch.cos(pos_y)], dim=2).flatten(1)  # (16, emb_dim//2)
    pos_x = torch.stack([torch.sin(pos_x), torch.cos(pos_x)], dim=2).flatten(1)  # (16, emb_dim//2)
    
    # Combine: (height, width, emb_dim)
    pos_encoding = torch.zeros(height, width, emb_dim, device=device)
    pos_encoding[:, :, :emb_dim//2] = pos_y[:, None, :]  # Broadcast over width
    pos_encoding[:, :, emb_dim//2:] = pos_x[None, :, :]  # Broadcast over height
    
    return pos_encoding  # (16, 16, 128)

# RND model architecture change
class RND_OE_PositionAware(nn.Module):
    def __init__(self, input_dim=2176, output_size=512):  # 2048 + 128 = 2176
        super().__init__()
        self.target_network = nn.Sequential(
            nn.Linear(2176, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 2048), nn.LeakyReLU(),
            nn.Linear(2048, 4096), nn.LeakyReLU(),
            nn.Linear(4096, output_size)
        )
        # ... (same for predictor_network)
```

**Experimental Validation Needed**:

To decide whether position-aware RND is beneficial, measure:
1. **False positive rate**: Does it flag valid scene rearrangements?
2. **True positive rate**: Does it detect genuine spatial anomalies?
3. **Correlation with success**: Does position-aware uncertainty better predict failure?
4. **Comparison**: Position-agnostic vs position-aware vs combined

**Recommendation**: Start with position-agnostic (current design), then experiment with position-aware if:
- Analysis shows spatial layout is critical for failure prediction
- Position-agnostic RND misses important spatial anomalies
- Task-specific evaluation shows position matters more than expected

---

## �📚 References

1. **FIPER Paper**: Römer et al. (2025). "Failure Prediction at Runtime for Generative Robot Policies." NeurIPS 2025.
   - arXiv: https://arxiv.org/abs/2510.09459
   - Code: https://github.com/tum-lsy/fiper

2. **RND Paper**: Burda et al. (2018). "Exploration by Random Network Distillation." ICLR 2019.

3. **Pi0.5 Paper**: Black et al. (2024). "π₀: A Vision-Language-Action Flow Model for General Robot Control."

4. **LIBERO**: Liu et al. (2023). "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning."

---

## ⚙️ Configuration Example

```yaml
# configs/rnd_training.yaml
rnd:
  task_type: "spatial"  # or "object", "goal", "long"
  camera: "agentview"   # or "wrist"
  
  model:
    obs_embedding_dim: 2048
    output_size: 512
    rnd_loss: "mse"
  
  training:
    learning_rate: 1e-4
    batch_size: 256
    num_epochs: 100
    early_stopping_patience: 10
    
  data:
    tokens_per_frame: 64  # Sampled during collection
    train_val_split: 0.8
```

---

## 🎯 Success Metrics

1. **Data Collection**: Successfully extract ~960K tokens per camera per task type
2. **Training**: RND converges (predictor MSE stabilizes)
3. **Inference**: Uncertainty scores correlate with task success/failure
4. **Visualization**: Clear spatial heatmaps showing high uncertainty in relevant regions
5. **Performance**: Inference adds <100ms per timestep (batched token processing)

---

## 📝 Next Steps

1. ✅ Create folder structure
2. ✅ Document design decisions (this file)
3. ⏭️ **Start Phase 1**: Implement data collection script
4. ⏭️ **Phase 2**: Train RND models
5. ⏭️ **Phase 3**: Integrate into evaluation pipeline
6. ⏭️ **Phase 4**: Create visualization tools

---

**Last Updated**: February 22, 2026
