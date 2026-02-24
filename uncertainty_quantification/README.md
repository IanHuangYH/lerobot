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
4. ✅ **Use all 256 tokens per camera per frame** during training for maximum spatial coverage
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
Save all 256 tokens per camera
    ↓
Save tokens: (2048,) embeddings
    ↓
Dataset: (N_frames × 256 × 2_cameras, 2048)
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
- **Training Data**: All 256 tokens per camera per frame (no sampling)
- **Inference**: Process all 256 tokens per camera → per-token uncertainty scores
- **Output**: Spatial uncertainty heatmap `(16, 16)` + overall score

**Key Characteristics**:
- ✅ Training/inference distributions match exactly (both use individual tokens)
- ✅ Enables spatial uncertainty heatmaps (which token regions are novel?)
- ✅ VLA-aligned: Leverages spatial generalization (doesn't penalize object rearrangements)
- ✅ Full token storage (256 tokens/frame): ~76 GB per camera, ~152 GB per task type (all 500 episodes)

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
| **Dataset Size (per task)** | ~70 GB | ~76 GB | ~1.1 GB |
| **Training Samples** | ~8.77M tokens | ~8.77M tokens | ~68.5K frames |
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

> **Note**: This section describes the original design option. The **actual implementation uses all 256 tokens** (no sampling) for maximum spatial coverage. See Phase 1 implementation details below.

**LIBERO Dataset Structure**:
- 10 task files per type × 50 demos per file = **500 episodes per task type**
- Average episode length: ~137 frames
- Total frames per task type: ~68,500 frames

**Full collection** (all tokens):
- 500 episodes × 137 frames × 256 tokens × 2 cameras = 35.1M tokens
- 35.1M × 2048 × 4 bytes ≈ **280 GB** per task type

**Sampled collection** (64 tokens):
- 500 episodes × 137 frames × 64 tokens × 2 cameras = 8.77M tokens
- 8.77M × 2048 × 4 bytes ≈ **70 GB** per task type

**Trade-off**: 75% storage reduction while preserving token-level variance

**Option 3 (no sampling needed)**:
- 500 episodes × 137 frames × 1 vector × 4096-D = 68.5K vectors
- 68.5K × 4096 × 4 bytes ≈ **1.1 GB** per task type (smallest)

**Configurable**: The data collection script supports `--num-episodes` to use fewer episodes if needed (e.g., 100 for faster experimentation)

---

## 🗂️ Data Structure

### **Training Dataset** (per task type)

**Chunked Format** (memory-efficient):

```
lerobot/uncertainty_quantification/rnd_dataset/
├── spatial/
│   ├── agentview_tokens_00000.pt # First chunk: (512000, 2048) ~2 GB
│   ├── agentview_tokens_00001.pt # Second chunk
│   ├── ... (38 chunks total)
│   ├── agentview_tokens_00037.pt # Last chunk (partial: ~73,792 tokens)
│   ├── wrist_tokens_00000.pt     # First chunk: (512000, 2048) ~2 GB
│   ├── wrist_tokens_00001.pt     # Second chunk
│   ├── ... (38 chunks total)
│   ├── wrist_tokens_00037.pt     # Last chunk (partial)
│   └── collection_stats.json     # Metadata
├── object/
│   ├── agentview_tokens_00000.pt
│   ├── ... (76 chunk files total: 38 per camera)
│   └── collection_stats.json
├── goal/
│   ├── agentview_tokens_00000.pt
│   ├── ... (76 chunk files total)
│   └── collection_stats.json
└── long/
    ├── agentview_tokens_00000.pt
    ├── ... (76 chunk files total)
    └── collection_stats.json

# Chunk Details:
# - Chunk size: ~2000 frames × 256 tokens = 512,000 tokens per chunk
# - Shape per chunk: (512000, 2048) for full chunks, smaller for last chunk
# - File size: ~2 GB per chunk (with all 256 tokens/frame)
# - Total chunks: 38 per camera (for 500 episodes, ~74,507 frames total)
# - Naming: Zero-padded 5-digit index ({camera}_tokens_{idx:05d}.pt)
#
# Total dataset size per task type: ~152 GB (2 cameras × 38 chunks × 2 GB)
# Configurable via --num-episodes and --max-frames-per-chunk flags
```

**Loading Chunked Data**:

```python
# Option 1: Load all chunks (for small datasets or experimentation)
from uncertainty_quantification.dataset import load_rnd_dataset
tokens = load_rnd_dataset("rnd_dataset", "spatial", "agentview")
# Returns: torch.Tensor of shape (19.07M, 2048)

# Option 2: On-demand loading for training (memory-efficient)
from uncertainty_quantification.dataset import ChunkedRNDDataset
from torch.utils.data import DataLoader

dataset = ChunkedRNDDataset("rnd_dataset", "spatial", "agentview")
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
# Loads chunks on-demand during iteration

# Option 3: Query metadata without loading
from uncertainty_quantification.dataset import get_chunk_info
info = get_chunk_info("rnd_dataset", "spatial")
# Returns: {'num_chunks': 35, 'total_tokens_agentview': 4384000, ...}
```

### **Trained RND Models**

```
lerobot/uncertainty_quantification/rnd_save_models/
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
| Full dataset | `tokens` | `(19.07M, 2048)` | All tokens per camera (default: 500 episodes, 256 tokens/frame) |
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

### **Phase 1: Data Collection for RND Training** ✅

**Status**: Complete (February 2026)

**Goal**: Extract and save token embeddings from LIBERO demonstrations in chunks

**Steps**:
1. Check if LIBERO is downloaded (third_party/LIBERO/datasets)
   - Expected: 10 demo files per task type, 50 episodes per file = **500 episodes total**
2. Load Pi0.5 policy (frozen, evaluation mode)
3. For each .hdf5 file (with progress tracking):
   - Extract SigLIP embeddings via vision encoder for each frame
   - Sample 64 random tokens per camera (configurable)
   - Accumulate tokens and save in chunks (2000 frames per chunk)
4. Save chunked datasets per task type and camera

**Output** (with actual implementation: all 500 episodes, 256 tokens/frame, ~2000 frames/chunk):
- Chunked format: `{camera}_tokens_00000.pt`, `{camera}_tokens_00001.pt`, etc.
  - **38 chunks per camera** (each ~2 GB for 256 tokens/frame)
  - Chunk size: 512,000 tokens per chunk (variable for last chunk)
- `uncertainty_quantification/rnd_dataset/{task_type}/collection_stats.json` (metadata)
- **Actual statistics** (from collection_stats.json):
  - Total tokens per camera: 19,073,792
  - Total frames: 74,507
  - Tokens per frame: 256 (full, no sampling)
  - Num chunks: 38
- **Total size per task type**: ~152 GB (2 cameras × 38 chunks × 2 GB)
- **Total for 3 task types**: ~456 GB (spatial, object, goal)

**Memory Management**:
- Data saved in chunks to avoid OOM errors during collection
- Default chunk size: 2000 frames (~2 GB per chunk with 256 tokens)
- Explicit garbage collection after each chunk save
- Can process full dataset even on systems with limited RAM

**Usage**:
```bash
# Collect data for spatial task type (all 500 episodes)
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial \
  --tokens-per-frame 64 \
  --max-frames-per-chunk 2000

# Use fewer episodes for faster experimentation
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial \
  --num-episodes 100 \
  --tokens-per-frame 64

# Save all 256 tokens (no sampling) - larger chunks
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial \
  --save-full-tokens \
  --max-frames-per-chunk 500
```

**Configuration Options**:
- `--task-type`: Choose from `spatial`, `object`, `goal`, `long`
- `--num-episodes`: Number of episodes to use (default: `None` = all ~500)
- `--tokens-per-frame`: Tokens to sample per camera (default: `256`, legacy option supports `64`)
- `--max-frames-per-chunk`: Max frames per chunk file (default: `2000`)
- `--save-full-tokens`: Save all 256 tokens (default: `True` in current implementation)
- `--libero-dataset-dir`: Path to LIBERO datasets (default: `third_party/LIBERO/datasets`)
- `--output-dir`: Output directory (default: `uncertainty_quantification/rnd_dataset`)
- `--policy-path`: Pi0.5 checkpoint (default: `pi-0-5-preview`)

**Files Created**: ✅
- `lerobot/uncertainty_quantification/dataset/data_collection.py` - Core extraction logic with chunked saving
- `lerobot/uncertainty_quantification/dataset/chunk_loader.py` - Utilities to load chunked datasets
- `lerobot/uncertainty_quantification/dataset/example_usage.py` - Examples for loading chunks
- `lerobot/uncertainty_quantification/scripts/collect_rnd_training_data.py` - CLI script
- `lerobot/uncertainty_quantification/configs/rnd_data_collection.yaml` - Configuration
- `lerobot/uncertainty_quantification/test/test_data_collection.sh` - Test script (2 episodes)

---

### **Phase 2: RND Model Training** ✅

**Goal**: Train 8 RND models (4 task types × 2 cameras)

**Steps**:
1. ✅ Port RND model code from `fiper_template` to `lerobot/uncertainty_quantification/`
   - ✅ Adapt `rnd_models.py` (RND_OE class with checkpoint saving)
   - ✅ Adapt `rnd_trainer.py` (memory-efficient training with ChunkedRNDDataset)
2. ✅ Create training script with PyTorch DataLoader
3. ✅ Train each RND model:
   - Input: `(2048,)` token embeddings
   - Output: `(512,)` features
   - Loss: MSE between predictor and frozen target
4. ✅ Save trained models as `.ckpt` files with full metadata

**Hyperparameters** (from FIPER):
- Learning rate: 1e-4 (with cosine annealing to 1e-6)
- Batch size: 256
- Epochs: up to 100 (with early stopping patience=10)
- Optimizer: Adam
- Loss: MSE
- Train/Val split: 90/10

**Output Structure**:
```
uncertainty_quantification/rnd_save_models/
├── spatial_agentview/
│   ├── best_model.ckpt               # Best validation loss (used for inference)
│   ├── model.ckpt                    # Final checkpoint (fallback)
│   ├── model_epoch_XX_loss_XXX_seed_42_YYYY-MM-DD_HH-MM.ckpt  # Timestamped
│   ├── metadata.json                 # Model config + hyperparameters
│   ├── training_progress.png         # Loss curves plot
│   └── tensorboard/                  # TensorBoard logs
│       └── events.out.tfevents.*
├── spatial_wrist/
│   ├── model.ckpt
│   └── ...
├── object_agentview/
├── object_wrist/
├── goal_agentview/
├── goal_wrist/
├── long_agentview/
└── long_wrist/

# Total: 8 model directories (each ~100-200 MB with all files)
```

**Checkpoint Format** (full metadata for reproducibility):
```python
{
    'state_dict': model.state_dict(),           # Trained weights
    'model_config': {                           # Model architecture
        'obs_embedding_dim': 2048,
        'output_size': 512,
        'rnd_loss': 'mse',
        'seed': 42
    },
    'epoch': 45,                                # Training progress
    'best_val_loss': 0.0234,
    'train_losses': [0.15, 0.10, ...],         # Full training history
    'val_losses': [0.16, 0.11, ...],
    'hyperparameters': {                        # Full training config
        'epochs': 100,
        'batch_size': 256,
        'lr': 1e-4,
        'lr_min': 1e-6,
        'patience': 10,
        'train_val_split': 0.9,
        'seed': 42,
        'device': 'cuda',
        'dataset_dir': 'rnd_dataset/spatial',
        'camera': 'agentview'
    }
}
```

**Memory Management** (Critical for Large Datasets):
- ✅ Uses `ChunkedRNDDataset` with **smart LRU cache**
- ✅ **Smart pre-loading**: If `max_cached_chunks >= 60% of total chunks`, pre-loads all specified chunks upfront
  - With `max_cached_chunks=38`: All chunks pre-loaded → **100% cache hit rate**, ~76 GB RAM
  - With `max_cached_chunks=25`: First 25 chunks pre-loaded → ~66% coverage, ~50 GB RAM
  - With `max_cached_chunks=4`: Lazy loading with LRU eviction → ~40-50% hit rate, ~8 GB RAM
- ✅ **LRU eviction** for smaller cache sizes (keeps most-used chunks in memory)
- ✅ **Cache statistics tracking**: Reports hit/miss rates, memory usage after training
- ✅ **BFloat16 to Float32 conversion**: Handles dtype mismatch from data collection
- ✅ Batch size 256 = ~2 MB per batch in GPU
- ✅ Configurable via `--max-cached-chunks` (default: 38 for maximum speed)

**Training Features**:
- ✅ Early stopping with validation split (90/10)
- ✅ TensorBoard logging (train/val loss, learning rate)
- ✅ Cosine learning rate scheduling (1e-4 → 1e-6)
- ✅ Automatic checkpoint saving (best + final + timestamped)
- ✅ Training progress plots (PNG)
- ✅ Full hyperparameter logging for reproducibility
- ✅ Deterministic training with seed control
- ✅ **Nested progress bars**: Epoch-level + batch-level with real-time loss display
- ✅ **Cache statistics reporting**: Hit rate, memory usage, chunks loaded
- ✅ **Parallel camera training**: Both cameras train simultaneously per task (saves 50% time)

**Monitoring Options**:
1. **Standard mode** (`train_all_rnd.sh`):
   - Both cameras train in parallel with log file redirection
   - Logs saved to: `{task}_{camera}_training.log`
   - Use `tail -f spatial_agentview_training.log` to monitor progress
   
2. **tmux split-screen mode** (`train_all_rnd_tmux.sh`, **recommended**):
   - Vertical split: agentview (left) | wrist (right)
   - Live progress bars visible in each pane
   - Navigate panes: `Ctrl+b` then arrow keys
   - Detach: `Ctrl+b d` | Reattach: `tmux attach -t rnd_training`
   - Kill session: `tmux kill-session -t rnd_training`

**Performance Notes**:
- **Smart pre-loading** eliminates disk I/O bottleneck (100% cache hit rate)
- Training time per model (~19M tokens):
  - With `max_cached_chunks=38`: ~2-3 hours (100% cache hit, **fastest**)
  - With `max_cached_chunks=4`: ~8-12 hours (40-50% cache hit)
- Parallel training: 4 tasks × 2 cameras = ~12-18 hours total (with pre-loading)
- RAM requirements scale linearly: ~2GB per cached chunk
- See [PHASE2_SUMMARY.md](uncertainty_quantification/PHASE2_SUMMARY.md) for detailed performance analysis

**Usage**:

```bash
# Train single model
python -m uncertainty_quantification.scripts.train_rnd \
  --task-type spatial \
  --camera agentview \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-4 \
  --device cuda

# Quick test with fewer epochs
python -m uncertainty_quantification.scripts.train_rnd \
  --task-type spatial \
  --camera agentview \
  --epochs 10 \
  --batch-size 128

# Train all 8 models (both cameras in parallel per task)
./uncertainty_quantification/scripts/train_all_rnd.sh

# Train with custom settings
EPOCHS=50 BATCH_SIZE=256 MAX_CACHED_CHUNKS=38 ./uncertainty_quantification/scripts/train_all_rnd.sh

# Train with tmux split-screen monitoring (recommended!)
./uncertainty_quantification/scripts/train_all_rnd_tmux.sh

# Adjust cache size for different RAM configurations
MAX_CACHED_CHUNKS=4 ./uncertainty_quantification/scripts/train_all_rnd.sh    # 8GB RAM
MAX_CACHED_CHUNKS=25 ./uncertainty_quantification/scripts/train_all_rnd.sh   # 50GB RAM
MAX_CACHED_CHUNKS=38 ./uncertainty_quantification/scripts/train_all_rnd.sh   # 76GB+ RAM (fastest)

# Monitor training in TensorBoard
tensorboard --logdir uncertainty_quantification/rnd_save_models/

# Monitor specific model
tensorboard --logdir uncertainty_quantification/rnd_save_models/spatial_agentview/tensorboard/
```

**Files Created**: ✅
- `uncertainty_quantification/rnd_models/rnd_models.py` - RND_OE model with checkpoint I/O
- `uncertainty_quantification/rnd_models/rnd_trainer.py` - Training loop with nested progress bars, cache stats
- `uncertainty_quantification/rnd_models/__init__.py` - Package exports
- `uncertainty_quantification/scripts/train_rnd.py` - CLI training script with --max-cached-chunks
- `uncertainty_quantification/scripts/train_all_rnd.sh` - Parallel training (both cameras per task)
- `uncertainty_quantification/scripts/train_all_rnd_tmux.sh` - tmux split-screen monitoring
- `uncertainty_quantification/configs/rnd_training.yaml` - Default hyperparameters
- `uncertainty_quantification/test/test_rnd_training.sh` - Test script for validation
- `uncertainty_quantification/PHASE2_SUMMARY.md` - Complete implementation guide with performance notes

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

### **Phase 4: Visualization** ✅

**Goal**: Visualize uncertainty heatmaps overlaid on rollout videos

**Implementation**: Complete - see `PHASE4_SUMMARY.md` for details

**Key Features**:
1. **2x2 Grid Snapshots**: Overlays + pure heatmaps with uncertainty scores
2. **Timeline Plots**: Overall, camera comparison, multi-episode
3. **Batch Processing**: Automate visualization for many episodes

**Usage**:
```bash
# Single episode verification (key timesteps)
python -m uncertainty_quantification.visualization.verify_uncertainty_video_alignment \
  --uncertainty_path eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00003_uncertainty.pt \
  --video_path eval_logs/with_uncertainty/videos/libero_object_0/eval_episode_00003.mp4 \
  --output_path eval_logs/with_uncertainty/uncertainty/libero_object_0/verification \
  --timesteps 0 10 20 50 100 \
  --colormap viridis \
  --alpha 0.55 \
  --episode_id 3

# Batch verification (all episodes)
./uncertainty_quantification/scripts/run_verify_uncertainty_video_alignment.sh

# Timeline visualization (single episode)
python -m uncertainty_quantification.visualization.visualize_uncertainty_timeline \
  --uncertainty_paths eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00003_uncertainty.pt \
  --output_dir eval_logs/with_uncertainty/uncertainty/libero_object_0/timelines \
  --plot_type all

# Timeline visualization (compare multiple episodes)
python -m uncertainty_quantification.visualization.visualize_uncertainty_timeline \
  --uncertainty_paths \
      eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00000_uncertainty.pt \
      eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00001_uncertainty.pt \
  --output_dir eval_logs/with_uncertainty/uncertainty/libero_object_0/timelines \
  --episode_names "Success-1" "Failed-1" \
  --compare_episodes \
  --plot_type all
```

**Output Structure**:
```
eval_logs/{eval_name}/uncertainty/{task}/
├── verification/episode_XXXXX/              # 2x2 grid snapshots
│   ├── timestep_000_grid.png               # Overlays + heatmaps
│   ├── timestep_010_grid.png
│   └── ...
└── timelines/                               # Timeline plots
    ├── episode_00000_uncertainty_timeline.png
    ├── episode_00000_uncertainty_camera_comparison.png
    ├── multi_episode_comparison.png
    └── ...
```

**Visualization Format** (2x2 Grid):
```
┌─────────────────────────┬─────────────────────────┐
│   Agentview + Overlay   │   Wrist + Overlay       │
│   Score: 0.1450         │   Score: 0.0890         │
├─────────────────────────┼─────────────────────────┤
│   Agentview Heatmap     │   Wrist Heatmap         │
│   (Pure)                │   (Pure)                │
└─────────────────────────┴─────────────────────────┘
       Overall: 0.1170 | Timestep: 50
```

**Files Created**: ✅
- `uncertainty_quantification/visualization/__init__.py`
- `uncertainty_quantification/visualization/verify_uncertainty_video_alignment.py`
- `uncertainty_quantification/visualization/visualize_uncertainty_timeline.py`
- `uncertainty_quantification/scripts/run_verify_uncertainty_video_alignment.sh`
- `uncertainty_quantification/PHASE4_SUMMARY.md` - Complete implementation guide

---

### **Experimental Validation Plan** (for comparing options)

To determine if alternative options (2 or 3) outperform the baseline (Option 1):

**Metrics to Compare**:
1. **False Positive Rate**: Does it flag valid scene variations as uncertain?
2. **True Positive Rate**: Does it detect genuine failures before they occur?
3. **Correlation with Success**: Pearson/Spearman correlation between uncertainty and task failure
4. **Spatial Localization Quality** (Options 1-2 only): Do high-uncertainty regions align with failure causes?

**Experimental Protocol**:
1. Train all three RND variants on same training data
2. Evaluate on held-out LIBERO episodes (success and failure cases)
3. Compute uncertainty scores and compare against ground truth outcomes
4. Analyze cases where each option succeeds/fails

**Decision Criteria**:
- If Option 2 significantly improves detection without excessive false positives → adopt position-aware
- If Option 3 matches Option 1 performance → switch to mean-pooled for efficiency
- If Option 1 is sufficient → keep as-is (simpler is better)

---

## 🔧 File Structure (After Implementation)

```
lerobot/
├── uncertainty_quantification/
│   ├── README.md                    # This file
│   ├── PHASE1_SUMMARY.md            # Phase 1 implementation guide
│   ├── PHASE2_SUMMARY.md            # Phase 2 implementation guide
│   ├── PHASE3_SUMMARY.md            # Phase 3 implementation guide
│   ├── PHASE4_SUMMARY.md            # Phase 4 implementation guide
│   ├── __init__.py
│   ├── configs/
│   │   ├── rnd_data_collection.yaml
│   │   └── rnd_training.yaml
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── data_collection.py       # Phase 1: Extract embeddings
│   │   ├── chunk_loader.py          # Load chunked datasets
│   │   └── example_usage.py         # Examples
│   ├── inference/                   # Phase 3: Inference utilities
│   │   ├── __init__.py
│   │   └── rnd_inference.py         # RND model loading & uncertainty computation
│   ├── visualization/               # Phase 4: Visualization tools
│   │   ├── __init__.py
│   │   ├── verify_uncertainty_video_alignment.py  # 2x2 grid snapshots
│   │   └── visualize_uncertainty_timeline.py      # Timeline plots
│   ├── scripts/
│   │   ├── collect_rnd_training_data.py # Phase 1 CLI
│   │   ├── collect_all_tasks.sh         # Batch collection
│   │   ├── train_rnd.py                 # Phase 2 CLI
│   │   ├── train_all_rnd.sh             # Train all 8 models
│   │   ├── train_all_rnd_tmux.sh        # tmux monitoring
│   │   └── run_verify_uncertainty_video_alignment.sh  # Phase 4 batch viz
│   ├── test/
│   │   ├── test_data_collection.sh      # Phase 1 test
│   │   ├── test_rnd_training.sh         # Phase 2 test
│   │   └── test_uncertainty_inference.sh # Phase 3 test
│   └── rnd_models/
│       ├── __init__.py
│       ├── rnd_models.py            # RND_OE model class
│       └── rnd_trainer.py           # Training logic with chunked loading
├── uncertainty_quantification/rnd_dataset/
│   ├── spatial/
│   ├── object/
│   ├── goal/
│   └── long/
├── uncertainty_quantification/rnd_save_models/
│   ├── spatial_agentview/
│   ├── spatial_wrist/
│   └── ... (8 models total)
└── src/lerobot/
    ├── policies/pi05/modeling_pi05.py  # Modified for RND
    ├── scripts/lerobot_eval.py         # Modified for uncertainty saving
    └── configs/default.py              # Add save_uncertainty_maps flag
```

---

## 📚 References

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

1. **Data Collection**: Successfully extract ~19.07M tokens per camera per task type (500 episodes, 256 tokens/frame)
2. **Training**: RND converges (predictor MSE stabilizes)
3. **Inference**: Uncertainty scores correlate with task success/failure
4. **Visualization**: Clear spatial heatmaps showing high uncertainty in relevant regions
5. **Performance**: Inference adds <100ms per timestep (batched token processing)

---

## 📝 Next Steps

1. ✅ Create folder structure
2. ✅ Document design decisions (this file)
3. ✅ **Phase 1**: Implement data collection script
   - ✅ Core logic (`data_collection.py`)
   - ✅ CLI script (`scripts/collect_rnd_training_data.py`)
   - ✅ Configuration file (`configs/rnd_data_collection.yaml`)
   - ✅ Chunked dataset loader (`chunk_loader.py`)
4. ✅ **Phase 1 Execution**: Run data collection on all task types
5. ✅ **Phase 2**: Train RND models
   - ✅ RND model class (`rnd_models/rnd_models.py`)
   - ✅ Trainer with LRU cache (`rnd_models/rnd_trainer.py`)
   - ✅ CLI training script (`scripts/train_rnd.py`)
   - ✅ Batch training scripts (parallel + tmux)
6. ✅ **Phase 3**: Integrate into evaluation pipeline
   - ✅ Modify `modeling_pi05.py` for RND integration
   - ✅ Modify `lerobot_eval.py` for uncertainty saving
   - ✅ Add `save_uncertainty_maps` config flag
   - ✅ Implement RND inference utilities
7. ✅ **Phase 4**: Create visualization tools
   - ✅ 2x2 grid snapshot visualization
   - ✅ Timeline plotting (overall + camera comparison)
   - ✅ Batch processing script
8. ⏭️ **Phase 5** (Future): Statistical analysis and threshold optimization
   - Correlation between uncertainty and task success
   - ROC curves for failure prediction
   - Optimal threshold determination

---

**Last Updated**: February 24, 2026  
**Status**: Phases 1-4 Complete ✅ | Phase 5 Deferred ⏭️
