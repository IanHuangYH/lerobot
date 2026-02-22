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

## 📊 Why This Design?

### **1. Train on Individual Tokens (Not Mean-Pooled)**

**❌ Initial idea**: Train on mean-pooled embeddings `(4096-D)` for efficiency
- **Problem**: Distribution shift between training (mean) and inference (individual tokens)
- Mean-pooled: smoothed, lower variance
- Individual tokens: higher variance, more extreme values
- RND would give unreliable scores on individual tokens!

**✅ Corrected approach**: Train on individual tokens `(2048-D)` sampled from demonstrations
- **Benefit**: Training/inference distributions match perfectly
- **Trade-off**: Larger dataset but still manageable with sampling

### **2. Separate RND per Camera**

**Reasoning**:
- Each camera sees different viewpoints (agentview = third-person, wrist = egocentric)
- Separate models learn view-specific OOD patterns
- Enables camera-specific uncertainty debugging

**Input**: `2048-D` per camera (not concatenated `4096-D`)

### **3. Train 4 RNDs per Task Type (Not Per Scene)**

**LIBERO Structure**:
- 4 task types: spatial, object, goal, long
- 10 scenes per type = 40 total scenes

**Options Considered**:
| Approach | # Models | Generalization | Specificity | Choice |
|----------|----------|----------------|-------------|--------|
| All tasks (1 RND) | 1 | Highest | Lowest | Too general |
| **Per task type (4 RNDs)** | **4** | **Good** | **Good** | **✅ Chosen** |
| Per scene (40 RNDs) | 40 | Lowest | Highest | Overhead |

**Chosen**: 4 RND models (one per task type) balances generalization and task-specific OOD detection.

### **4. Sample 64 Tokens per Frame**

**Full collection**:
- 100 episodes × 150 frames × 256 tokens × 2 cameras = 7.68M tokens
- 7.68M × 2048 × 4 bytes ≈ **61 GB** per task type

**Sampled collection** (64 tokens):
- 100 episodes × 150 frames × 64 tokens × 2 cameras = 1.92M tokens
- 1.92M × 2048 × 4 bytes ≈ **15.7 GB** per task type

**Trade-off**: 75% storage reduction while preserving token-level variance

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
