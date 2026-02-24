# Phase 2: RND Model Training - Implementation Summary

## ✅ What Was Implemented

### Core Files Created

1. **`uncertainty_quantification/rnd_models/rnd_models.py`**
   - `RND_OE`: Random Network Distillation model for observation embeddings
     - Target network (frozen, randomly initialized): 2048 → 512
     - Predictor network (trainable): 2048 → 512
     - MSE loss between predictor and target outputs
     - Checkpoint saving/loading with full metadata
   - `RNDEnsemble`: Multi-camera RND ensemble wrapper
   - Methods:
     - `forward()`: Compute uncertainty scores
     - `save_checkpoint()`: Save with full metadata (state, config, hyperparameters)
     - `load_checkpoint()`: Load trained model
     - `get_config()`: Export model configuration

2. **`uncertainty_quantification/rnd_models/rnd_trainer.py`**
   - `RNDTrainer`: Memory-efficient training with ChunkedRNDDataset
   - Features:
     - On-demand chunk loading (only 1 chunk in RAM at a time)
     - Early stopping with validation split (90/10)
     - TensorBoard logging (train/val loss, learning rate)
     - Cosine learning rate scheduling
     - Training progress plots (PNG)
     - Full checkpoint metadata
   - Methods:
     - `train()`: Main training loop with early stopping
     - `_plot_training_progress()`: Save loss curves

3. **`uncertainty_quantification/scripts/train_rnd.py`**
   - CLI script with argument parsing
   - Configurable hyperparameters via command-line
   - Automatic path construction for datasets and outputs
   - Progress reporting

4. **`uncertainty_quantification/scripts/train_all_rnd.sh`**
   - Batch script to train all 8 models sequentially
   - Trains: spatial, object, goal, long × agentview, wrist
   - Progress tracking (X/8 models)
   - Environment variable configuration (EPOCHS, BATCH_SIZE, LR)

5. **`uncertainty_quantification/configs/rnd_training.yaml`**
   - Default hyperparameters for training
   - Documentation of settings
   - Memory usage notes

6. **`uncertainty_quantification/rnd_models/__init__.py`**
   - Package initialization with exports
   - Provides: `RND_OE`, `RNDEnsemble`, `RNDTrainer`

7. **`uncertainty_quantification/test/test_rnd_training.sh`**
   - Test script for verifying training implementation
   - Trains on Phase 1 test dataset (if available)
   - Quick 3-epoch test run

8. **Updated `uncertainty_quantification/README.md`**
   - Complete Phase 2 documentation
   - Usage examples
   - Checkpoint format specification
   - Memory management details

---

## 📊 Training Specifications

### Model Architecture (from FIPER)

**RND_OE (Observation Embedding RND)**:

```
Input: (2048,) SigLIP token embeddings

Target Network (frozen):
  Linear(2048, 1024) → LeakyReLU
  Linear(1024, 2048) → LeakyReLU
  Linear(2048, 4096) → LeakyReLU
  Linear(4096, 512)

Predictor Network (trainable):
  Linear(2048, 1024) → LeakyReLU
  Linear(1024, 2048) → LeakyReLU
  Linear(2048, 4096) → LeakyReLU
  Linear(4096, 2048) → ReLU
  Linear(2048, 1024) → ReLU
  Linear(1024, 512)

Loss: MSE(predictor(x), target(x))

Total parameters: ~35M total, ~30M trainable
```

### Hyperparameters

**Default Settings**:
- Learning rate: 1e-4 (initial)
- LR scheduler: Cosine annealing to 1e-6
- Batch size: 256 (reduce to 128 for <8GB RAM systems)
- Max epochs: 100
- Early stopping patience: 10 epochs
- Optimizer: Adam (default eps=1e-8)
- Loss function: MSE
- Train/Val split: 90/10
- Random seed: 42 (for reproducibility)

**Memory Footprint**:
- ChunkedRNDDataset cache: ~1 GB (1 chunk)
- Model batch: ~2 MB (256 × 2048 × 4 bytes)
- Model parameters: ~140 MB
- Gradients: ~140 MB
- **Total: ~1.3 GB** (safe for 8GB+ RAM systems)

---

## 🚀 Usage

### 1. Train Single Model

```bash
cd /home_net/go68xoq/code/vla/lerobot_libero_uq/workspace/lerobot

# Train RND for spatial task, agentview camera
python -m uncertainty_quantification.scripts.train_rnd \
  --task-type spatial \
  --camera agentview \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-4 \
  --device cuda
```

### 2. Quick Test (Few Epochs)

```bash
# Test training pipeline with reduced epochs
python -m uncertainty_quantification.scripts.train_rnd \
  --task-type spatial \
  --camera agentview \
  --epochs 10 \
  --batch-size 128 \
  --device cuda
```

### 3. Train All 8 Models

```bash
# Train all task types and cameras (default: 100 epochs each)
./uncertainty_quantification/scripts/train_all_rnd.sh

# With custom settings
EPOCHS=50 BATCH_SIZE=128 ./uncertainty_quantification/scripts/train_all_rnd.sh
```

### 4. Monitor Training

```bash
# Monitor all models
tensorboard --logdir uncertainty_quantification/rnd_save_models/

# Monitor specific model
tensorboard --logdir uncertainty_quantification/rnd_save_models/spatial_agentview/tensorboard/

# Then open: http://localhost:6006
```

### 5. Test Implementation

```bash
# Run quick test (requires Phase 1 test data)
./uncertainty_quantification/test/test_rnd_training.sh
```

---

## 📋 Command-Line Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task-type` | str | *required* | Task type: `spatial`, `object`, `goal`, `long` |
| `--camera` | str | *required* | Camera: `agentview`, `wrist` |
| `--dataset-dir` | str | `uncertainty_quantification/rnd_dataset` | Dataset root directory |
| `--output-dir` | str | `uncertainty_quantification/rnd_save_models` | Output directory |
| `--epochs` | int | `100` | Maximum training epochs |
| `--batch-size` | int | `256` | Batch size |
| `--lr` | float | `1e-4` | Initial learning rate |
| `--lr-min` | float | `1e-6` | Minimum LR (cosine scheduler) |
| `--patience` | int | `10` | Early stopping patience |
| `--train-val-split` | float | `0.9` | Train/validation split ratio |
| `--obs-embedding-dim` | int | `2048` | Input embedding dimension |
| `--output-size` | int | `512` | RND feature dimension |
| `--rnd-loss` | str | `mse` | Loss function: `mse` or `l2` |
| `--device` | str | `cuda` | Device: `cuda` or `cpu` |
| `--seed` | int | `42` | Random seed |

---

## 📁 Expected Output Structure

```
uncertainty_quantification/rnd_save_models/
├── spatial_agentview/
│   ├── model.ckpt                               # Final model (for loading)
│   ├── best_model.ckpt                          # Best validation loss
│   ├── model_epoch_45_loss_0.0234_seed_42_2026-02-23_15-30.ckpt  # Timestamped
│   ├── metadata.json                            # Model config (JSON)
│   ├── training_progress.png                    # Loss curves plot
│   └── tensorboard/
│       └── events.out.tfevents.1708704600.hostname  # TensorBoard logs
├── spatial_wrist/
│   ├── model.ckpt
│   ├── best_model.ckpt
│   ├── metadata.json
│   ├── training_progress.png
│   └── tensorboard/
├── object_agentview/
│   └── ... (same structure)
├── object_wrist/
│   └── ...
├── goal_agentview/
│   └── ...
├── goal_wrist/
│   └── ...
├── long_agentview/
│   └── ...
└── long_wrist/
    └── ...

# Total: 8 model directories
# Size per directory: ~200 MB (model + logs + plots)
# Total size: ~1.6 GB for all 8 models
```

---

## 📝 Checkpoint Format

### model.ckpt (PyTorch checkpoint)

```python
{
    # Model state
    'state_dict': OrderedDict([...]),      # Trained weights (30M params)
    
    # Model configuration
    'model_config': {
        'obs_embedding_dim': 2048,
        'output_size': 512,
        'rnd_loss': 'mse',
        'seed': 42
    },
    
    # Training state
    'epoch': 45,                            # Final epoch
    'best_val_loss': 0.02345,              # Best validation loss
    'optimizer_state': {...},               # Optimizer state dict
    
    # Training history
    'train_losses': [0.15, 0.10, ...],     # Per-epoch train loss
    'val_losses': [0.16, 0.11, ...],       # Per-epoch val loss
    
    # Full hyperparameters (for reproducibility)
    'hyperparameters': {
        'epochs': 100,
        'batch_size': 256,
        'lr': 1e-4,
        'lr_min': 1e-6,
        'patience': 10,
        'train_val_split': 0.9,
        'obs_embedding_dim': 2048,
        'output_size': 512,
        'rnd_loss': 'mse',
        'seed': 42,
        'device': 'cuda',
        'dataset_dir': 'uncertainty_quantification/rnd_dataset/spatial',
        'camera': 'agentview'
    }
}
```

### metadata.json (Human-readable)

```json
{
  "model_config": {
    "obs_embedding_dim": 2048,
    "output_size": 512,
    "rnd_loss": "mse",
    "seed": 42
  },
  "epoch": 45,
  "best_val_loss": 0.02345,
  "hyperparameters": {
    "epochs": 100,
    "batch_size": 256,
    "lr": 0.0001,
    "lr_min": 1e-06,
    "patience": 10,
    "train_val_split": 0.9,
    "seed": 42,
    "device": "cuda",
    "dataset_dir": "uncertainty_quantification/rnd_dataset/spatial",
    "camera": "agentview"
  }
}
```

---

## 🎯 Training Process Details

### Initialization
1. Load `ChunkedRNDDataset` (metadata only, ~1 KB RAM)
2. Split into train (90%) and validation (10%)
3. Create DataLoaders with `num_workers=0` (for chunk caching)
4. Initialize RND_OE model with seed
5. Initialize Adam optimizer + Cosine scheduler
6. Initialize TensorBoard writer

### Training Loop (per epoch)
1. **Training phase**:
   - Set model to train mode
   - For each batch (256 tokens):
     - Load batch to device (~2 MB)
     - Forward pass: uncertainty = model(batch)
     - Backward pass: loss.backward()
     - Optimizer step
   - Compute average epoch loss

2. **Validation phase**:
   - Set model to eval mode
   - With torch.no_grad():
     - For each val batch:
       - Compute loss (no gradients)
   - Compute average val loss

3. **Logging**:
   - Log to TensorBoard: train loss, val loss, learning rate
   - Update progress bar

4. **Checkpoint saving**:
   - If val loss improved:
     - Save best_model.ckpt
     - Reset patience counter
   - Else:
     - Increment patience counter

5. **Early stopping**:
   - If patience >= 10:
     - Stop training
   - If NaN loss detected:
     - Stop training

### Finalization
1. Load best model state
2. Save final checkpoints:
   - model.ckpt (for easy loading)
   - Timestamped checkpoint (for archival)
3. Plot training progress (PNG)
4. Close TensorBoard writer

---

## ⚙️ Memory Management Strategy

### Why ChunkedRNDDataset?

**Problem**: Full dataset is too large for RAM
- Spatial task: 4.38M tokens × 2048 × 4 bytes = **~35 GB** per camera
- Loading all at once causes OOM errors

**Solution**: On-demand chunk loading
- Dataset split into chunks of 128K tokens (~1 GB each)
- Only current chunk kept in memory
- When batch requests token from different chunk:
  - Load new chunk
  - Discard old chunk (garbage collected)

### Memory Footprint Breakdown

```
During Training:
├─ ChunkedRNDDataset cache:   ~1.0 GB  (current chunk)
├─ Current batch:              ~0.002 GB (256 × 2048 × 4 bytes)
├─ Model parameters:           ~0.14 GB  (~35M params × 4 bytes)
├─ Gradients:                  ~0.14 GB  (same as parameters)
└─ Overhead (CUDA, etc.):      ~0.1 GB
   ────────────────────────────────────
   Total:                      ~1.4 GB

Safe for systems with 8GB+ RAM (leaves 6.6 GB for OS and other processes)
```

### Optimization Tips

**If OOM occurs**:
1. Reduce batch size: `--batch-size 128` or `--batch-size 64`
2. Use CPU if GPU RAM is limited: `--device cpu`
3. Close other programs to free RAM

**For faster training**:
1. Use GPU: `--device cuda`
2. Increase batch size (if RAM allows): `--batch-size 512`
3. Pin memory is enabled automatically for CUDA

---

## 🔍 Loading Trained Models

### For Inference (Phase 3)

```python
from uncertainty_quantification.rnd_models import RND_OE

# Load single model
model = RND_OE.load_checkpoint(
    "uncertainty_quantification/rnd_save_models/spatial_agentview/model.ckpt",
    device="cuda"
)

# Use for inference
with torch.no_grad():
    uncertainty = model(tokens)  # (batch_size,)
```

### For Multi-Camera Ensemble

```python
from uncertainty_quantification.rnd_models import RNDEnsemble

# Load both cameras
ensemble = RNDEnsemble.load_from_checkpoints(
    checkpoint_paths={
        'agentview': 'rnd_save_models/spatial_agentview/model.ckpt',
        'wrist': 'rnd_save_models/spatial_wrist/model.ckpt'
    },
    device="cuda"
)

# Compute uncertainty for both cameras
uncertainties = ensemble({
    'agentview': agentview_tokens,  # (256, 2048)
    'wrist': wrist_tokens            # (256, 2048)
})
# Returns: {
#   'agentview': (256,),
#   'wrist': (256,),
#   'overall': (256,)  # mean of both
# }
```

---

## 📊 Expected Training Behavior

### Typical Training Curves

**Healthy training** (loss decreases smoothly):
```
Epoch   Train Loss   Val Loss    LR
1       0.1500       0.1600      1e-4
10      0.0800       0.0900      9e-5
20      0.0500       0.0600      7e-5
30      0.0300       0.0350      5e-5
40      0.0200       0.0250      3e-5
45      0.0180       0.0234      2e-5  <- Best val loss
46      0.0175       0.0236      2e-5  <- Val loss increases
...     ...          ...         ...
55      0.0160       0.0245      1e-5  <- Early stopping (patience=10)
```

**Signs of issues**:
- Val loss >> train loss (overfitting): Reduce model size or increase regularization
- Loss plateaus early: Increase learning rate
- Loss fluctuates wildly: Reduce learning rate
- NaN loss: Data normalization issue or LR too high

### Training Time Estimates

With CUDA GPU (e.g., RTX 3090):
- Spatial task (4.38M tokens): ~30-45 minutes (100 epochs)
- With early stopping: typically ~20-30 minutes (stops around epoch 40-60)

With CPU:
- Same task: ~3-5 hours

**Total time for all 8 models**: ~4-6 hours (GPU, with early stopping)

---

## 🎯 What to Do Next

1. **Test the implementation**:
   ```bash
   ./uncertainty_quantification/test/test_rnd_training.sh
   ```

2. **Verify test outputs**:
   - Check `uncertainty_quantification/test/rnd_models_test/spatial_agentview/`
   - Review `metadata.json`
   - Inspect training plots
   - View TensorBoard logs

3. **Train on small subset** (optional, for quick verification):
   ```bash
   python -m uncertainty_quantification.scripts.train_rnd \
     --task-type spatial \
     --camera agentview \
     --epochs 10 \
     --batch-size 128
   ```

4. **Run full training** (if satisfied with tests):
   ```bash
   ./uncertainty_quantification/scripts/train_all_rnd.sh
   ```

5. **Monitor training**:
   ```bash
   tensorboard --logdir uncertainty_quantification/rnd_save_models/
   ```

6. **Move to Phase 3**: Integrate RND into evaluation pipeline

---

## 🐛 Troubleshooting

### Issue: Out of memory during training

**Solution**:
```bash
# Reduce batch size
python -m uncertainty_quantification.scripts.train_rnd \
  --task-type spatial --camera agentview \
  --batch-size 128  # or even 64
```

### Issue: Training too slow

**Possible causes**:
- Using CPU instead of GPU
- Batch size too small

**Solution**:
```bash
# Ensure using GPU
python -m uncertainty_quantification.scripts.train_rnd \
  --task-type spatial --camera agentview \
  --device cuda \
  --batch-size 512  # if GPU RAM allows
```

### Issue: Loss becomes NaN

**Possible causes**:
- Learning rate too high
- Data corruption

**Solution**:
```bash
# Reduce learning rate
python -m uncertainty_quantification.scripts.train_rnd \
  --task-type spatial --camera agentview \
  --lr 1e-5 --lr-min 1e-7
```

### Issue: Model not improving

**Possible causes**:
- Learning rate too low
- Dataset too small

**Solution**:
- Check dataset size in logs
- Try higher learning rate: `--lr 1e-3`
- Reduce patience: `--patience 5`

### Issue: Cannot find dataset

**Error**: `Dataset not found at uncertainty_quantification/rnd_dataset/spatial`

**Solution**: Run Phase 1 data collection first:
```bash
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial
```

---

## 📝 README Updates

The main README has been updated with:
- ✅ Complete Phase 2 implementation details
- ✅ Checkpoint format specification
- ✅ Memory management strategy
- ✅ Usage examples
- ✅ TensorBoard logging info
- ✅ Phase 2 marked as complete

---

**Implementation Date**: February 23, 2026  
**Status**: ✅ Phase 2 Complete - Ready for Testing and Training
