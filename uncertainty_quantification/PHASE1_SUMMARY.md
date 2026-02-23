# Phase 1: Data Collection - Implementation Summary

## ✅ What Was Implemented

### Core Files Created

1. **`uncertainty_quantification/dataset/data_collection.py`**
   - `extract_siglip_embeddings()`: Extracts SigLIP embeddings from Pi0.5 vision encoder
   - `sample_tokens_from_embeddings()`: Randomly samples tokens from embeddings
   - `load_libero_demo_file()`: Loads LIBERO HDF5 demonstration files
   - `collect_rnd_dataset()`: Main function to collect and save RND training data

2. **`uncertainty_quantification/scripts/collect_rnd_training_data.py`**
   - CLI script with argument parsing
   - Loads Pi0.5 policy and processes LIBERO demonstrations
   - Saves token embeddings and collection statistics

3. **`uncertainty_quantification/configs/rnd_data_collection.yaml`**
   - Configuration template with default settings
   - Documentation of expected dataset sizes

4. **Helper Scripts**
   - `test/test_data_collection.sh`: Test on 2 episodes before full run
   - `scripts/collect_all_tasks.sh`: Collect data for all task types

5. **`uncertainty_quantification/__init__.py`** & **`dataset/__init__.py`**
   - Package initialization with exports

## 📊 Updated Specifications

### LIBERO Dataset Structure (Actual)
- **10 demo files per task type** × **50 episodes per file** = **500 episodes per task type**
- Average episode length: ~137 frames
- Total frames per task type: ~68,500 frames

### Dataset Sizes (with default settings)

**Option A: Sampled (64 tokens/frame)** - ✅ **Implemented as default**
- Tokens per task: ~8.77M tokens (500 × 137 × 64 × 2 cameras)
- Size per task: **~70 GB** (2 cameras × 35 GB each)
- Total (3 tasks): **~210 GB** (spatial, object, goal)

**Option B: Full tokens (256 tokens/frame)** - Available via `--save-full-tokens`
- Tokens per task: ~35.1M tokens
- Size per task: **~280 GB**
- Total (3 tasks): **~840 GB**

## 🚀 Usage

### 1. Test on Small Sample (Recommended First)

```bash
cd /home_net/go68xoq/code/vla/lerobot_libero_uq/workspace/lerobot

# Test with just 2 episodes
./uncertainty_quantification/test/test_data_collection.sh
```

### 2. Collect Single Task Type

```bash
# Collect all 500 episodes for spatial task (default)
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial

# Or with explicit parameters
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial \
  --tokens-per-frame 64 \
  --device cuda
```

### 3. Collect Subset of Episodes

```bash
# Use only first 100 episodes (for faster experimentation)
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial \
  --num-episodes 100
```

### 4. Save Full Tokens (No Sampling)

```bash
# Save all 256 tokens per frame (⚠️ Large: ~280 GB per task)
python -m uncertainty_quantification.scripts.collect_rnd_training_data \
  --task-type spatial \
  --save-full-tokens
```

### 5. Collect All Task Types

```bash
# Run full collection for all 3 task types (~210 GB total)
./uncertainty_quantification/scripts/collect_all_tasks.sh
```

## 📋 Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--task-type` | *required* | Task type: `spatial`, `object`, `goal`, `long` |
| `--num-episodes` | `None` (all ~500) | Limit number of episodes |
| `--tokens-per-frame` | `64` | Tokens to sample per camera |
| `--save-full-tokens` | `False` | Save all 256 tokens (overrides sampling) |
| `--libero-dataset-dir` | `third_party/LIBERO/datasets` | Path to LIBERO data |
| `--output-dir` | `uncertainty_quantification/rnd_dataset` | Output directory |
| `--policy-path` | `pi-0-5-preview` | Pi0.5 checkpoint |
| `--device` | `cuda` (if available) | Device for inference |

## 📁 Expected Output Structure

```
uncertainty_quantification/rnd_dataset/
├── spatial/
│   ├── agentview_tokens.pt        # (4.38M, 2048) ~35 GB
│   ├── wrist_tokens.pt            # (4.38M, 2048) ~35 GB
│   └── collection_stats.json      # Metadata
├── object/
│   ├── agentview_tokens.pt
│   ├── wrist_tokens.pt
│   └── collection_stats.json
└── goal/
    ├── agentview_tokens.pt
    ├── wrist_tokens.pt
    └── collection_stats.json
```

### collection_stats.json Example
```json
{
  "total_episodes": 500,
  "total_frames": 68500,
  "total_tokens_agentview": 4384000,
  "total_tokens_wrist": 4384000
}
```

## ⚙️ Technical Details

### Camera Name Mapping
- LIBERO: `agentview_rgb`, `eye_in_hand_rgb`
- Pi0.5: `image`, `image2`
- Output: `agentview`, `wrist`

### Image Processing Pipeline
1. Load from HDF5: `(H, W, C)` uint8 `[0, 255]`
2. Convert to tensor: `(1, C, H, W)` float32 `[0, 1]`
3. Pi0.5 preprocessing: Resize + normalize to `[-1, 1]`
4. SigLIP encoder: `(1, 256, 2048)` embeddings
5. Sample/save: `(64, 2048)` or `(256, 2048)` tokens

## 🎯 What to Do Next

1. **Test the implementation**:
   ```bash
   ./uncertainty_quantification/test/test_data_collection.sh
   ```

2. **Verify test outputs**:
   - Check `uncertainty_quantification/rnd_dataset_test/spatial/`
   - Review `collection_stats.json`
   - Inspect token shapes with PyTorch

3. **Run full collection** (if test passes):
   ```bash
   ./uncertainty_quantification/scripts/collect_all_tasks.sh
   ```

4. **Move to Phase 2**: Train RND models using collected data

## 📝 README Updates

The main README has been updated with:
- ✅ Correct episode counts (500 per task type)
- ✅ Accurate dataset size calculations (~70 GB per task with sampling)
- ✅ Configurable options documentation
- ✅ Usage examples
- ✅ Phase 1 marked as complete

## 🔍 Troubleshooting

**Issue**: Out of memory during collection
- **Solution**: Reduce `--num-episodes` or process task types separately

**Issue**: LIBERO dataset not found
- **Solution**: Check path with `ls third_party/LIBERO/datasets/libero_*/`

**Issue**: Pi0.5 model download fails
- **Solution**: Check internet connection or provide local checkpoint path

**Issue**: CUDA out of memory
- **Solution**: Use `--device cpu` (slower but uses system RAM)

---

**Implementation Date**: February 23, 2026
**Status**: ✅ Phase 1 Complete - Ready for Testing
