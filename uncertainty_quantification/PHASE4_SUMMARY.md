# Phase 4: Uncertainty Visualization - Implementation Summary

**Implementation Date**: February 24, 2026  
**Status**: ✅ Complete

---

## 🎯 Overview

Phase 4 implements comprehensive visualization tools for uncertainty quantification results. The visualization suite enables inspection, debugging, and analysis of RND-based uncertainty predictions during LIBERO task evaluation.

**Key Design**: Following the existing attention visualization pattern, but adapted for uncertainty's simpler data structure (no layers/denoising steps/action indices).

---

## ✅ Completed Tasks

### 1. **Visualization Module Structure** ✅
- **Folder Created**: `uncertainty_quantification/visualization/`
- **Files**:
  - `__init__.py` - Package initialization with exports
  - `verify_uncertainty_video_alignment.py` - Main visualization script
  - `visualize_uncertainty_timeline.py` - Timeline plotting

### 2. **Main Verification Script** ✅
**File**: `uncertainty_quantification/visualization/verify_uncertainty_video_alignment.py`

**Purpose**: Create 2x2 grid visualizations with uncertainty overlays

**Key Functions**:
- `load_uncertainty_data()` - Load episode_XXXXX_uncertainty.pt file
- `extract_spatial_map()` - Get (16, 16) heatmap for specific camera
- `load_video_frame()` - Extract frame from MP4 video
- `split_concatenated_frame()` - Separate agentview (left) and wrist (right)
- `create_uncertainty_heatmap()` - Convert spatial map to colored RGBA heatmap
- `overlay_heatmap_on_image()` - Blend heatmap on video frame with transparency
- `add_text_to_image()` - Add text overlays with background
- `create_visualization_grid()` - Main 2x2 grid generator

**Output Format** (2x2 Grid):
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

**Features**:
- ✅ Uncertainty scores displayed on top row
- ✅ Configurable colormap (default: "viridis")
- ✅ Adjustable overlay transparency (default: 0.55)
- ✅ Overall uncertainty + timestep at bottom

### 3. **Batch Runner Script** ✅
**File**: `uncertainty_quantification/scripts/run_verify_uncertainty_video_alignment.sh`

**Purpose**: Automate visualization for multiple episodes/tasks/timesteps

**Features**:
- Loop over task IDs and episodes
- Configurable timestep selection (e.g., "0 10 20 30 50 100")
- File existence checks before processing
- Progress reporting

**Configuration Variables**:
```bash
EVAL_FOLDER="with_uncertainty"    # Which evaluation run
EVAL_SCENE_INDEX=0                # Max task ID (0-9)
EVAL_TASK_AMOUNT=9                # Max episode per task
TIMESTEPS="0 10 20 30 ..."       # Timesteps to visualize
COLORMAP="viridis"                # Heatmap colormap
ALPHA=0.55                        # Overlay transparency
```

### 4. **Timeline Visualization** ✅
**File**: `uncertainty_quantification/visualization/visualize_uncertainty_timeline.py`

**Purpose**: Plot uncertainty trends over episode duration

**Key Functions**:
- `load_episode_uncertainties()` - Extract all scores from episode .pt file
- `plot_uncertainty_timeline()` - Single-line plot with statistics
- `plot_camera_comparison()` - Dual-line plot (agentview vs wrist)
- `plot_multi_episode_comparison()` - Compare multiple episodes (optional)

**Plot Types**:

1. **Overall Timeline**:
   - Single line showing overall uncertainty vs timestep
   - Mean line (red dashed)
   - Max uncertainty point highlighted
   - Grid and legend

2. **Camera Comparison**:
   - Three lines: overall, agentview, wrist
   - Mean lines for each camera
   - Color-coded legend with statistics

3. **Multi-Episode Comparison** (optional):
   - Top subplot: Overall uncertainty for all episodes
   - Bottom subplot: Camera-specific trends
   - Color-coded by episode

**Features**:
- ✅ Automatic statistics (mean, max, location of max)
- ✅ Configurable plot types (overall, camera, all)
- ✅ Optional multi-episode comparison mode
- ✅ High-resolution outputs (150 DPI)

---

## 📁 File Structure

```
uncertainty_quantification/
├── visualization/                           # NEW
│   ├── __init__.py
│   ├── verify_uncertainty_video_alignment.py
│   └── visualize_uncertainty_timeline.py
├── scripts/
│   ├── run_verify_uncertainty_video_alignment.sh  # NEW
│   ├── collect_rnd_training_data.py        # Existing (Phase 1)
│   ├── train_rnd.py                        # Existing (Phase 2)
│   └── ...
├── PHASE4_SUMMARY.md                       # This file
├── PHASE3_SUMMARY.md                       # Phase 3 docs
├── PHASE2_SUMMARY.md                       # Phase 2 docs
└── PHASE1_SUMMARY.md                       # Phase 1 docs

eval_logs/{eval_name}/uncertainty/{task}/
├── episode_XXXXX_uncertainty.pt            # Phase 3 output
├── verification/                           # NEW (Phase 4)
│   ├── episode_00000/
│   │   ├── timestep_000_grid.png
│   │   ├── timestep_010_grid.png
│   │   └── ...
│   ├── episode_00001/
│   │   └── ...
│   └── ...
└── timelines/                              # NEW (Phase 4)
    ├── episode_00000_uncertainty_timeline.png
    ├── episode_00000_uncertainty_camera_comparison.png
    ├── multi_episode_comparison.png
    └── ...
```

---

## 🚀 Usage

### **1. Single Episode Verification**

```bash
cd /home_net/go68xoq/code/vla/lerobot_libero_uq/workspace/lerobot

# Visualize specific timesteps for one episode
python -m uncertainty_quantification.visualization.verify_uncertainty_video_alignment \
    --uncertainty_path eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00003_uncertainty.pt \
    --video_path eval_logs/with_uncertainty/videos/libero_object_0/eval_episode_00003.mp4 \
    --output_path eval_logs/with_uncertainty/uncertainty/libero_object_0/verification \
    --timesteps 0 10 20 50 100 \
    --colormap viridis \
    --alpha 0.55 \
    --episode_id 3
```

**Output**: `verification/episode_00003/timestep_XXX_grid.png`

### **2. Batch Verification (Multiple Episodes)**

```bash
# Verify all episodes in evaluation run
./uncertainty_quantification/scripts/run_verify_uncertainty_video_alignment.sh

# Or customize settings:
# Edit the script to change EVAL_FOLDER, EVAL_SCENE_INDEX, TIMESTEPS, etc.
```

**Output**: Verification images for all processed episodes

### **3. Timeline Visualization (Single Episode)**

```bash
# Generate timeline plots for one episode
python -m uncertainty_quantification.visualization.visualize_uncertainty_timeline \
    --uncertainty_paths eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00003_uncertainty.pt \
    --output_dir eval_logs/with_uncertainty/uncertainty/libero_object_0/timelines \
    --plot_type all
```

**Output**: 
- `episode_00003_uncertainty_timeline.png` (overall)
- `episode_00003_uncertainty_camera_comparison.png` (cameras)

### **4. Timeline Visualization (Multiple Episodes)**

```bash
# Compare multiple episodes
python -m uncertainty_quantification.visualization.visualize_uncertainty_timeline \
    --uncertainty_paths \
        eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00000_uncertainty.pt \
        eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00001_uncertainty.pt \
        eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00005_uncertainty.pt \
    --output_dir eval_logs/with_uncertainty/uncertainty/libero_object_0/timelines \
    --episode_names "Success-1" "Success-2" "Failed-1" \
    --compare_episodes \
    --plot_type all
```

**Output**: Individual timelines + `multi_episode_comparison.png`

### **5. Quick Test (Minimal Setup)**

```bash
# Test on first episode with few timesteps
python -m uncertainty_quantification.visualization.verify_uncertainty_video_alignment \
    --uncertainty_path eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00000_uncertainty.pt \
    --video_path eval_logs/with_uncertainty/videos/libero_object_0/eval_episode_00000.mp4 \
    --output_path eval_logs/with_uncertainty/uncertainty/libero_object_0/verification \
    --timesteps 0 50 100 \
    --episode_id 0
```

---

## 📋 Command-Line Options

### **verify_uncertainty_video_alignment.py**

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--uncertainty_path` | str | ✅ | - | Path to episode_XXXXX_uncertainty.pt |
| `--video_path` | str | ✅ | - | Path to eval_episode_XXXXX.mp4 |
| `--output_path` | str | ✅ | - | Output directory for images |
| `--timesteps` | int+ | ❌ | `[0, 50, 100]` | Timesteps to visualize |
| `--colormap` | str | ❌ | `viridis` | Matplotlib colormap |
| `--alpha` | float | ❌ | `0.55` | Overlay transparency (0-1) |
| `--episode_id` | int | ❌ | `None` | Episode ID (for folder organization) |

### **visualize_uncertainty_timeline.py**

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--uncertainty_paths` | str+ | ✅ | - | Path(s) to uncertainty .pt file(s) |
| `--output_dir` | str | ✅ | - | Output directory for plots |
| `--episode_names` | str+ | ❌ | Auto | Episode names for legend |
| `--plot_type` | str | ❌ | `all` | Plot type: overall, camera, all |
| `--compare_episodes` | flag | ❌ | `False` | Create multi-episode comparison |

---

## 🎨 Visualization Examples

### **Example 1: 2x2 Grid Snapshot**

**Input**:
- Timestep: 50
- Overall uncertainty: 0.117
- Agentview: 0.145
- Wrist: 0.089

**Output**: `timestep_050_grid.png`

```
┌─────────────────────────┬─────────────────────────┐
│  [Video frame]          │  [Video frame]          │
│  + Heatmap overlay      │  + Heatmap overlay      │
│  Agentview: 0.1450      │  Wrist: 0.0890          │
├─────────────────────────┼─────────────────────────┤
│  [Pure heatmap]         │  [Pure heatmap]         │
│  Agentview Heatmap      │  Wrist Heatmap          │
└─────────────────────────┴─────────────────────────┘
       Overall: 0.1170 | Timestep: 50
```

### **Example 2: Overall Timeline**

**Features**:
- Blue line: Overall uncertainty over time
- Red dashed: Mean uncertainty (0.112)
- Red dot: Max uncertainty (0.234 @ t=67)
- Grid and legend

**Interpretation**: Identifies uncertainty spikes during episode

### **Example 3: Camera Comparison**

**Features**:
- Black line: Overall uncertainty
- Orange line: Agentview uncertainty
- Green line: Wrist uncertainty
- Dashed lines: Means for each

**Interpretation**: Shows which camera contributes more to uncertainty

---

## 🔍 Key Design Decisions

### **1. Colormap Choice: "viridis"**
- **Why**: Better accessibility (colorblind-friendly)
- **Alternatives**: `plasma`, `inferno`, `coolwarm`, `hot`
- **Configurable**: Users can override via `--colormap`

### **2. Text Overlays on Top Row**
- **Why**: Immediate visibility of uncertainty scores
- **Implementation**: Black background with white text for readability
- **Location**: Top of each quadrant + bottom for overall

### **3. Separate Pure Heatmaps**
- **Why**: Enables inspection of spatial patterns without video distraction
- **Use case**: Compare heatmaps across timesteps to see pattern evolution

### **4. No Full Video Generation (Task 4 Skipped)**
- **Reason**: Snapshots sufficient for debugging/analysis
- **Benefit**: Faster processing, lower storage
- **Future**: Can add if needed for presentations

### **5. Modular Visualization Functions**
- **Why**: Reusable components for custom analyses
- **Benefit**: Users can import functions for custom plots
- **Example**: `from uncertainty_quantification.visualization import create_uncertainty_heatmap`

---

## ⚙️ Technical Details

### **Data Flow**

```
episode_XXXXX_uncertainty.pt
    ↓
load_uncertainty_data(timestep=50)
    ↓
Extract:
  - overall: 0.117
  - agentview: 0.145
  - wrist: 0.089
  - spatial_maps: {
      'agentview': (16, 16),
      'wrist': (16, 16)
    }
    ↓
Load video frame (50)
    ↓
Split into agentview, wrist
    ↓
For each camera:
  1. Upsample (16, 16) → (224, 224)
  2. Apply colormap → RGBA heatmap
  3. Overlay on video frame with alpha=0.55
  4. Add text overlay (score)
    ↓
Combine into 2x2 grid
    ↓
Add overall info at bottom
    ↓
Save as PNG
```

### **Heatmap Processing Pipeline**

1. **Input**: Spatial map (16, 16) - raw RND prediction errors
2. **Upsampling**: Bilinear interpolation to (224, 224)
3. **Normalization**: Scale to [0, 1] based on max value
4. **Colormap**: Apply matplotlib colormap → RGBA (224, 224, 4)
5. **Alpha blending**: Combine with video frame
6. **Output**: RGB image (224, 224, 3)

### **Timeline Statistics**

For each episode, the timeline plots compute:
- **Mean uncertainty**: Average across all timesteps
- **Max uncertainty**: Highest value and its timestep
- **Per-camera trends**: Separate statistics for agentview/wrist

**Use cases**:
- Identify when uncertainty spikes (potential failure points)
- Compare uncertainty patterns between successful/failed episodes
- Detect systematic biases (e.g., wrist always higher than agentview)

---

## 🧪 Testing

### **Test Checklist**

Before running on full dataset:

1. **Single episode test**:
   ```bash
   python -m uncertainty_quantification.visualization.verify_uncertainty_video_alignment \
       --uncertainty_path eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00000_uncertainty.pt \
       --video_path eval_logs/with_uncertainty/videos/libero_object_0/eval_episode_00000.mp4 \
       --output_path eval_logs/test_viz \
       --timesteps 0 10 20 \
       --episode_id 0
   ```
   
   **Expected**: 3 PNG files in `eval_logs/test_viz/episode_00000/`

2. **Timeline test**:
   ```bash
   python -m uncertainty_quantification.visualization.visualize_uncertainty_timeline \
       --uncertainty_paths eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00000_uncertainty.pt \
       --output_dir eval_logs/test_viz \
       --plot_type all
   ```
   
   **Expected**: 2 PNG files (timeline + camera comparison)

3. **Batch test** (small scale):
   - Edit `run_verify_uncertainty_video_alignment.sh`
   - Set `EVAL_TASK_AMOUNT=1` (process only 2 episodes)
   - Run script
   
   **Expected**: Verification images for episodes 0-1

### **Common Issues & Solutions**

**Issue**: "Timestep X not found"
- **Cause**: Requested timestep exceeds episode length
- **Solution**: Check `metadata.num_steps` in uncertainty .pt file

**Issue**: Video frame mismatch
- **Cause**: Video FPS mismatch or trimmed video
- **Solution**: Verify video has same frame count as uncertainty timesteps

**Issue**: Blank heatmaps
- **Cause**: All uncertainty values are zero (or very small)
- **Solution**: Check RND models loaded correctly during evaluation

---

## 📊 Integration with Existing Pipeline

### **Phase 3 Output → Phase 4 Input**

```
lerobot-eval (Phase 3)
    ↓ Saves
episode_XXXXX_uncertainty.pt
    ↓ Input to
verify_uncertainty_video_alignment.py (Phase 4)
    ↓ Generates
timestep_XXX_grid.png
```

### **Workflow for Analysis**

1. **Run evaluation** with uncertainty enabled:
   ```bash
   lerobot-eval --eval.save_uncertainty_maps=true
   ```

2. **Generate snapshots** at key timesteps:
   ```bash
   ./uncertainty_quantification/scripts/run_verify_uncertainty_video_alignment.sh
   ```

3. **Create timelines** for trend analysis:
   ```bash
   python -m uncertainty_quantification.visualization.visualize_uncertainty_timeline \
       --uncertainty_paths eval_logs/.../episode_*.pt \
       --output_dir eval_logs/.../timelines \
       --compare_episodes
   ```

4. **Inspect results**:
   - Review snapshots to identify spatial patterns
   - Analyze timelines for uncertainty spikes
   - Compare successful vs failed episodes

---

## 🔄 Future Enhancements (Deferred)

### **Phase 4.5 (Optional Future Work)**

1. **Full Video Generation**:
   - Generate MP4 with uncertainty overlays for entire episode
   - Similar to attention video generation
   - Use case: Presentations, debugging long episodes

2. **Statistical Analysis (Phase 5)**:
   - Correlation between uncertainty and task success
   - ROC curves for failure prediction
   - Threshold optimization
   - See Phase 1 README for details

3. **Attention + Uncertainty Comparison**:
   - 3x2 grid: attention (top row) + uncertainty (bottom row)
   - Analyze correlation between attention patterns and uncertainty
   - Identify if high uncertainty regions align with low attention

4. **Interactive Visualization**:
   - Web-based viewer (Plotly/Bokeh)
   - Scrub through timesteps dynamically
   - Click to show detailed statistics

---

## 📚 Dependencies

All visualization scripts use standard scientific Python libraries:

```python
# Core
import torch
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import zoom

# Standard library
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
```

**No new dependencies** added beyond existing lerobot requirements.

---

## 📝 Summary

### **What Was Implemented**

✅ **Task 1**: Main verification script with 2x2 grid visualization  
✅ **Task 2**: Batch runner shell script for automation  
✅ **Task 3**: Timeline plotting for trend analysis  
✅ **Task 6**: Comprehensive documentation (this file)  
❌ **Task 4**: Full video generation (deferred - snapshots sufficient)  
❌ **Task 5**: Statistical analysis (deferred to future phase)

### **Key Features**

- 🎨 **2x2 Grid Visualization**: Overlays + pure heatmaps with scores
- 📈 **Timeline Plots**: Overall, camera comparison, multi-episode
- 🔧 **Batch Processing**: Automate visualization for many episodes
- 🎯 **Configurable**: Colormap, transparency, timesteps
- 📊 **Statistics**: Mean, max, per-camera trends

### **Output Locations**

```
eval_logs/{eval_name}/uncertainty/{task}/
├── verification/episode_XXXXX/     # Snapshots
└── timelines/                      # Timeline plots
```

### **Total Files Created**

1. `uncertainty_quantification/visualization/__init__.py`
2. `uncertainty_quantification/visualization/verify_uncertainty_video_alignment.py`
3. `uncertainty_quantification/visualization/visualize_uncertainty_timeline.py`
4. `uncertainty_quantification/scripts/run_verify_uncertainty_video_alignment.sh`
5. `uncertainty_quantification/PHASE4_SUMMARY.md` (this file)

---

**Implementation Complete**: February 24, 2026  
**Status**: ✅ Ready for Testing and Use  
**Next Steps**: Test on sample data, then run on full evaluation results
