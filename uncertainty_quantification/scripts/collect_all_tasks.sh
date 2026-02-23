#!/bin/bash
# Collect RND training data for all LIBERO task types

set -e  # Exit on error

echo "=================================================="
echo "RND Training Data Collection - All Task Types"
echo "=================================================="
echo ""

# Initialize conda for module-based systems
if command -v module &> /dev/null; then
    module load Anaconda
    source /sq/shares/opt/anaconda3/2025.12.1/etc/profile.d/conda.sh
    conda deactivate
    conda activate lerobot
    echo "✓ Conda environment activated: $(conda info --envs | grep '*')"
fi

# Configuration
TASK_TYPES=("spatial" "object" "goal")
NUM_EPISODES=""  # Empty = use all episodes (~500 each)
TOKENS_PER_FRAME=""  # Empty = save all 256 tokens, or set a number (e.g., 64) to sample
GPU_ID="cuda:0"  # Empty = use default cuda, or set to specific GPU (e.g., "cuda:0", "cuda:1")
OUTPUT_DIR="uncertainty_quantification/rnd_dataset"

# if output_dir already exists, prompt to stop the script to avoid overwriting data
if [ -d "$OUTPUT_DIR" ]; then
    echo "the output directory already exists. Please remove it manually or choose a different output directory."
    exit 1
fi


# Determine collection mode
if [ -z "$TOKENS_PER_FRAME" ]; then
    COLLECTION_MODE="ALL (256) - no sampling"
    EXPECTED_SIZE_PER_TASK="~280 GB"
    TOTAL_SIZE="~840 GB"
else
    COLLECTION_MODE="$TOKENS_PER_FRAME (sampled)"
    # Rough estimate: 280 GB / 4 = 70 GB for 64 tokens
    SIZE_FACTOR=$(echo "scale=0; 280 * $TOKENS_PER_FRAME / 256" | bc)
    EXPECTED_SIZE_PER_TASK="~${SIZE_FACTOR} GB"
    TOTAL_SIZE="~$(echo "$SIZE_FACTOR * 3" | bc) GB"
fi

echo "Configuration:"
echo "  Task types: ${TASK_TYPES[@]}"
echo "  Episodes per task: ALL (~500)"
echo "  Tokens per frame: $COLLECTION_MODE"
if [ -n "$GPU_ID" ]; then
    echo "  GPU Device: $GPU_ID"
fi
echo "  Expected size: $EXPECTED_SIZE_PER_TASK per task type"
echo "  Total expected size: $TOTAL_SIZE (for 3 task types)"
echo "  Output dir: $OUTPUT_DIR"
echo ""

read -p "Continue with full data collection? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting data collection..."
echo ""

# Collect data for each task type
for TASK_TYPE in "${TASK_TYPES[@]}"; do
    echo "=================================================="
    echo "Collecting: $TASK_TYPE"
    echo "=================================================="
    
    if [ -z "$TOKENS_PER_FRAME" ]; then
        # Save all 256 tokens (no sampling)
        python -m uncertainty_quantification.scripts.collect_rnd_training_data \
            --task-type "$TASK_TYPE" \
            --save-full-tokens \
            --output-dir "$OUTPUT_DIR"
    else
        # Sample specified number of tokens
        python -m uncertainty_quantification.scripts.collect_rnd_training_data \
            --task-type "$TASK_TYPE" \
            --tokens-per-frame "$TOKENS_PER_FRAME" \
            --output-dir "$OUTPUT_DIR"
    fi
    
    echo ""
    echo "✅ Completed: $TASK_TYPE"
    echo ""
done

echo "=================================================="
echo "All Data Collection Complete!"
echo "=================================================="
echo ""

# Show dataset sizes
echo "Dataset sizes:"
for TASK_TYPE in "${TASK_TYPES[@]}"; do
    echo ""
    echo "$TASK_TYPE:"
    du -h uncertainty_quantification/rnd_dataset/$TASK_TYPE/* 2>/dev/null || echo "  (not found)"
done

echo ""
echo "Total size:"
du -sh uncertainty_quantification/rnd_dataset/ 2>/dev/null || echo "(not found)"

echo ""
echo "Next step: Train RND models (Phase 2)"
echo ""
