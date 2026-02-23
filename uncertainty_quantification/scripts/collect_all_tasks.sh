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
TASK_TYPES=("object")
NUM_EPISODES=""  # Empty = use all episodes (~500 each)
TOKENS_PER_FRAME=""  # Number of tokens to sample (64 recommended, or "" for all 256 tokens)
MAX_FRAMES_PER_CHUNK="2000"  # Frames per chunk file (controls memory usage)
GPU_ID="cuda:0"  # GPU device to use (e.g., "cuda:0", "cuda:1")
OUTPUT_DIR="uncertainty_quantification/rnd_dataset"

# if output_dir already exists, prompt to stop the script to avoid overwriting data
if [ -d "$OUTPUT_DIR" ]; then
    echo "the output directory already exists. Please remove it manually or choose a different output directory."
    exit 1
fi


# Determine collection mode and estimate sizes
NUM_TASKS=${#TASK_TYPES[@]}
if [ -z "$TOKENS_PER_FRAME" ]; then
    COLLECTION_MODE="ALL (256 tokens) - no sampling"
    # With 256 tokens: 2000 frames × 256 tokens × 2 cameras × 2048 × 4 bytes ≈ 8 GB per chunk
    CHUNK_SIZE_GB="~8"
    # Approx 35 chunks per task (68,500 frames / 2000)
    EXPECTED_SIZE_PER_TASK="~280 GB (35 chunks of ~8 GB)"
    TOTAL_SIZE_GB=$((280 * NUM_TASKS))
    TOTAL_SIZE="~${TOTAL_SIZE_GB} GB"
else
    COLLECTION_MODE="$TOKENS_PER_FRAME tokens (sampled)"
    # With 64 tokens: 2000 frames × 64 tokens × 2 cameras × 2048 × 4 bytes ≈ 2 GB per chunk
    CHUNK_SIZE_GB="~2"
    # Approx 35 chunks per task
    EXPECTED_SIZE_PER_TASK="~70 GB (35 chunks of ~2 GB)"
    TOTAL_SIZE_GB=$((70 * NUM_TASKS))
    TOTAL_SIZE="~${TOTAL_SIZE_GB} GB"
fi

echo "Configuration:"
echo "  Task types: ${TASK_TYPES[@]}"
echo "  Episodes per task: ALL (~500)"
echo "  Tokens per frame: $COLLECTION_MODE"
echo "  Max frames per chunk: $MAX_FRAMES_PER_CHUNK"
echo "  Chunk size: $CHUNK_SIZE_GB"
if [ -n "$GPU_ID" ]; then
    echo "  GPU Device: $GPU_ID"
fi
echo "  Expected size per task: $EXPECTED_SIZE_PER_TASK"
echo "  Total expected size: $TOTAL_SIZE (for ${NUM_TASKS} task type(s))"
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
    
    # Build command with all arguments
    CMD="python -m uncertainty_quantification.scripts.collect_rnd_training_data"
    CMD="$CMD --task-type $TASK_TYPE"
    CMD="$CMD --output-dir $OUTPUT_DIR"
    CMD="$CMD --max-frames-per-chunk $MAX_FRAMES_PER_CHUNK"
    
    if [ -n "$GPU_ID" ]; then
        CMD="$CMD --device $GPU_ID"
    fi
    
    if [ -z "$TOKENS_PER_FRAME" ]; then
        # Save all 256 tokens (no sampling)
        CMD="$CMD --save-full-tokens"
    else
        # Sample specified number of tokens
        CMD="$CMD --tokens-per-frame $TOKENS_PER_FRAME"
    fi
    
    eval $CMD
    
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
