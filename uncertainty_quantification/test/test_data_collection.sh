#!/bin/bash
# Test data collection on a small sample before running full collection

set -e  # Exit on error

# Initialize conda for module-based systems
if command -v module &> /dev/null; then
    module load Anaconda
    source /sq/shares/opt/anaconda3/2025.12.1/etc/profile.d/conda.sh
    conda deactivate
    conda activate lerobot
    echo "✓ Conda environment activated: $(conda info --envs | grep '*')"
fi

echo "=================================================="
echo "Testing RND Data Collection (Small Sample)"
echo "=================================================="
echo ""

# Configuration
TASK_TYPE="spatial"
NUM_EPISODES=2  # Test with just 2 episodes
TOKENS_PER_FRAME=64
OUTPUT_DIR="uncertainty_quantification/rnd_dataset_test"
GPU_ID="cuda:0"  # Uncomment to specify GPU (e.g., cuda:0, cuda:1, cuda:2)

echo "Test Configuration:"
echo "  Task type: $TASK_TYPE"
echo "  Episodes: $NUM_EPISODES"
echo "  Tokens per frame: $TOKENS_PER_FRAME"
echo "  Output: $OUTPUT_DIR"
echo ""

# Clean up previous test results
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning up previous test results..."
    rm -rf "$OUTPUT_DIR"
fi

# Run collection
echo "Running data collection..."
echo ""

if [ -n "$GPU_ID" ]; then
    python -m uncertainty_quantification.scripts.collect_rnd_training_data \
        --task-type "$TASK_TYPE" \
        --num-episodes "$NUM_EPISODES" \
        --tokens-per-frame "$TOKENS_PER_FRAME" \
        --output-dir "$OUTPUT_DIR" \
        --device "$GPU_ID"
else
    python -m uncertainty_quantification.scripts.collect_rnd_training_data \
        --task-type "$TASK_TYPE" \
        --num-episodes "$NUM_EPISODES" \
        --tokens-per-frame "$TOKENS_PER_FRAME" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "=================================================="
echo "Test Complete!"
echo "=================================================="
echo ""

# Validate outputs
echo "Validating outputs..."

AGENTVIEW_FILE="$OUTPUT_DIR/$TASK_TYPE/agentview_tokens.pt"
WRIST_FILE="$OUTPUT_DIR/$TASK_TYPE/wrist_tokens.pt"
STATS_FILE="$OUTPUT_DIR/$TASK_TYPE/collection_stats.json"

if [ -f "$AGENTVIEW_FILE" ]; then
    echo "✅ Agentview tokens file created"
    ls -lh "$AGENTVIEW_FILE"
else
    echo "❌ Agentview tokens file missing!"
    exit 1
fi

if [ -f "$WRIST_FILE" ]; then
    echo "✅ Wrist tokens file created"
    ls -lh "$WRIST_FILE"
else
    echo "❌ Wrist tokens file missing!"
    exit 1
fi

if [ -f "$STATS_FILE" ]; then
    echo "✅ Statistics file created"
    echo "Statistics:"
    cat "$STATS_FILE"
else
    echo "❌ Statistics file missing!"
    exit 1
fi

echo ""
echo "=================================================="
echo "All tests passed! ✅"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review the test outputs in: $OUTPUT_DIR"
echo "2. If everything looks good, run full collection:"
echo "   python -m uncertainty_quantification.scripts.collect_rnd_training_data \\"
echo "     --task-type spatial"
echo ""
