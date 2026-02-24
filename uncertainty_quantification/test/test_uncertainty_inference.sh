#!/bin/bash

# Test script for uncertainty inference integration (Phase 3)
# Tests RND uncertainty prediction on a few LIBERO episodes

set -e  # Exit on error

echo "================================================"
echo "Testing Uncertainty Inference (Phase 3)"
echo "================================================"

# Test configuration
TASK_SUITE="libero_object"  # Test with object task type (matches trained RND)
TASK_IDS='[0]'              # Test on first scene
N_EPISODES=2                # Just 2 episodes for quick test
OUTPUT_DIR="uncertainty_quantification/eval_log/test_rnd_inference"

echo ""
echo "Configuration:"
echo "  Task Suite: $TASK_SUITE"
echo "  Task IDs: $TASK_IDS"
echo "  Episodes: $N_EPISODES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Clean previous test results
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous test results..."
    rm -rf "$OUTPUT_DIR"
fi

# Run evaluation with uncertainty prediction enabled
echo "Running evaluation with uncertainty prediction..."
echo ""

CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --env.type=libero \
    --env.task=$TASK_SUITE \
    --eval.batch_size=$N_EPISODES \
    --eval.n_episodes=$N_EPISODES \
    --policy.path=lerobot/pi05_libero_finetuned \
    --policy.n_action_steps=10 \
    --policy.device=cuda:0 \
    --output_dir=$OUTPUT_DIR \
    --env.max_parallel_tasks=1 \
    --env.task_ids=$TASK_IDS \
    --env.init_states=true \
    --eval.save_uncertainty_maps=true

# Verify outputs
echo ""
echo "================================================"
echo "Verification"
echo "================================================"

UNCERTAINTY_DIR="$OUTPUT_DIR/uncertainty/${TASK_SUITE}_0"

if [ -d "$UNCERTAINTY_DIR" ]; then
    echo "✓ Uncertainty directory created: $UNCERTAINTY_DIR"
    
    NUM_FILES=$(ls -1 "$UNCERTAINTY_DIR"/*.pt 2>/dev/null | wc -l)
    echo "✓ Found $NUM_FILES uncertainty files"
    
    if [ $NUM_FILES -eq $N_EPISODES ]; then
        echo "✓ Correct number of uncertainty files ($N_EPISODES episodes)"
    else
        echo "⚠ Expected $N_EPISODES files, found $NUM_FILES"
    fi
    
    # Check first file structure
    if [ $NUM_FILES -gt 0 ]; then
        FIRST_FILE=$(ls "$UNCERTAINTY_DIR"/*.pt | head -n 1)
        echo ""
        echo "Sample file: $(basename $FIRST_FILE)"
        python -c "
import torch
data = torch.load('$FIRST_FILE')
print('  Keys:', list(data.keys()))
print('  Episode index:', data.get('episode_index'))
print('  Num steps:', data['metadata']['num_steps'])
if data['rollout_steps']:
    step0 = data['rollout_steps'][0]
    print('  Step 0 keys:', list(step0.keys()))
    print('  Step 0 uncertainty keys:', list(step0['uncertainty'].keys()))
"
    fi
else
    echo "❌ Uncertainty directory not found!"
    exit 1
fi

echo ""
echo "================================================"
echo "Test completed successfully!"
echo "================================================"
echo ""
echo "Output location: $OUTPUT_DIR"
echo "Uncertainty files: $UNCERTAINTY_DIR"
echo ""
