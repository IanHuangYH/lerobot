#!/bin/bash
#
# Test RND training on a small subset to verify implementation.
#
# This script tests the training pipeline by:
# 1. Training on test data from Phase 1 (if available)
# 2. Or creating a small dummy dataset for testing
# 3. Training for just a few epochs to verify no errors
#
# Usage:
#   ./uncertainty_quantification/test/test_rnd_training.sh

set -e  # Exit on error

echo "============================================================"
echo "Testing RND Training Implementation"
echo "============================================================"
echo ""

# Check if test dataset exists from Phase 1
TEST_DATASET_DIR="uncertainty_quantification/rnd_dataset"
TASK_TYPE="object"
FULL_TEST_DATASET_PATH=$TEST_DATASET_DIR/$TASK_TYPE

OUTPUT_DIR="uncertainty_quantification/train_model_rnd_test"
MAX_CACHED_CHUNKS=${MAX_CACHED_CHUNKS:-38}
BATCH_SIZE=256
EPOCHS=3



if [ -d "$FULL_TEST_DATASET_PATH" ]; then
    echo "✓ Found test dataset from Phase 1: $FULL_TEST_DATASET_PATH"
    echo ""
    
    # Test training on existing test data
    echo "Testing training with agentview camera (3 epochs)..."
    python -m uncertainty_quantification.scripts.train_rnd \
        --task-type $TASK_TYPE \
        --camera agentview \
        --dataset-dir $TEST_DATASET_DIR \
        --output-dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --patience 5 \
        --max-cached-chunks $MAX_CACHED_CHUNKS \
        --seed 42
    
    echo ""
    echo "✓ Training test completed successfully!"
    echo ""
    echo "Check outputs at:"
    echo "  $OUTPUT_DIR/object_agentview/"
    echo ""
    echo "View training in TensorBoard:"
    echo "  tensorboard --logdir $OUTPUT_DIR/object_agentview/tensorboard/"
    echo ""
else
    echo "⚠ Test dataset not found: $TEST_DATASET_DIR"
    echo ""
    echo "Please run Phase 1 test first to create test data:"
    echo "  ./uncertainty_quantification/test/test_data_collection.sh"
    echo ""
    echo "Or test training directly on full dataset:"
    echo "  python -m uncertainty_quantification.scripts.train_rnd \\"
    echo "    --task-type object \\"
    echo "    --camera agentview \\"
    echo "    --epochs 3 \\"
    echo "    --batch-size 64"
    echo ""
fi

echo "============================================================"
echo "Test Complete"
echo "============================================================"
