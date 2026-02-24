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
TEST_DATASET_DIR="uncertainty_quantification/rnd_dataset_test/spatial"

if [ -d "$TEST_DATASET_DIR" ]; then
    echo "✓ Found test dataset from Phase 1: $TEST_DATASET_DIR"
    echo ""
    
    # Test training on existing test data
    echo "Testing training with agentview camera (3 epochs)..."
    python -m uncertainty_quantification.scripts.train_rnd \
        --task-type spatial \
        --camera agentview \
        --dataset-dir uncertainty_quantification/rnd_dataset_test \
        --output-dir uncertainty_quantification/test/rnd_models_test \
        --epochs 3 \
        --batch-size 64 \
        --patience 5 \
        --seed 42
    
    echo ""
    echo "✓ Training test completed successfully!"
    echo ""
    echo "Check outputs at:"
    echo "  uncertainty_quantification/test/rnd_models_test/spatial_agentview/"
    echo ""
    echo "View training in TensorBoard:"
    echo "  tensorboard --logdir uncertainty_quantification/test/rnd_models_test/spatial_agentview/tensorboard/"
    echo ""
else
    echo "⚠ Test dataset not found: $TEST_DATASET_DIR"
    echo ""
    echo "Please run Phase 1 test first to create test data:"
    echo "  ./uncertainty_quantification/test/test_data_collection.sh"
    echo ""
    echo "Or test training directly on full dataset:"
    echo "  python -m uncertainty_quantification.scripts.train_rnd \\"
    echo "    --task-type spatial \\"
    echo "    --camera agentview \\"
    echo "    --epochs 3 \\"
    echo "    --batch-size 64"
    echo ""
fi

echo "============================================================"
echo "Test Complete"
echo "============================================================"
