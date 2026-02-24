#!/bin/bash
#
# Train all RND models for all task types and cameras.
#
# This script trains 8 RND models total:
# - 4 task types: spatial, object, goal, long
# - 2 cameras per task: agentview, wrist
#
# Each model is trained sequentially to avoid GPU memory issues.
# Training logs and checkpoints are saved to:
#   uncertainty_quantification/rnd_save_models/{task_type}_{camera}/
#
# Usage:
#   ./uncertainty_quantification/scripts/train_all_rnd.sh
#
# Options:
#   Set EPOCHS environment variable to control training length:
#     EPOCHS=50 ./uncertainty_quantification/scripts/train_all_rnd.sh
#
#   Set BATCH_SIZE for memory-constrained systems:
#     BATCH_SIZE=128 ./uncertainty_quantification/scripts/train_all_rnd.sh

set -e  # Exit on error

# Configuration
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-4}
PATIENCE=${PATIENCE:-10}

# Task types to train on
TASK_TYPES=("spatial" "object" "goal" "long")
CAMERAS=("agentview" "wrist")

echo "============================================================"
echo "Training All RND Models"
echo "============================================================"
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Patience: $PATIENCE"
echo "============================================================"
echo ""

# Counter for progress tracking
TOTAL_MODELS=$((${#TASK_TYPES[@]} * ${#CAMERAS[@]}))
CURRENT_MODEL=0

# Train each combination
for task_type in "${TASK_TYPES[@]}"; do
    for camera in "${CAMERAS[@]}"; do
        CURRENT_MODEL=$((CURRENT_MODEL + 1))
        
        echo ""
        echo "============================================================"
        echo "Training Model $CURRENT_MODEL/$TOTAL_MODELS"
        echo "  Task: $task_type"
        echo "  Camera: $camera"
        echo "============================================================"
        echo ""
        
        # Train the model
        python -m uncertainty_quantification.scripts.train_rnd \
            --task-type "$task_type" \
            --camera "$camera" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --patience "$PATIENCE" \
            --seed 42
        
        echo ""
        echo "✓ Completed: $task_type $camera ($CURRENT_MODEL/$TOTAL_MODELS)"
        echo ""
    done
done

echo ""
echo "============================================================"
echo "All RND Models Trained Successfully!"
echo "============================================================"
echo "Models saved to: uncertainty_quantification/rnd_save_models/"
echo ""
echo "To view training progress for all models:"
echo "  tensorboard --logdir uncertainty_quantification/rnd_save_models/"
echo ""
echo "Model structure:"
for task_type in "${TASK_TYPES[@]}"; do
    for camera in "${CAMERAS[@]}"; do
        echo "  - ${task_type}_${camera}/model.ckpt"
    done
done
echo ""
