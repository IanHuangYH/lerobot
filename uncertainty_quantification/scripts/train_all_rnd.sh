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
#   Set environment variables to control training:
#     EPOCHS=50 ./uncertainty_quantification/scripts/train_all_rnd.sh
#     BATCH_SIZE=128 ./uncertainty_quantification/scripts/train_all_rnd.sh
#     MAX_CACHED_CHUNKS=8 ./uncertainty_quantification/scripts/train_all_rnd.sh
#
#   Combine multiple options:
#     EPOCHS=50 BATCH_SIZE=128 MAX_CACHED_CHUNKS=2 ./uncertainty_quantification/scripts/train_all_rnd.sh
#
#   Custom directories:
#     DATASET_DIR="custom_rnd_dataset" OUTPUT_DIR="custom_models" ./uncertainty_quantification/scripts/train_all_rnd.sh

set -e  # Exit on error

# Configuration
LR=${LR:-1e-4}
PATIENCE=${PATIENCE:-10}
MAX_CACHED_CHUNKS=${MAX_CACHED_CHUNKS:-38}
BATCH_SIZE=256
EPOCHS=50

DATASET_DIR=${DATASET_DIR:-"uncertainty_quantification/rnd_dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"uncertainty_quantification/rnd_save_models"}



# Task types to train on
TASK_TYPES=("object") #("spatial" "object" "goal" "long")
CAMERAS=("agentview" "wrist")

echo "============================================================"
echo "Training All RND Models"
echo "============================================================"
echo "Configuration:"
echo "  Dataset dir: $DATASET_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Patience: $PATIENCE"
echo "  Max cached chunks: $MAX_CACHED_CHUNKS"
echo "============================================================"
echo ""

# Counter for progress tracking
TOTAL_MODELS=$((${#TASK_TYPES[@]} * ${#CAMERAS[@]}))
CURRENT_MODEL=0

# Train each task type (with both cameras in parallel)
for task_type in "${TASK_TYPES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Training Task: $task_type (both cameras in parallel)"
    echo "============================================================"
    echo ""
    
    # Launch both cameras in parallel (background jobs)
    for camera in "${CAMERAS[@]}"; do
        CURRENT_MODEL=$((CURRENT_MODEL + 1))
        
        # Create log file for this training
        LOG_FILE="$OUTPUT_DIR/${task_type}_${camera}_training.log"
        
        echo "Starting training: $task_type - $camera (Model $CURRENT_MODEL/$TOTAL_MODELS)"
        echo "  Log file: $LOG_FILE"
        
        # Train the model in background, redirect output to log file
        (
            python -m uncertainty_quantification.scripts.train_rnd \
                --task-type "$task_type" \
                --camera "$camera" \
                --dataset-dir "$DATASET_DIR" \
                --output-dir "$OUTPUT_DIR" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --lr "$LR" \
                --patience "$PATIENCE" \
                --max-cached-chunks "$MAX_CACHED_CHUNKS" \
                --seed 42 2>&1 | tee "$LOG_FILE"
            
            echo "✓ Completed: $task_type - $camera" | tee -a "$LOG_FILE"
        ) &
    done
    
    echo ""
    echo "Monitor training progress in separate terminals:"
    echo "  tail -f $OUTPUT_DIR/${task_type}_agentview_training.log"
    echo "  tail -f $OUTPUT_DIR/${task_type}_wrist_training.log"
    echo ""
    
    # Wait for both cameras to finish before moving to next task
    echo "Waiting for both cameras to finish training..."
    wait
    
    echo ""
    echo "✓ Task $task_type completed (both cameras)"
    echo ""
done

echo ""
echo "============================================================"
echo "All RND Models Trained Successfully!"
echo "============================================================"
echo "Models saved to: $OUTPUT_DIR/"
echo ""
echo "To view training progress for all models:"
echo "  tensorboard --logdir $OUTPUT_DIR/"
echo ""
echo "Model structure:"
for task_type in "${TASK_TYPES[@]}"; do
    for camera in "${CAMERAS[@]}"; do
        echo "  - ${task_type}_${camera}/model.ckpt"
    done
done
echo ""
