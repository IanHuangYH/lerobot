#!/bin/bash
#
# Train all RND models with tmux split-screen monitoring.
#
# This script uses tmux to display training(s) with live monitoring.
# Each task type can run both cameras in parallel or a single camera.
#
# Usage:
#   ./uncertainty_quantification/scripts/train_all_rnd_tmux.sh
#
# Options:
#   CAMERA_MODE=both|agentview|wrist (default: both)
#   EPOCHS=50 (default: 150)
#   BATCH_SIZE=128 (default: 256)
#   MAX_CACHED_CHUNKS=38 (default: 38)
#
# Examples:
#   # Train both cameras in split view (default)
#   ./uncertainty_quantification/scripts/train_all_rnd_tmux.sh
#
#   # Train only agentview
#   CAMERA_MODE=agentview ./uncertainty_quantification/scripts/train_all_rnd_tmux.sh
#
#   # Train only wrist view with custom epochs
#   CAMERA_MODE=wrist EPOCHS=50 ./uncertainty_quantification/scripts/train_all_rnd_tmux.sh

set -e

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    echo "  macOS: brew install tmux"
    exit 1
fi

# Configuration
LR=${LR:-1e-4}
PATIENCE=${PATIENCE:-10}
MAX_CACHED_CHUNKS=${MAX_CACHED_CHUNKS:-38}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-150}
CAMERA_MODE=${CAMERA_MODE:-"both"}  # Options: "both", "agentview", "wrist"

DATASET_DIR=${DATASET_DIR:-"uncertainty_quantification/rnd_dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"uncertainty_quantification/rnd_save_models_long_150"}

# Task types to train on
TASK_TYPES=("object") #("spatial" "object" "goal" "long")

# Validate camera mode
if [[ "$CAMERA_MODE" != "both" && "$CAMERA_MODE" != "agentview" && "$CAMERA_MODE" != "wrist" ]]; then
    echo "Error: CAMERA_MODE must be 'both', 'agentview', or 'wrist'"
    exit 1
fi

echo "============================================================"
echo "Training All RND Models with tmux Monitoring"
echo "============================================================"
echo "Configuration:"
echo "  Dataset dir: $DATASET_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Camera mode: $CAMERA_MODE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Patience: $PATIENCE"
echo "  Max cached chunks: $MAX_CACHED_CHUNKS"
echo "============================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Train each task type with tmux monitoring
for task_type in "${TASK_TYPES[@]}"; do
    echo ""
    echo "============================================================"
    if [[ "$CAMERA_MODE" == "both" ]]; then
        echo "Training Task: $task_type (both cameras in parallel)"
    else
        echo "Training Task: $task_type (camera: $CAMERA_MODE)"
    fi
    echo "============================================================"
    echo ""
    
    SESSION_NAME="rnd_train_${task_type}"
    LOG_AGENTVIEW="$OUTPUT_DIR/${task_type}_agentview_training.log"
    LOG_WRIST="$OUTPUT_DIR/${task_type}_wrist_training.log"
    
    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    
    if [[ "$CAMERA_MODE" == "both" ]]; then
        # Create tmux session with split panes for both cameras
        echo "Creating tmux session: $SESSION_NAME"
        echo "  Left pane: agentview training"
        echo "  Right pane: wrist training"
        echo ""
        
        # Create new tmux session (detached)
        tmux new-session -d -s "$SESSION_NAME"
        
        # Split window vertically (left/right panes)
        tmux split-window -h -t "$SESSION_NAME"
        
        # Left pane: Launch agentview training
        tmux send-keys -t "$SESSION_NAME:0.0" "
echo '=== Training: ${task_type} - agentview ==='
python -m uncertainty_quantification.scripts.train_rnd \
    --task-type '$task_type' \
    --camera agentview \
    --dataset-dir '$DATASET_DIR' \
    --output-dir '$OUTPUT_DIR' \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --max-cached-chunks $MAX_CACHED_CHUNKS \
    --seed 42 2>&1 | tee '$LOG_AGENTVIEW'
echo ''
echo '✓ Agentview training completed!'
echo 'Press Ctrl+C to close this pane or wait for both to finish.'
" C-m
        
        # Right pane: Launch wrist training
        tmux send-keys -t "$SESSION_NAME:0.1" "
echo '=== Training: ${task_type} - wrist ==='
python -m uncertainty_quantification.scripts.train_rnd \
    --task-type '$task_type' \
    --camera wrist \
    --dataset-dir '$DATASET_DIR' \
    --output-dir '$OUTPUT_DIR' \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --max-cached-chunks $MAX_CACHED_CHUNKS \
    --seed 42 2>&1 | tee '$LOG_WRIST'
echo ''
echo '✓ Wrist training completed!'
echo 'Press Ctrl+C to close this pane or wait for both to finish.'
" C-m
        
        # Add status bar at bottom
        tmux set-option -t "$SESSION_NAME" status-right "Task: $task_type | Epochs: $EPOCHS | Batch: $BATCH_SIZE"
        
    elif [[ "$CAMERA_MODE" == "agentview" ]]; then
        # Single pane for agentview only
        echo "Creating tmux session: $SESSION_NAME"
        echo "  Single pane: agentview training"
        echo ""
        
        tmux new-session -d -s "$SESSION_NAME"
        
        tmux send-keys -t "$SESSION_NAME:0.0" "
echo '=== Training: ${task_type} - agentview ==='
python -m uncertainty_quantification.scripts.train_rnd \
    --task-type '$task_type' \
    --camera agentview \
    --dataset-dir '$DATASET_DIR' \
    --output-dir '$OUTPUT_DIR' \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --max-cached-chunks $MAX_CACHED_CHUNKS \
    --seed 42 2>&1 | tee '$LOG_AGENTVIEW'
echo ''
echo '✓ Agentview training completed!'
" C-m
        
        tmux set-option -t "$SESSION_NAME" status-right "Task: $task_type | Camera: agentview | Epochs: $EPOCHS"
        
    else  # wrist only
        # Single pane for wrist only
        echo "Creating tmux session: $SESSION_NAME"
        echo "  Single pane: wrist training"
        echo ""
        
        tmux new-session -d -s "$SESSION_NAME"
        
        tmux send-keys -t "$SESSION_NAME:0.0" "
echo '=== Training: ${task_type} - wrist ==='
python -m uncertainty_quantification.scripts.train_rnd \
    --task-type '$task_type' \
    --camera wrist \
    --dataset-dir '$DATASET_DIR' \
    --output-dir '$OUTPUT_DIR' \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --max-cached-chunks $MAX_CACHED_CHUNKS \
    --seed 42 2>&1 | tee '$LOG_WRIST'
echo ''
echo '✓ Wrist training completed!'
" C-m
        
        tmux set-option -t "$SESSION_NAME" status-right "Task: $task_type | Camera: wrist | Epochs: $EPOCHS"
    fi
    
    echo "tmux session created!"
    echo ""
    echo "Attaching to tmux session..."
    if [[ "$CAMERA_MODE" == "both" ]]; then
        echo "  - Left pane: agentview"
        echo "  - Right pane: wrist"
    else
        echo "  - Single pane: $CAMERA_MODE"
    fi
    echo ""
    echo "tmux controls:"
    if [[ "$CAMERA_MODE" == "both" ]]; then
        echo "  Ctrl+b then ← → : Switch between panes"
    fi
    echo "  Ctrl+b then d   : Detach (trainings continue in background)"
    echo "  tmux attach -t $SESSION_NAME : Re-attach later"
    echo ""
    echo "Press Enter to attach to session..."
    read
    
    # Attach to session (blocks until detached or trainings finish)
    tmux attach -t "$SESSION_NAME"
    
    # Wait for trainings to actually finish
    echo ""
    if [[ "$CAMERA_MODE" == "both" ]]; then
        echo "Waiting for both trainings to complete..."
        while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
            # Check if both trainings are done by looking for completion messages
            if grep -q "✓ Agentview training completed!" "$LOG_AGENTVIEW" 2>/dev/null && \
               grep -q "✓ Wrist training completed!" "$LOG_WRIST" 2>/dev/null; then
                echo "Both trainings completed!"
                tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
                break
            fi
            sleep 5
        done
        echo ""
        echo "✓ Task $task_type completed (both cameras)"
        echo "  Logs saved:"
        echo "    - $LOG_AGENTVIEW"
        echo "    - $LOG_WRIST"
    elif [[ "$CAMERA_MODE" == "agentview" ]]; then
        echo "Waiting for agentview training to complete..."
        while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
            if grep -q "✓ Agentview training completed!" "$LOG_AGENTVIEW" 2>/dev/null; then
                echo "Training completed!"
                tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
                break
            fi
            sleep 5
        done
        echo ""
        echo "✓ Task $task_type completed (agentview)"
        echo "  Log saved: $LOG_AGENTVIEW"
    else
        echo "Waiting for wrist training to complete..."
        while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
            if grep -q "✓ Wrist training completed!" "$LOG_WRIST" 2>/dev/null; then
                echo "Training completed!"
                tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
                break
            fi
            sleep 5
        done
        echo ""
        echo "✓ Task $task_type completed (wrist)"
        echo "  Log saved: $LOG_WRIST"
    fi
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
