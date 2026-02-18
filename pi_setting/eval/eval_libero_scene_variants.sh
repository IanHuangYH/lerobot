#!/bin/bash
# Evaluate a policy on LIBERO scene variants.
#
# This script evaluates scene variants (different object layouts) for multiple tasks.
# Supports parallel execution of multiple tasks to improve throughput.
#
# FOLDER STRUCTURE:
#   eval_logs/OUTPUTS_DIR/                          <- All results (Level 1)
#   ├── eval_info.json                              <- Summary of all tasks
#   ├── attention/
#   │   ├── libero_object_0/                        <- Task ID 0 (Level 2)
#   │   │   ├── episode_00000_attention.pt          <- Variant 0 (Level 3)
#   │   │   ├── episode_00001_attention.pt          <- Variant 1
#   │   │   └── episode_00002_attention.pt          <- Variant 2
#   │   ├── libero_object_1/                        <- Task ID 1
#   │   │   └── episode_00000_attention.pt
#   │   └── ...
#   └── videos/
#       ├── libero_object_0/
#       │   ├── eval_episode_00000.mp4
#       │   ├── eval_episode_00001.mp4
#       │   └── eval_episode_00002.mp4
#       ├── libero_object_1/
#       │   └── eval_episode_00000.mp4
#       └── ...
#
# HOW IT WORKS:
#   1. For each task_id, discovers the task name from LIBERO's benchmark API
#   2. Finds all variant BDDL files (task_name_0.bddl, task_name_1.bddl, etc.)
#   3. For each variant:
#      - Temporarily swaps variant files to base location
#      - Runs lerobot-eval with n_episodes=1
#      - Renames episode_00000 → episode_{N} 
#      - Moves results to final location under OUTPUTS_DIR
#      - Restores original files
#   4. Merges all eval_info.json results into single file

# ============================================================================
# CONFIGURATION
# ============================================================================
ALL_GPU=0,1
POLICY_GPU_ID=0  # Which physical GPU to use (0 or 1)
MAX_PARALLEL_TASKS=2  # Number of tasks to evaluate in parallel (recommended: 1 per GPU)

TASK_SUITE=libero_object  # libero_spatial, libero_object, libero_goal, libero_10

# Task IDs to evaluate (just specify the IDs you want)
TASK_IDS=(0 1 2 3 4 5 6 7 8 9)  # Add more like: TASK_IDS=(0 1 2 3)

# Task names will be auto-discovered from BDDL directory
# But if auto-discovery fails, you can manually specify the mapping:
declare -A TASK_NAME_OVERRIDE=(
    # [0]="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
)

EPISODE=7  # Maximum number of scene variants to evaluate per task (will be capped by available variants)

OUTPUTS_DIR=./eval_logs/scene_variants_eval  # Main output directory (all tasks go here)
POLICY_PATH=lerobot/pi05_libero_finetuned
N_ACTION_STEPS=10
COMPILE_MODEL=false
USE_INIT_STATES=true  # Use saved init states from variant files
SAVE_ATTENTION_MAPS=true

# Paths to BDDL and init files
BDDL_DIR=/workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/$TASK_SUITE
INIT_DIR=/workspace/lerobot/third_party/LIBERO/libero/libero/init_files/$TASK_SUITE

# ============================================================================
# RUN EVALUATION
# ============================================================================
echo "============================================================================"
echo "EVALUATING SCENE VARIANTS"
echo "============================================================================"
echo "Task Suite: $TASK_SUITE"
echo "Task IDs: ${TASK_IDS[@]}"
echo "Max Episodes per task: $EPISODE"
echo "Policy: $POLICY_PATH"
echo "Output Dir: $OUTPUTS_DIR"
echo "============================================================================"
echo ""

# Function to evaluate one task (used by parallel execution)
eval_task() {
    local TASK_ID=$1
    local GPU_ID=$2
    
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Evaluating Task ID $TASK_ID on GPU $GPU_ID"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    # Task name will be auto-discovered by the Python script
    python3 /workspace/lerobot/pi_setting/eval/eval_libero_scene_variants.py \
        --task_suite=$TASK_SUITE \
        --task_id=$TASK_ID \
        --max_episodes=$EPISODE \
        --policy_path=$POLICY_PATH \
        --policy_gpu_id=$GPU_ID \
        --cuda_devices=$ALL_GPU \
        --n_action_steps=$N_ACTION_STEPS \
        --compile_model=$COMPILE_MODEL \
        --use_init_states=$USE_INIT_STATES \
        --save_attention_maps=$SAVE_ATTENTION_MAPS \
        --output_dir=$OUTPUTS_DIR \
        --bddl_dir=$BDDL_DIR \
        --init_dir=$INIT_DIR
    
    if [ $? -eq 0 ]; then
        echo "✓ Task $TASK_ID completed successfully"
        return 0
    else
        echo "✗ Task $TASK_ID failed"
        return 1
    fi
}

export -f eval_task
export TASK_SUITE EPISODE POLICY_PATH ALL_GPU N_ACTION_STEPS COMPILE_MODEL USE_INIT_STATES SAVE_ATTENTION_MAPS OUTPUTS_DIR BDDL_DIR INIT_DIR

# Parallel execution with GPU assignment
if [ $MAX_PARALLEL_TASKS -gt 1 ]; then
    echo "Running with parallel execution: $MAX_PARALLEL_TASKS tasks at a time"
    echo ""
    
    # Extract available GPU IDs from ALL_GPU string
    IFS=',' read -ra GPU_ARRAY <<< "$ALL_GPU"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    # Process tasks in batches
    task_idx=0
    total_tasks=${#TASK_IDS[@]}
    
    while [ $task_idx -lt $total_tasks ]; do
        # Start a batch of parallel tasks
        batch_pids=()
        batch_tasks=()
        
        for ((i=0; i<$MAX_PARALLEL_TASKS && task_idx<$total_tasks; i++)); do
            TASK_ID=${TASK_IDS[$task_idx]}
            # Round-robin GPU assignment
            GPU_ID=${GPU_ARRAY[$((i % NUM_GPUS))]}
            
            echo "[Batch] Starting Task $TASK_ID on GPU $GPU_ID (background job)"
            eval_task $TASK_ID $GPU_ID &
            batch_pids+=($!)
            batch_tasks+=($TASK_ID)
            ((task_idx++))
        done
        
        # Wait for all tasks in this batch to complete
        echo ""
        echo "Waiting for batch of ${#batch_pids[@]} tasks to complete..."
        failed_tasks=()
        for idx in "${!batch_pids[@]}"; do
            pid=${batch_pids[$idx]}
            task_id=${batch_tasks[$idx]}
            if wait $pid; then
                echo "  ✓ Task $task_id (PID $pid) completed"
            else
                echo "  ✗ Task $task_id (PID $pid) failed"
                failed_tasks+=($task_id)
            fi
        done
        
        if [ ${#failed_tasks[@]} -gt 0 ]; then
            echo "⚠ Warning: ${#failed_tasks[@]} tasks failed in this batch: ${failed_tasks[*]}"
        fi
        echo ""
    done
else
    echo "Running with sequential execution (MAX_PARALLEL_TASKS=1)"
    echo ""
    
    # Sequential execution (original behavior)
    for TASK_ID in "${TASK_IDS[@]}"; do
        eval_task $TASK_ID $POLICY_GPU_ID
    done
fi

echo ""
echo "============================================================================"
echo "All tasks completed! Check results in: $OUTPUTS_DIR"
echo "============================================================================"
