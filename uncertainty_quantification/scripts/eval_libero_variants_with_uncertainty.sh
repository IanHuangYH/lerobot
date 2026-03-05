#!/bin/bash

# Evaluation script for LIBERO scene variants with uncertainty prediction
# Uses eval_libero_scene_variants.py to handle variant file naming

ALL_GPU=0,1
POLICY_GPU_ID=0

OUTPUT_DIR="uncertainty_quantification/eval_log/vlm_general_attention_rnd_long_scene_variants"
TASK_SUITE=libero_object_variants

# Number of episodes (variants) per task
EPISODES_PER_TASK=7

# Task IDs to evaluate
TASK_IDS=(0 1 2 3 4 5 6 7 8 9)

# Paths to variant files
LIBERO_BASE="/workspace/lerobot/third_party/LIBERO/libero/libero"
BDDL_DIR="$LIBERO_BASE/bddl_files/$TASK_SUITE"
INIT_DIR="$LIBERO_BASE/init_files/$TASK_SUITE"

# Policy settings
POLICY_PATH="lerobot/pi05_libero_finetuned"
N_ACTION_STEPS=10
COMPILE_MODEL="false"
USE_INIT_STATES="true"
SAVE_ATTENTION_MAPS="true"
SAVE_VLM_ATTENTION_MAPS="true"
SAVE_GENERAL_VLM_ATTENTION_MAPS="true"
SAVE_UNCERTAINTY_MAPS="true"
RND_MODELS_DIR="uncertainty_quantification/rnd_save_models_long"  # Directory containing trained RND models

echo "================================================================================"
echo "LIBERO Scene Variants Evaluation with Uncertainty Prediction"
echo "================================================================================"
echo "Task Suite: $TASK_SUITE"
echo "Episodes per task: $EPISODES_PER_TASK"
echo "Tasks: ${TASK_IDS[@]}"
echo "Output Dir: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Evaluate each task
for TASK_ID in "${TASK_IDS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "Evaluating Task ID: $TASK_ID"
    echo "================================================================================"
    
    CUDA_VISIBLE_DEVICES=$ALL_GPU python pi_setting/eval/eval_libero_scene_variants.py \
        --task_suite "$TASK_SUITE" \
        --task_id "$TASK_ID" \
        --max_episodes "$EPISODES_PER_TASK" \
        --policy_path "$POLICY_PATH" \
        --policy_gpu_id "$POLICY_GPU_ID" \
        --cuda_devices "$ALL_GPU" \
        --n_action_steps "$N_ACTION_STEPS" \
        --compile_model "$COMPILE_MODEL" \
        --use_init_states "$USE_INIT_STATES" \
        --save_attention_maps "$SAVE_ATTENTION_MAPS" \
        --save_vlm_attention_maps "$SAVE_VLM_ATTENTION_MAPS" \
        --save_general_vlm_attention_maps "$SAVE_GENERAL_VLM_ATTENTION_MAPS" \
        --save_uncertainty_maps "$SAVE_UNCERTAINTY_MAPS" \
        --rnd_models_dir "$RND_MODELS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --bddl_dir "$BDDL_DIR" \
        --init_dir "$INIT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for task $TASK_ID"
        # Continue with other tasks
    fi
done

echo ""
echo "================================================================================"
echo "Evaluation completed!"
echo "================================================================================"
echo "Results:"
echo "  Output: $OUTPUT_DIR/"
echo "  Videos: $OUTPUT_DIR/videos/"
echo "  Attention: $OUTPUT_DIR/attention/"
echo "  Uncertainty: $OUTPUT_DIR/uncertainty/"
echo "  Summary: $OUTPUT_DIR/eval_info.json"
echo "================================================================================"
