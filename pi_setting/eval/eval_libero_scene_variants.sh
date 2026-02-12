#!/bin/bash
# Evaluate a policy on LIBERO scene variants.
# This script evaluates each scene variant (e.g., basket in different positions)
# and saves results in separate folders per variant.

# ============================================================================
# CONFIGURATION
# ============================================================================
ALL_GPU=0,1
POLICY_GPU_ID=0  # Which physical GPU to use (0 or 1)

TASK_SUITE=libero_object  # libero_spatial, libero_object, libero_goal, libero_10
TASK_NAME=pick_up_the_alphabet_soup_and_place_it_in_the_basket
TASK_ID=0  # Task ID in the suite (0-9 for most suites)

EPISODE=10  # Maximum number of scene variants to evaluate (will be capped by available variants)

OUTPUTS_DIR=./eval_logs
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
echo "Task Name: $TASK_NAME"
echo "Task ID: $TASK_ID"
echo "Max Episodes: $EPISODE"
echo "Policy: $POLICY_PATH"
echo "Output Dir: $OUTPUTS_DIR"
echo "============================================================================"
echo ""

python3 /workspace/lerobot/pi_setting/eval/eval_libero_scene_variants.py \
    --task_suite=$TASK_SUITE \
    --task_name=$TASK_NAME \
    --task_id=$TASK_ID \
    --max_episodes=$EPISODE \
    --policy_path=$POLICY_PATH \
    --policy_gpu_id=$POLICY_GPU_ID \
    --cuda_devices=$ALL_GPU \
    --n_action_steps=$N_ACTION_STEPS \
    --compile_model=$COMPILE_MODEL \
    --use_init_states=$USE_INIT_STATES \
    --save_attention_maps=$SAVE_ATTENTION_MAPS \
    --output_dir=$OUTPUTS_DIR \
    --bddl_dir=$BDDL_DIR \
    --init_dir=$INIT_DIR

echo ""
echo "Done! Check results in: $OUTPUTS_DIR"
