#!/bin/bash

# Evaluation script with uncertainty prediction enabled
# Similar to eval_libero_quick_test.sh but includes RND uncertainty quantification

ALL_GPU=0,1
POLICY_GPU_ID=0  # Which physical GPU to use (0 or 1)

OUTPUTS_DIR=./eval_logs/with_uncertainty
TASK_SUITE=libero_object  # libero_spatial,libero_object,libero_goal,libero_10

EPISODE=10  # run amount for each task (for success rate)
TASK_IDS='[0]'  # different scenes for one group

# Set batch_size to min(EPISODE, 10) to avoid validation errors
BATCH_SIZE=$(( EPISODE < 10 ? EPISODE : 10 ))

echo "================================================"
echo "LIBERO Evaluation with Uncertainty Prediction"
echo "================================================"
echo "Task Suite: $TASK_SUITE"
echo "Episodes: $EPISODE"
echo "Output Dir: $OUTPUTS_DIR"
echo "================================================"
echo ""

# Override Docker's CUDA_VISIBLE_DEVICES to make both GPUs visible
CUDA_VISIBLE_DEVICES=$ALL_GPU lerobot-eval \
    --env.type=libero \
    --env.task=$TASK_SUITE \
    --eval.batch_size=$BATCH_SIZE \
    --eval.n_episodes=$EPISODE \
    --policy.path=lerobot/pi05_libero_finetuned \
    --policy.n_action_steps=10 \
    --policy.device=cuda:$POLICY_GPU_ID \
    --output_dir=$OUTPUTS_DIR \
    --env.max_parallel_tasks=1 \
    --env.task_ids=$TASK_IDS \
    --env.init_states=true \
    --eval.save_attention_maps=true \
    --eval.save_uncertainty_maps=true

echo ""
echo "================================================"
echo "Evaluation completed!"
echo "================================================"
echo "Results:"
echo "  Videos: $OUTPUTS_DIR/videos/"
echo "  Attention: $OUTPUTS_DIR/attention/"
echo "  Uncertainty: $OUTPUTS_DIR/uncertainty/"
echo "================================================"
