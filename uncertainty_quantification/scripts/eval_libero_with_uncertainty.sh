#!/bin/bash

# Evaluation script with uncertainty prediction enabled
# Similar to eval_libero_quick_test.sh but includes RND uncertainty quantification

ALL_GPU=0,1
POLICY_GPU_ID=0  # Which physical GPU to use (0 or 1)

OUTPUT_DIR="uncertainty_quantification/eval_log/rnd_object_all"
TASK_SUITE=libero_object  # libero_spatial,libero_object,libero_goal,libero_10

EPISODE=3  # run amount for each task (for success rate)
TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'  # different scenes for one group

# Set batch_size to min(EPISODE, 10) to avoid validation errors
BATCH_SIZE=$(( EPISODE < 10 ? EPISODE : 10 ))

echo "================================================"
echo "LIBERO Evaluation with Uncertainty Prediction"
echo "================================================"
echo "Task Suite: $TASK_SUITE"
echo "Episodes: $EPISODE"
echo "Output Dir: $OUTPUT_DIR"
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
    --output_dir=$OUTPUT_DIR \
    --env.max_parallel_tasks=1 \
    --env.task_ids=$TASK_IDS \
    --env.init_states=true \
    --policy.compile_model=false \
    --eval.save_attention_maps=true \
    --eval.save_uncertainty_maps=true

echo ""
echo "================================================"
echo "Evaluation completed!"
echo "================================================"
echo "Results:"
echo "  Videos: $OUTPUT_DIR/videos/"
echo "  Attention: $OUTPUT_DIR/attention/"
echo "  Uncertainty: $OUTPUT_DIR/uncertainty/"
echo "================================================"
