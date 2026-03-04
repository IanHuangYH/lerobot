#!/bin/bash
# =============================================================================
# Parallel VLM Attention Visualization Runner
# =============================================================================
#
# This script uses Python multiprocessing to parallelize VLM attention
# visualization across multiple CPU cores. Much faster than sequential!
#
# SPEEDUP: ~8-16x faster depending on number of CPU cores
#
# ATTENTION TYPES:
#   Set ATTENTION_TYPE variable below:
#   - ATTENTION_TYPE="task"    → Visualize task-specific VLM attention
#   - ATTENTION_TYPE="general" → Visualize general VLM attention baseline
#
# =============================================================================

# Configuration (same as run_visualize_vlm_attention.sh but runs in parallel)
EVAL_FOLDER="uncertainty_quantification/eval_log/vlm_attention_scene_variants"
TASK_NAME="libero_object_variants"

# Which type of VLM attention to visualize
ATTENTION_TYPE="task"   # "task" = task-specific VLM attention, "general" = baseline with dummy task

# Task/episode range
MAX_TASK_ID=9
MAX_EPISODE_ID=7

# Which parameters to visualize
LAYERS=(17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0)  # Can specify multiple: (15 16 17)
TOKENS=(774 776 770 773  775  780)    # Can specify multiple token indices to loop through
# tokens for "pick up the alphabet soup and place it in the basket":
# Token 768: 'Task           ' (chars   0-  4)
#   Token 769: ':              ' (chars   4-  5)
#   Token 770: ' pick          ' (chars   5- 10)
#   Token 771: ' up            ' (chars  10- 13)
#   Token 772: ' the           ' (chars  13- 17)
#   Token 773: ' alphabet      ' (chars  17- 26) ★
#   Token 774: ' soup          ' (chars  26- 31) ★
#   Token 775: ' and           ' (chars  31- 35)
#   Token 776: ' place         ' (chars  35- 41)
#   Token 777: ' it            ' (chars  41- 44)
#   Token 778: ' in            ' (chars  44- 47)
#   Token 779: ' the           ' (chars  47- 51)
#   Token 780: ' basket        ' (chars  51- 58) ★
#   Token 781: ',              ' (chars  58- 59)
HEADS=(0 1 2 3 4 5 6 7)  # Empty () = aggregate all, or specific: (0 1 2 3 4 5 6 7)
TIMESTEPS=(0 10 20 30 40 50 60 70 80 90 100 110 120 130 140)

# Visualization settings
HEAD_AGG="mean"
TOKEN_AGG="mean"
ALPHA=0.5
COLORMAP="hot"

# Parallelization
NUM_WORKERS=8  # Set to empty "" to auto-detect CPU count

# =============================================================================
# Run parallel visualization
# =============================================================================

echo "================================================================================"
echo "Parallel VLM Attention Visualization"
echo "================================================================================"
echo "Attention type: $ATTENTION_TYPE"
echo "Eval folder: $EVAL_FOLDER"
echo "Tasks: 0-${MAX_TASK_ID}, Episodes: 0-${MAX_EPISODE_ID}"
echo "Layers: ${LAYERS[@]}"
echo "Tokens: ${TOKENS[@]:-All (aggregated)}"
echo "Heads: ${HEADS[@]:-All (aggregated)}"
echo "Workers: ${NUM_WORKERS:-Auto-detect}"
echo "================================================================================"

# Build command
CMD="python pi_setting/eval/visualize_vlm_attention_parallel.py \
    --eval_folder $EVAL_FOLDER \
    --task_name $TASK_NAME \
    --max_task_id $MAX_TASK_ID \
    --max_episode_id $MAX_EPISODE_ID \
    --layers ${LAYERS[@]} \
    --timesteps ${TIMESTEPS[@]} \
    --attention_type $ATTENTION_TYPE \
    --head_aggregation $HEAD_AGG \
    --token_aggregation $TOKEN_AGG \
    --alpha $ALPHA \
    --colormap $COLORMAP"

# Add token indices if specified
if [ ${#TOKENS[@]} -gt 0 ]; then
    CMD="$CMD --tokens ${TOKENS[@]}"
fi

# Add head indices if specified
if [ ${#HEADS[@]} -gt 0 ]; then
    CMD="$CMD --heads ${HEADS[@]}"
fi

# Add num_workers if specified
if [ -n "$NUM_WORKERS" ]; then
    CMD="$CMD --num_workers $NUM_WORKERS"
fi

# Run it!
eval $CMD

echo ""
echo "================================================================================"
echo "Done! Check output in:"
if [ "$ATTENTION_TYPE" = "task" ]; then
    echo "  $EVAL_FOLDER/vlm_attention/libero_object_X/viz_episode_XXXXX/"
else
    echo "  $EVAL_FOLDER/general_vlm_attention/libero_object_X/viz_episode_XXXXX/"
fi
echo "================================================================================"
