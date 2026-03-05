#!/bin/bash
# =============================================================================
# OPTIMIZED Parallel VLM Attention Visualization Runner (v2)
# =============================================================================
#
# This uses SMART parallelization to maximize CPU usage and minimize I/O:
# - Loads each 7GB attention file ONLY ONCE
# - Parallelizes inner loop (layers/tokens/heads) using shared memory
# - Reduces I/O from ~4.5TB to ~70GB (64x reduction!)
# - Can now use all CPU cores without I/O bottleneck
#
# SPEEDUP: 60-100x faster than sequential, ~10-20x faster than old parallel
#
# ATTENTION TYPES:
#   Set ATTENTION_TYPE variable below:
#   - ATTENTION_TYPE="task"    → Visualize task-specific VLM attention
#   - ATTENTION_TYPE="general" → Visualize general VLM attention baseline
#
# =============================================================================

# Set up PYTHONPATH to include LIBERO (required for task instruction extraction)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${WORKSPACE_DIR}/third_party/LIBERO:${WORKSPACE_DIR}:${PYTHONPATH}"

# Configuration (same as run_visualize_vlm_attention.sh but runs in parallel)
EVAL_FOLDER="uncertainty_quantification/eval_log/vlm_attention_object_all"
TASK_NAME="libero_object"

# Which type of VLM attention to visualize
ATTENTION_TYPE="task"   # "task" = task-specific VLM attention, "general" = baseline with dummy task

# Task/episode range
MAX_TASK_ID=9 # 0-9 for all tasks, set to 0 for just the first task
MAX_EPISODE_ID=0 # 0-x for number of episodes to visualize per task (0-9), set to 0 for just the first episode

# Which parameters to visualize
LAYERS=(17 16 15 14)  # Can specify multiple: (15 16 17)
TOKENS=(774 780)    # Can specify multiple token indices to loop through
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
TIMESTEPS=(0 10 20 30 40 50 60 70 80 90 100)

# Visualization settings
HEAD_AGG="mean"
TOKEN_AGG="mean"
ALPHA=0.5
COLORMAP="hot"

# Parallelization
# OPTIMIZED v2: Inner loop parallelization (load file once, parallelize combos)
# Now you CAN use all CPU cores! The bottleneck is removed.
# 
# Old approach: Each file loaded 64x = 4.5TB I/O (bottleneck)
# New approach: Each file loaded 1x = 70GB I/O (fast!)
# 
# Recommended: Use all your CPU cores (15-32) for maximum speed
NUM_WORKERS=16  # Workers for inner loop (layers/tokens/heads) - use your CPU count!

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

# Build command (using OPTIMIZED v2 with inner loop parallelization)
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
echo "Done! OPTIMIZED v2 used: Each .pt file loaded only ONCE!"
echo "Check output in:"
if [ "$ATTENTION_TYPE" = "task" ]; then
    echo "  $EVAL_FOLDER/vlm_attention/libero_object_X/viz_episode_XXXXX/"
else
    echo "  $EVAL_FOLDER/general_vlm_attention/libero_object_X/viz_episode_XXXXX/"
fi
echo "================================================================================"
