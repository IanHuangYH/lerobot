#!/bin/bash
# =============================================================================
# VLM Attention Visualization Batch Runner
# =============================================================================
#
# PURPOSE:
#   Batch process VLM attention maps to create visualizations showing how the
#   model's task instruction tokens attend to different image regions.
#
#   Supports TWO types of VLM attention:
#   1. Task-specific (default): Uses actual task instruction from LIBERO
#   2. General baseline: Uses dummy task "task: perform the task"
#
# SWITCHING BETWEEN TYPES:
#   Set ATTENTION_TYPE variable below:
#   - ATTENTION_TYPE="task"    → Visualize task-specific VLM attention
#   - ATTENTION_TYPE="general" → Visualize general VLM attention baseline
#
# WHAT IT DOES:
#   1. Reads VLM attention maps (.pt files) from evaluation runs
#   2. Reads corresponding video frames (.mp4 files)
#   3. For each specified timestep:
#      - Extracts task→image attention (aggregated across task tokens)
#      - Creates side-by-side visualization: overlay (left) | pure heatmap (right)
#      - Adds white title bar with camera name and timestep
#      - Saves 2 images per timestep:
#        * agentview.png (side-by-side with title bar)
#        * wrist.png (side-by-side with title bar)
#
# KEY DIFFERENCES FROM ACTION ATTENTION:
#   - VLM attention captured once per action (during prefix encoding)
#   - Shows task language → image relationship (not action → image)
#   - Helps understand: "What does the model look at when reading the task?"
#
# USE CASES:
#   - Understand how task instructions guide visual attention
#   - Analyze which objects/regions are relevant to task commands
#   - Debug VLM grounding behavior (does it look at correct objects?)
#   - Compare attention patterns across different tasks/scenes
#
# OUTPUT STRUCTURE:
#   eval_logs/{EVAL_FOLDER}/vlm_attention/{TASK_FOLDER}/viz_episode_{N}/
#   ├── timestep_000_agentview.png  (overlay | heatmap side-by-side)
#   ├── timestep_000_wrist.png      (overlay | heatmap side-by-side)
#   ├── timestep_010_agentview.png
#   └── ...
#
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Which evaluation folder to process
EVAL_FOLDER="uncertainty_quantification/eval_log/vlm_general_attention_object_all" #eval_logs/quick_test_vlm_attention_1
TASK_NAME="libero_object"

# Which type of VLM attention to visualize
ATTENTION_TYPE="general"   # "task" = task-specific VLM attention, "general" = baseline with dummy task
# Task range to process
EVAL_SCENE_INDEX=1      # Max task ID (0-9 for per taks, e.g., object_0 to object_9)
EVAL_TASK_AMOUNT=1      # Max episode per task

# Which rollout steps to visualize
TIMESTEPS=(0 10 20 30 40 50 60 70 80 90 100 110 120 130 140)  # Visualize at these rollout steps (0 = prefix encoding, 10/20/30 = during execution)

# Visualization parameters
LAYERS=(17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0)         # Which transformer layers to visualize (0-17, can specify multiple)
HEAD_AGG="min"          # How to aggregate attention heads: mean/max/sum
TOKEN_AGG="mean"         # How to aggregate task tokens: mean/max/sum
ALPHA=0.5                # Overlay transparency (0=transparent, 1=opaque)
COLORMAP="hot"           # Matplotlib colormap: hot/viridis/jet/etc

# Optional: Visualize specific tokens (leave empty for aggregated view)
# Example: SPECIFIC_TOKEN_IDXS=(773 774 780)  # for 'alphabet', 'soup', 'basket' tokens
# Leave empty or set to () to aggregate all task tokens with TOKEN_AGG
SPECIFIC_TOKEN_IDXS=(770)    # Can specify multiple token indices to loop through
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


# Optional: Visualize specific attention heads (leave empty for aggregated view)
# Example: SPECIFIC_HEAD_IDXS=(0 3 6)  # for heads 0, 3, 6 (out of 0-7)
# Leave empty or set to () to aggregate all heads with HEAD_AGG
SPECIFIC_HEAD_IDXS=(0 1 2 3 4 5 6 7)    # Can specify multiple head indices to loop through

# NOTE: Task texts are now auto-detected from LIBERO instead of hardcoded!
# This ensures we use the exact task instruction from the dataset.

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "VLM Attention Visualization Batch Processing"
echo "================================================================================"
echo "Attention type: $ATTENTION_TYPE"
echo "Eval folder: $EVAL_FOLDER"
echo "Processing tasks 0-${EVAL_SCENE_INDEX}, episodes 0-$((EVAL_TASK_AMOUNT-1))"
echo "Timesteps: ${TIMESTEPS[@]}"
echo "Layers: ${LAYERS[@]}, Aggregation: heads=$HEAD_AGG, tokens=$TOKEN_AGG"
echo "Specific tokens: ${SPECIFIC_TOKEN_IDXS[@]:-None} (overrides token aggregation)"
echo "Specific heads: ${SPECIFIC_HEAD_IDXS[@]:-None} (overrides head aggregation)"
echo "================================================================================"

TOTAL_PROCESSED=0
TOTAL_ERRORS=0

# Loop through layers
for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "################################################################################"
    echo "Processing Layer: $LAYER"
    echo "################################################################################"

    # Determine token indices to process (check once)
    if [ ${#SPECIFIC_TOKEN_IDXS[@]} -eq 0 ]; then
        TOKEN_INDICES=("")
        echo "  Token mode: Aggregating all tokens with $TOKEN_AGG"
    else
        TOKEN_INDICES=("${SPECIFIC_TOKEN_IDXS[@]}")
        echo "  Token mode: Processing specific tokens: ${SPECIFIC_TOKEN_IDXS[@]}"
    fi

    # Determine head indices to process (check once)
    if [ ${#SPECIFIC_HEAD_IDXS[@]} -eq 0 ]; then
        HEAD_INDICES=("")
        echo "  Head mode: Aggregating all heads with $HEAD_AGG"
    else
        HEAD_INDICES=("${SPECIFIC_HEAD_IDXS[@]}")
        echo "  Head mode: Processing specific heads: ${SPECIFIC_HEAD_IDXS[@]}"
    fi

    # Loop through token indices
    for TOKEN_IDX in "${TOKEN_INDICES[@]}"; do
        if [ -n "$TOKEN_IDX" ]; then
            echo ""
            echo "  ============================================================================"
            echo "  Processing Token Index: $TOKEN_IDX"
            echo "  ============================================================================"
        fi

        # Loop through head indices
        for HEAD_IDX in "${HEAD_INDICES[@]}"; do
            if [ -n "$HEAD_IDX" ]; then
                echo ""
                echo "    ------------------------------------------------------------------------"
                echo "    Processing Head Index: $HEAD_IDX"
                echo "    ------------------------------------------------------------------------"
            fi

            # Loop through tasks (NOW PROPERLY NESTED!)
            for TASK_ID in $(seq 0 $EVAL_SCENE_INDEX); do
                TASK_NAME_ID="${TASK_NAME}_${TASK_ID}"
                
                echo ""
                echo "      ----------------------------------------------------------------------"
                echo "      Task: $TASK_NAME_ID"
                echo "      ----------------------------------------------------------------------"
                
                # Loop through episodes
                for EPISODE_ID in $(seq 0 $((EVAL_TASK_AMOUNT-1))); do
                    EPISODE_NUM=$(printf "%05d" $EPISODE_ID)
                    
                    # Define file paths based on attention type
                    if [ "$ATTENTION_TYPE" = "task" ]; then
                        ATTENTION_FILE="${EVAL_FOLDER}/vlm_attention/${TASK_NAME_ID}/episode_${EPISODE_NUM}_vlm_attention.pt"
                        OUTPUT_DIR="${EVAL_FOLDER}/vlm_attention/${TASK_NAME_ID}/viz_episode_${EPISODE_NUM}"
                    else
                        ATTENTION_FILE="${EVAL_FOLDER}/general_vlm_attention/${TASK_NAME_ID}/episode_${EPISODE_NUM}_general_vlm_attention.pt"
                        OUTPUT_DIR="${EVAL_FOLDER}/general_vlm_attention/${TASK_NAME_ID}/viz_episode_${EPISODE_NUM}"
                    fi
                    VIDEO_FILE="${EVAL_FOLDER}/videos/${TASK_NAME_ID}/eval_episode_${EPISODE_NUM}.mp4"
                    
                    # Check if files exist
                    if [ ! -f "$ATTENTION_FILE" ]; then
                        echo "        Episode ${EPISODE_NUM}: SKIP (VLM attention file not found)"
                        continue
                    fi
                    
                    if [ ! -f "$VIDEO_FILE" ]; then
                        echo "        Episode ${EPISODE_NUM}: SKIP (video file not found)"
                        continue
                    fi
                    
                    echo "        Episode ${EPISODE_NUM}: Processing..."
                    
                    # Build command with new argument structure
                    CMD="python pi_setting/eval/visualize_vlm_attention.py \
                        --eval_folder '$EVAL_FOLDER' \
                        --task_name $TASK_NAME \
                        --task_id $TASK_ID \
                        --episode_id $EPISODE_ID \
                        --rollout_steps ${TIMESTEPS[@]} \
                        --layer $LAYER \
                        --attention_type $ATTENTION_TYPE \
                        --head_aggregation $HEAD_AGG \
                        --alpha $ALPHA \
                        --colormap $COLORMAP"
                    
                    # Add specific_token_idx if set (ignores TOKEN_AGG when specific token is used)
                    if [ -n "$TOKEN_IDX" ]; then
                        CMD="$CMD --specific_token_idx $TOKEN_IDX"
                        echo "            → Visualizing specific token: $TOKEN_IDX (TOKEN_AGG ignored)"
                    else
                        CMD="$CMD --token_aggregation $TOKEN_AGG"
                        echo "            → Aggregating all task tokens with: $TOKEN_AGG"
                    fi
                    
                    # Add specific_head_idx if set (ignores HEAD_AGG when specific head is used)
                    if [ -n "$HEAD_IDX" ]; then
                        CMD="$CMD --specific_head_idx $HEAD_IDX"
                        echo "            → Visualizing specific head: $HEAD_IDX (HEAD_AGG ignored)"
                    else
                        echo "            → Aggregating all attention heads with: $HEAD_AGG"
                    fi
                    
                    # Run visualization (task text auto-detected from LIBERO)
                    eval $CMD
                    
                    if [ $? -eq 0 ]; then
                        echo "        Episode ${EPISODE_NUM}: ✓ SUCCESS"
                        TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
                    else
                        echo "        Episode ${EPISODE_NUM}: ✗ ERROR"
                        TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
                    fi
                done  # End episode loop
            done  # End task loop

        done  # End head loop
    done  # End token loop
done  # End layer loop

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

echo ""
echo "================================================================================"
echo "Batch Processing Complete"
echo "================================================================================"
echo "Successfully processed: $TOTAL_PROCESSED episodes"
echo "Errors encountered: $TOTAL_ERRORS episodes"
echo "================================================================================"

if [ $TOTAL_PROCESSED -gt 0 ]; then
    echo ""
    echo "Example output locations:"
    if [ "$ATTENTION_TYPE" = "task" ]; then
        echo "  ${EVAL_FOLDER}/vlm_attention/libero_object_0/viz_episode_00000/"
        echo ""
        echo "To view results:"
        echo "  ls ${EVAL_FOLDER}/vlm_attention/libero_object_0/viz_episode_00000/"
    else
        echo "  ${EVAL_FOLDER}/general_vlm_attention/libero_object_0/viz_episode_00000/"
        echo ""
        echo "To view results:"
        echo "  ls ${EVAL_FOLDER}/general_vlm_attention/libero_object_0/viz_episode_00000/"
    fi
fi
