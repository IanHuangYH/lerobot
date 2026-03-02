#!/bin/bash
# =============================================================================
# Demo Script: Analyze Specific Token Attention
# =============================================================================
#
# PURPOSE:
#   Analyze which task tokens have the highest attention to image regions.
#   This helps identify important words before visualizing their attention.
#
# WHAT IT DOES:
#   1. Loads VLM attention from a specific episode and rollout step
#   2. Tokenizes the task instruction to identify token boundaries
#   3. Shows all task tokens with their indices
#   4. Ranks tokens by their average attention to images
#   5. Highlights tokens matching specified keywords
#
# USE CASES:
#   - Find which tokens to visualize (Step 1 before visualization)
#   - Understand which words drive visual attention
#   - Debug tokenization and token boundary detection
#   - Compare attention patterns across different tasks
#
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Which evaluation folder to analyze
EVAL_FOLDER="quick_test_vlm_attention_1"

# Task and episode to analyze
TASK_NAME="libero_object"
TASK_ID=0
EPISODE_ID=0

# Which rollout step to analyze
ROLLOUT_STEP=10

# Keywords to highlight (space-separated)
TOKENS_OF_INTEREST="alphabet soup basket"

# Transformer layer to analyze
LAYER=17  # Last layer (default)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

EPISODE_NUM=$(printf "%05d" $EPISODE_ID)
ATTENTION_FILE="eval_logs/${EVAL_FOLDER}/vlm_attention/${TASK_NAME}_${TASK_ID}/episode_${EPISODE_NUM}_vlm_attention.pt"

# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "Demo: Specific Token Attention Analysis"
echo "================================================================================"
echo "Evaluation folder: $EVAL_FOLDER"
echo "Task: $TASK_NAME (ID: $TASK_ID)"
echo "Episode: $EPISODE_NUM"
echo "Rollout step: $ROLLOUT_STEP"
echo "Layer: $LAYER"
echo "Tokens of interest: $TOKENS_OF_INTEREST"
echo "================================================================================"
echo ""

# Check if attention file exists
if [ ! -f "$ATTENTION_FILE" ]; then
    echo "❌ ERROR: Attention file not found: $ATTENTION_FILE"
    echo ""
    echo "Make sure you have:"
    echo "  1. Run evaluation with --eval.save_vlm_attention_maps=true"
    echo "  2. Correct EVAL_FOLDER, TASK_NAME, TASK_ID, and EPISODE_ID"
    echo ""
    exit 1
fi

echo "✓ Found attention file: $ATTENTION_FILE"
echo ""

# -----------------------------------------------------------------------------
# RUN ANALYSIS
# -----------------------------------------------------------------------------

python pi_setting/eval/demo_specific_token_attention.py \
    --attention_file "$ATTENTION_FILE" \
    --rollout_step $ROLLOUT_STEP \
    --task_name $TASK_NAME \
    --task_id $TASK_ID \
    --tokens_of_interest $TOKENS_OF_INTEREST \
    --layer $LAYER

# -----------------------------------------------------------------------------
# NEXT STEPS
# -----------------------------------------------------------------------------

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Analysis Complete!"
    echo "================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review the token rankings above"
    echo "  2. Choose interesting tokens to visualize (e.g., token 774 for 'soup')"
    echo "  3. Visualize specific tokens:"
    echo ""
    echo "     # Edit run_visualize_vlm_attention.sh:"
    echo "     SPECIFIC_TOKEN_IDX=774"
    echo "     ./pi_setting/eval/run_visualize_vlm_attention.sh"
    echo ""
    echo "     # Or run directly:"
    echo "     python pi_setting/eval/visualize_vlm_attention.py \\"
    echo "         --attention_file $ATTENTION_FILE \\"
    echo "         --video_file eval_logs/${EVAL_FOLDER}/videos/${TASK_NAME}_${TASK_ID}/eval_episode_${EPISODE_NUM}.mp4 \\"
    echo "         --output_dir eval_logs/${EVAL_FOLDER}/vlm_attention/${TASK_NAME}_${TASK_ID}/viz_token774 \\"
    echo "         --task_name $TASK_NAME --task_id $TASK_ID \\"
    echo "         --rollout_steps $ROLLOUT_STEP \\"
    echo "         --specific_token_idx 774"
    echo ""
    echo "================================================================================"
fi
