#!/bin/bash
# =============================================================================
# VLM Attention Grid Visualization Creator
# =============================================================================
#
# PURPOSE:
#   Combine multiple attention visualizations into a single grid image showing
#   how attention patterns vary across transformer layers and attention heads.
#
# WHAT IT DOES:
#   1. Finds PNG files created by run_visualize_vlm_attention.sh
#   2. Extracts the left part (overlay) from each side-by-side visualization
#   3. Arranges them in a grid: rows=layers, columns=heads
#   4. Adds labels and title bar
#
# GRID LAYOUT:
#             Head 0    Head 1    Head 2  ...  Head 7
#   Layer 0   [img]     [img]     [img]   ...  [img]
#   Layer 1   [img]     [img]     [img]   ...  [img]
#   ...
#   Layer 17  [img]     [img]     [img]   ...  [img]
#
# OUTPUT:
#   timestep_{step:03d}_{camera}_token{token}_layer{min}to{max}_head{min}to{max}.png
#
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Source folder containing the visualizations
EVAL_FOLDER="quick_test_vlm_attention_1"
TASK_NAME="libero_object_0"
EPISODE_NUM="00000"

# Build the input directory path
INPUT_DIR="eval_logs/${EVAL_FOLDER}/vlm_attention/${TASK_NAME}/viz_episode_${EPISODE_NUM}"

# Where to save the grid visualizations
OUTPUT_DIR="${INPUT_DIR}/grids"

# Which visualizations to combine
TIMESTEPS=(30 40 50 60 70 80 90 100 110 120 130 140)                    # Which rollout steps (0 10 20) 
CAMERAS=("agentview")          # Which cameras ("agentview" "wrist") 
SPECIFIC_TOKEN_IDXS=(774 780)  # Which tokens to create grids for

# Grid configuration
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)  # All layers
HEADS=(0 1 2 3 4 5 6 7)                                # All heads

# Visual settings
PADDING=5           # Pixels between subplots
LABEL_FONT_SIZE=12  # Font size for head/layer labels
TITLE_FONT_SIZE=16  # Font size for title bar

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "VLM Attention Grid Visualization Creator"
echo "================================================================================"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Timesteps: ${TIMESTEPS[@]}"
echo "Cameras: ${CAMERAS[@]}"
echo "Tokens: ${SPECIFIC_TOKEN_IDXS[@]}"
echo "Grid size: ${#LAYERS[@]} layers × ${#HEADS[@]} heads"
echo "================================================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

TOTAL_PROCESSED=0
TOTAL_ERRORS=0

# Loop through each configuration
for TIMESTEP in "${TIMESTEPS[@]}"; do
    for CAMERA in "${CAMERAS[@]}"; do
        for TOKEN_IDX in "${SPECIFIC_TOKEN_IDXS[@]}"; do
            echo ""
            echo "Processing: timestep=$TIMESTEP, camera=$CAMERA, token=$TOKEN_IDX"
            
            # Convert arrays to comma-separated strings for Python
            LAYERS_STR=$(IFS=,; echo "${LAYERS[*]}")
            HEADS_STR=$(IFS=,; echo "${HEADS[*]}")
            
            # Run Python script to create grid
            python pi_setting/eval/create_attention_grid.py \
                --input_dir "$INPUT_DIR" \
                --output_dir "$OUTPUT_DIR" \
                --timestep $TIMESTEP \
                --camera "$CAMERA" \
                --token_idx $TOKEN_IDX \
                --layers "$LAYERS_STR" \
                --heads "$HEADS_STR" \
                --padding $PADDING \
                --label_font_size $LABEL_FONT_SIZE \
                --title_font_size $TITLE_FONT_SIZE
            
            if [ $? -eq 0 ]; then
                echo "  ✓ SUCCESS"
                TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
            else
                echo "  ✗ ERROR"
                TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
            fi
        done
    done
done

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

echo ""
echo "================================================================================"
echo "Grid Creation Complete"
echo "================================================================================"
echo "Successfully created: $TOTAL_PROCESSED grids"
echo "Errors encountered: $TOTAL_ERRORS grids"
echo "================================================================================"

if [ $TOTAL_PROCESSED -gt 0 ]; then
    echo ""
    echo "Output location: $OUTPUT_DIR"
    echo ""
    echo "To view results:"
    echo "  ls $OUTPUT_DIR"
fi
