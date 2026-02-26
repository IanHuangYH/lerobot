#!/bin/bash
# =============================================================================
# Uncertainty Heatmap & Video Alignment Verification Script
# =============================================================================
#
# PURPOSE:
#   Verify that saved uncertainty heatmaps correctly align with video frames from
#   policy evaluation. Creates 2x2 grid visualizations for debugging and analysis.
#
# WHAT IT DOES:
#   1. Reads uncertainty data (.pt files) from evaluation runs
#   2. Reads corresponding video frames (.mp4 files)
#   3. For each specified timestep:
#      - Extracts uncertainty spatial maps for both cameras (16x16)
#      - Overlays heatmaps on video frames with transparency
#      - Creates 2x2 grid visualization:
#        * Top-left: Agentview + overlay with score
#        * Top-right: Wrist + overlay with score
#        * Bottom-left: Agentview pure heatmap
#        * Bottom-right: Wrist pure heatmap
#      - Displays overall uncertainty and timestep at bottom
#
# USE CASES:
#   - Verify uncertainty maps are correctly synchronized with video frames
#   - Analyze which image regions have high uncertainty during execution
#   - Debug RND model behavior across different camera views
#   - Compare uncertainty patterns across different tasks/episodes
#
# OUTPUT STRUCTURE:
#   eval_logs/{EVAL_FOLDER}/uncertainty/{TASK_FOLDER}/verification/episode_{N}/
#   ├── timestep_000_grid.png
#   ├── timestep_010_grid.png
#   ├── timestep_020_grid.png
#   └── ...
#
# CONFIGURATION:
#   - EVAL_FOLDER: Which evaluation run to verify (e.g., "with_uncertainty")
#   - EVAL_SCENE_INDEX: Max task ID to process (0-9)
#   - EVAL_EPISODE_AMOUNT: Max episode to process per task
#   - TIMESTEPS: Which rollout steps to visualize (e.g., "0 10 20 30")
#   - COLORMAP: Heatmap color scheme (default: "viridis")
#   - ALPHA: Overlay transparency (default: 0.55)
#
# EXAMPLE:
#   Verifying task 0, episode 3 at timesteps 0, 50, 100:
#   - Loads: eval_logs/with_uncertainty/uncertainty/libero_object_0/episode_00003_uncertainty.pt
#   - Loads: eval_logs/with_uncertainty/videos/libero_object_0/eval_episode_00003.mp4
#   - Saves: verification/episode_00003/timestep_000_grid.png, etc.
#
# =============================================================================

BASE_EVAL_DIR="uncertainty_quantification/eval_log"
EVAL_FOLDER="rnd_variant_object"
EVAL_SCENE_INDEX=9  # Which task IDs to process (0-9, default: 0 = first task)
EVAL_EPISODE_AMOUNT=7  # How many episodes to verify per task (default: 9, set to 1 for quick test)
TASK="unseen_object"  # Task type (e.g., "object", "spatial", "goal")


echo "============================================================"
echo "Uncertainty Visualization - Batch Verification"
echo "============================================================"
echo "Evaluation folder: $EVAL_FOLDER"
echo "Task range: 0 to $EVAL_SCENE_INDEX"
echo "Episodes per task: 0 to $EVAL_EPISODE_AMOUNT"
echo "============================================================"
echo ""

for i in $(seq 0 $EVAL_SCENE_INDEX); do
    TASK_FOLDER="libero_${TASK}_$i"
    
    echo "------------------------------------------------------------"
    echo "Processing task: $TASK_FOLDER"
    echo "------------------------------------------------------------"
    
    for j in $(seq 0 $EVAL_EPISODE_AMOUNT); do
        UNCERTAINTY_FILE="$BASE_EVAL_DIR/$EVAL_FOLDER/uncertainty/$TASK_FOLDER/episode_$(printf "%05d" $j)_uncertainty.pt"
        VIDEO_FILE="$BASE_EVAL_DIR/$EVAL_FOLDER/videos/$TASK_FOLDER/eval_episode_$(printf "%05d" $j).mp4"
        OUTPUT_DIR="$BASE_EVAL_DIR/$EVAL_FOLDER/uncertainty/$TASK_FOLDER/verification"

        # Check if files exist
        if [ ! -f "$UNCERTAINTY_FILE" ]; then
            echo "  ⚠ Skipping episode $j: uncertainty file: $UNCERTAINTY_FILE not found"
            continue
        fi
        
        if [ ! -f "$VIDEO_FILE" ]; then
            echo "  ⚠ Skipping episode $j: video file: $VIDEO_FILE not found"
            continue
        fi

        # Default visualization parameters
        TIMESTEPS="0 10 20 30 40 50 60 70 80 90 100 110 120 130 140"  # Which rollout steps to visualize
        COLORMAP="viridis"      # Matplotlib colormap (viridis, plasma, hot, coolwarm, etc.)
        ALPHA=0.55              # Transparency for overlay (0.0=invisible, 1.0=opaque)
        
        # Run the verification script
        python -m uncertainty_quantification.visualization.verify_uncertainty_video_alignment \
            --uncertainty_path "$UNCERTAINTY_FILE" \
            --video_path "$VIDEO_FILE" \
            --output_path "$OUTPUT_DIR" \
            --timesteps $TIMESTEPS \
            --colormap "$COLORMAP" \
            --alpha $ALPHA \
            --episode_id $j

        echo ""
    done
    
    echo "✓ Task $TASK_FOLDER complete!"
    echo ""
done

echo "============================================================"
echo "✓ Batch verification complete!"
echo "============================================================"
echo "Outputs saved to: $BASE_EVAL_DIR/$EVAL_FOLDER/uncertainty/*/verification/"
echo "  - timestep_XXX_grid.png (2x2 visualization)"
echo "============================================================"
