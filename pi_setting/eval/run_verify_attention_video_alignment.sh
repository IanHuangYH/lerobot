#!/bin/bash
# =============================================================================
# Attention Map & Video Alignment Verification Script
# =============================================================================
#
# PURPOSE:
#   Verify that saved attention heatmaps correctly align with video frames from
#   policy evaluation. Creates side-by-side visualizations for debugging and analysis.
#
# WHAT IT DOES:
#   1. Reads attention maps (.pt files) from evaluation runs
#   2. Reads corresponding video frames (.mp4 files)
#   3. For each specified timestep:
#      - Extracts attention heatmap for specific layer/denoising step/action
#      - Overlays heatmap on video frame with transparency
#      - Saves 4 images per timestep:
#        * agentview overlay (attention + video)
#        * agentview pure heatmap (attention only)
#        * wrist overlay (attention + video)
#        * wrist pure heatmap (attention only)
#
# USE CASES:
#   - Verify attention maps are correctly synchronized with video frames
#   - Analyze which image regions the policy focuses on during execution
#   - Debug attention mechanism behavior across different camera views
#   - Compare attention patterns across different tasks/variants
#
# OUTPUT STRUCTURE:
#   eval_logs/{EVAL_FOLDER}/attention/{TASK_FOLDER}/verification/episode_{N}/
#   ├── timestep_000_agentview_overlay.png
#   ├── timestep_000_agentview_heatmap.png
#   ├── timestep_000_wrist_overlay.png
#   ├── timestep_000_wrist_heatmap.png
#   ├── timestep_010_agentview_overlay.png
#   └── ...
#
# CONFIGURATION:
#   - EVAL_FOLDER: Which evaluation run to verify (e.g., "scene_variants_eval")
#   - EVAL_SCENE_INDEX: Max task ID to process (0-9, checks all variants per task)
#   - EVAL_TASK_AMOUNT: Max episode/variant to process per task
#   - TIMESTEPS: Which rollout steps to visualize (e.g., "0 10 20 30")
#   - LAYER: Which transformer layer's attention to visualize (0-17, default: 17 = last)
#   - DENOISING_STEP: Which diffusion step (0-9, default: 0 = final denoised)
#   - ACTION_IDX: Which action dimension to visualize (0-48)
#
# EXAMPLE:
#   Verifying task 0, episode 3 at timesteps 0, 50, 100:
#   - Loads: eval_logs/scene_variants_eval/attention/libero_object_0/episode_00003_attention.pt
#   - Loads: eval_logs/scene_variants_eval/videos/libero_object_0/eval_episode_00003.mp4
#   - Saves: verification/episode_00003/timestep_000_*.png, timestep_050_*.png, timestep_100_*.png
#
# =============================================================================

# Verification script to check alignment between attention heatmaps and MP4 video frames
# This creates side-by-side visualizations for multiple timesteps

# Default paths (modify as needed)
EVAL_FOLDER="scene_variants_eval"
EVAL_SCENE_INDEX=9  # Which evaluation run to use (0-9, default: 0 = first run)
EVAL_TASK_AMOUNT=10  # How many tasks to verify (default: 10, set to 1 for quick test)

for i in $(seq 0 $EVAL_SCENE_INDEX); do
    for j in $(seq 0 $EVAL_TASK_AMOUNT); do
        TASK_FOLDER="libero_object_$i"
        ATTENTION_FILE="eval_logs/$EVAL_FOLDER/attention/$TASK_FOLDER/episode_$(printf "%05d" $j)_attention.pt"
        VIDEO_FILE="eval_logs/$EVAL_FOLDER/videos/$TASK_FOLDER/eval_episode_$(printf "%05d" $j).mp4"
        OUTPUT_DIR="eval_logs/$EVAL_FOLDER/attention/$TASK_FOLDER/verification"

        # Default parameters
        TIMESTEPS="0 10 20 30 40 50 60 70 80 90 100 110 120 130 140"  # Which rollout steps to visualize
        ACTION_IDX=0            # Which action index to visualize (0-48)
        DENOISING_STEP=0        # Which denoising step (0-9, default: 0 = final, t=1)
        LAYER=17                # Which transformer layer (0-17, default: 17 = last)
        COLORMAP="hot"          # Matplotlib colormap (hot, viridis, jet, etc.)
        ALPHA=0.55               # Transparency for overlay (0.0=invisible, 1.0=opaque)
        # Run the verification script
        python pi_setting/eval/verify_attention_video_alignment.py \
            --attention_path "$ATTENTION_FILE" \
            --video_path "$VIDEO_FILE" \
            --output_path "$OUTPUT_DIR" \
            --timesteps $TIMESTEPS \
            --action_idx $ACTION_IDX \
            --denoising_step $DENOISING_STEP \
            --layer $LAYER \
            --colormap "$COLORMAP" \
            --alpha $ALPHA \
            --episode_id $j

        echo ""
        echo "✓ Verification complete! Check outputs at: $OUTPUT_DIR/episode_$(printf "%05d" $j)"
        echo "  - timestep_XXX_agentview.png (overlay | pure heatmap)"
        echo "  - timestep_XXX_wrist.png (overlay | pure heatmap)"
    done
done
