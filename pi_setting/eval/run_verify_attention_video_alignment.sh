#!/bin/bash

# Verification script to check alignment between attention heatmaps and MP4 video frames
# This creates side-by-side visualizations for multiple timesteps

# Default paths (modify as needed)
EVAL_FOLDER="object_0_variants"
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
