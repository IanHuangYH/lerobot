#!/bin/bash

# Verification script to check alignment between attention heatmaps and MP4 video frames
# This creates side-by-side visualizations for multiple timesteps

# Default paths (modify as needed)
ATTENTION_FILE="eval_logs/quick_test_object/attention/libero_object_0/episode_00000_attention.pt"
VIDEO_FILE="eval_logs/quick_test_object/videos/libero_object_0/eval_episode_0.mp4"
OUTPUT_DIR="eval_logs/quick_test_object/attention/libero_object_0/verification"

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
    --alpha $ALPHA

echo ""
echo "✓ Verification complete! Check outputs at: $OUTPUT_DIR"
echo "  - timestep_XXX_agentview.png (overlay | pure heatmap)"
echo "  - timestep_XXX_wrist.png (overlay | pure heatmap)"
