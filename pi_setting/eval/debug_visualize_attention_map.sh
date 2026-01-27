ATTENTION_FILE="eval_logs/quick_test_object/attention/libero_object_0/episode_00000_attention.pt"
OUTPUT_DIR="eval_logs/quick_test_object/attention/libero_object_0/visualization"
ROLLOUT_STEP=0
DENOISING_STEP=9 # 0: flow time = 1 
ACTION_VIS_DIM="0 1 2" # which action component in the chunk to visualize, e.g., "0 1 2" for action_chunk[0:2]
LAYERS=17
CAMERA_IDX=0 #0: third-person view, 1: wrist view

cd /workspace/lerobot && python pi_setting/eval/visualize_action_to_img_attention.py \
  --attention_file $ATTENTION_FILE \
  --output_dir $OUTPUT_DIR \
  --rollout_step $ROLLOUT_STEP \
  --denoising_step $DENOISING_STEP \
  --act_vis_dim $ACTION_VIS_DIM \
  --layers $LAYERS \
  --camera_index $CAMERA_IDX \
  --img_size 224 \
  --head_aggregation mean \
  --colormap hot