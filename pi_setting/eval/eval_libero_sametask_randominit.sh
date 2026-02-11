
ALL_GPU=0,1
POLICY_GPU_ID=0  # Which physical GPU to use (0 or 1)

OUTPUTS_DIR=./eval_logs/object_same_scene_random_init
TASK_SUITE=libero_object # libero_spatial,libero_object,libero_goal,libero_10

EPISODE=10  # run amount for each task (for success rate)
TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'  # different scenes for one group

# Set batch_size to min(EPISODE, 3) to avoid validation errors
BATCH_SIZE=$(( EPISODE < 3 ? EPISODE : 3 ))

# Override Docker's CUDA_VISIBLE_DEVICES to make both GPUs visible
CUDA_VISIBLE_DEVICES=$ALL_GPU lerobot-eval  \
 --env.type=libero      \
 --env.task=$TASK_SUITE  \
 --eval.batch_size=$BATCH_SIZE    \
 --eval.n_episodes=$EPISODE   \
 --policy.path=lerobot/pi05_libero_finetuned    \
 --policy.n_action_steps=10 \
 --policy.device=cuda:$POLICY_GPU_ID \
 --policy.compile_model=false \
 --output_dir=$OUTPUTS_DIR  \
 --env.max_parallel_tasks=1 \
 --env.task_ids=$TASK_IDS \
 --env.init_states=false \
 --eval.save_attention_maps=true 