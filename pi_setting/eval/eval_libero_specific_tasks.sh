


ALL_GPU=0,1
POLICY_GPU_ID=0  # Which physical GPU to use (0 or 1)

OUTPUTS_DIR=./eval_logs/goal_10
TASK_SUITE=libero_goal,libero_10
# OUTPUTS_DIR=./eval_logs/spatial_object
# TASK=libero_spatial,libero_object

EPISODE=2  # run amount for each task (for success rate)
TASK_IDS='[0,1]'  # different scenes for one group

# Override Docker's CUDA_VISIBLE_DEVICES to make both GPUs visible
CUDA_VISIBLE_DEVICES=$ALL_GPU lerobot-eval  \
 --env.type=libero      \
 --env.task=$TASK_SUITE  \
 --eval.batch_size=2    \
 --eval.n_episodes=$EPISODE   \
 --policy.path=lerobot/pi05_libero_finetuned    \
 --policy.n_action_steps=10 \
 --policy.device=cuda:$POLICY_GPU_ID \
 --output_dir=$OUTPUTS_DIR  \
 --env.max_parallel_tasks=1 \
 --env.task_ids=$TASK_IDS