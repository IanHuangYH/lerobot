# OUTPUTS_DIR=./eval_logs/spatial_object
# TASK=libero_spatial,libero_object
# GPU_ID=0

OUTPUTS_DIR=./eval_logs/goal_10
TASK_SUITE=libero_goal,libero_10
GPU_ID=0  # IMPORTANT: EGL only detects GPU 0, so use GPU_ID=0 for now
EPISODE=2  # run amount for each task (for success rate)
TASK_IDS='[0,1]'  # different scenes for one group

# EGL only detects 1 GPU on this system, so both scripts must use GPU_ID=0
# To run multiple evaluations in parallel, use different task_ids or task suites
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_EGL_DEVICE_ID=$GPU_ID lerobot-eval  \
 --env.type=libero      \
 --env.task=$TASK_SUITE  \
 --eval.batch_size=2    \
 --eval.n_episodes=$EPISODE   \
 --policy.path=lerobot/pi05_libero_finetuned    \
 --policy.n_action_steps=10 \
 --policy.device=cuda:$GPU_ID \
 --output_dir=$OUTPUTS_DIR  \
 --env.max_parallel_tasks=1 \
 --env.task_ids=$TASK_IDS
 
