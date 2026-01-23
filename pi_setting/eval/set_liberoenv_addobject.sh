
# spatial_0
# TASK=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
# cd /workspace/lerobot/pi_setting/eval && python preserve_exact_positions.py \
#     --old_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/$TASK.bddl.old \
#     --new_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/$TASK.bddl \
#     --old_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/libero_spatial/$TASK.bak \
#     --output_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/libero_spatial/$TASK.pruned_init


# spatial_1
TASK=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
cd /workspace/lerobot/pi_setting/eval && python preserve_exact_positions.py \
    --old_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/$TASK.bddl.old \
    --new_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/$TASK.bddl \
    --old_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/libero_spatial/$TASK.bak \
    --output_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/libero_spatial/$TASK.pruned_init