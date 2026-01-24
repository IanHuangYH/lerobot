
 
GROUP=libero_object
TASK=pick_up_the_alphabet_soup_and_place_it_in_the_basket

# spatial_0, GROUP=libero_spatial, TASK=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate

# spatial_1, GROUP=libero_spatial, TASK=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate

# object_0, GROUP=libero_object, TASK=pick_up_the_alphabet_soup_and_place_it_in_the_basket


cd /workspace/lerobot/pi_setting/eval && python preserve_exact_positions.py \
    --old_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/$GROUP/$TASK.bddl.old \
    --new_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/$GROUP/$TASK.bddl \
    --old_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/$GROUP/$TASK.bak \
    --output_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/$GROUP/$TASK.pruned_init

    