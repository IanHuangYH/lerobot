#!/bin/bash

# Configuration
GROUP=libero_object
TASK=pick_up_the_alphabet_soup_and_place_it_in_the_basket

# Variant generation parameters
N_TARGET_OBJECT=2      # Number of variants swapping pickup target
N_TARGET_LOCATION=2    # Number of variants swapping place target  
N_BOTH_OBJECT_TARGET=2 # Number of variants swapping both targets
NUM_INIT_STATES=10     # Number of init states per variant (10 for speed, 50 for production)

# Paths
BDDL_DIR=/workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/$GROUP
INIT_DIR=/workspace/lerobot/third_party/LIBERO/libero/libero/init_files/$GROUP

BDDL_FILE=$BDDL_DIR/$TASK.bddl
INIT_FILE=$INIT_DIR/$TASK.init

# Output directory for visualizations (optional)
OUTPUT_VIS_DIR=$INIT_DIR/variant_visualizations/$TASK

# Run the scene variant generation
cd /workspace/lerobot/pi_setting/eval && python3 generate_scene_variants.py \
    --bddl $BDDL_FILE \
    --init $INIT_FILE \
    --n_target_object $N_TARGET_OBJECT \
    --n_target_location $N_TARGET_LOCATION \
    --n_both_object_target $N_BOTH_OBJECT_TARGET \
    --num_init_states $NUM_INIT_STATES \
    --output_dir $OUTPUT_VIS_DIR
