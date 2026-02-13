
 
#!/bin/bash
# =============================================================================
# LIBERO Custom BDDL File with Preserved Scene Positions
# =============================================================================
#
# PURPOSE:
#   When you customize a BDDL file (add/remove/modify objects), generate a new
#   init file that preserves the EXACT positions of unchanged objects from the
#   original scene. This minimizes disruption to the existing scene layout.
#
# WHAT IT DOES:
#   1. Compares old BDDL (original) with new BDDL (your customized version)
#   2. Identifies which objects were:
#      - Kept unchanged (preserve their exact positions)
#      - Added (generate new positions for these)
#      - Removed (delete their positions)
#   3. Creates new init file that:
#      - Keeps exact xyz, quaternion, velocity for unchanged objects
#      - Generates new positions for added objects (via env reset)
#      - Maintains same MuJoCo state structure
#
# USE CASES:
#   - Add distractor objects to existing task without moving original objects
#   - Remove objects while keeping rest of scene identical
#   - Modify object properties (size, color) without repositioning everything
#   - Create minimal-change variants for controlled experiments
#
# WORKFLOW:
#   1. Backup original files:
#      cp task.bddl task.bddl.old
#      cp task.init task.bak
#   
#   2. Edit task.bddl with your changes (add/remove objects, change regions, etc.)
#   
#   3. Run this script to generate new init file
#   
#   4. New init file (task.pruned_init) has:
#      - Original object positions preserved exactly
#      - New objects placed via physics engine
#
# EXAMPLE:
#   Original: alphabet_soup, basket, plate (3 objects)
#   Modified: alphabet_soup, basket, plate, butter, milk (5 objects - added 2)
#   Result: alphabet_soup, basket, plate stay at EXACT same positions
#           butter, milk generated at new positions
#
# INPUT FILES:
#   --old_bddl: Original BDDL file (before modifications)
#   --new_bddl: Modified BDDL file (your customized version)
#   --old_init: Original init file (backup of positions to preserve)
#
# OUTPUT:
#   --output_init: New init file with preserved positions + new objects
#
# =============================================================================

GROUP=libero_object
TASK=pick_up_the_alphabet_soup_and_place_it_in_the_basket

# EXAMPLES OF OTHER TASKS:
# spatial_0, GROUP=libero_spatial, TASK=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
# spatial_1, GROUP=libero_spatial, TASK=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
# object_0, GROUP=libero_object, TASK=pick_up_the_alphabet_soup_and_place_it_in_the_basket


cd /workspace/lerobot/pi_setting/eval && python preserve_exact_positions.py \
    --old_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/$GROUP/$TASK.bddl.old \
    --new_bddl /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/$GROUP/$TASK.bddl \
    --old_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/$GROUP/$TASK.bak \
    --output_init /workspace/lerobot/third_party/LIBERO/libero/libero/init_files/$GROUP/$TASK.pruned_init

    