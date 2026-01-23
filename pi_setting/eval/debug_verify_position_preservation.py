#!/usr/bin/env python3
"""
Verify that object positions were actually preserved when adding objects.
"""

import torch
import numpy as np
from libero.libero.envs import OffScreenRenderEnv


def compare_positions(
    old_bddl_path,
    new_bddl_path,
    old_init_path,
    new_init_path,
    num_states_to_check=5
):
    """
    Compare object positions between old and new init states.
    """
    print("=" * 80)
    print("VERIFYING POSITION PRESERVATION")
    print("=" * 80)
    
    # Load init states
    print(f"\n1. Loading init states...")
    old_states = torch.load(old_init_path, weights_only=False)
    new_states = torch.load(new_init_path, weights_only=False)
    print(f"   Old states: {old_states.shape}")
    print(f"   New states: {new_states.shape}")
    
    # Create environments
    print(f"\n2. Creating environments...")
    old_env = OffScreenRenderEnv(
        bddl_file_name=old_bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    old_env.reset()
    
    new_env = OffScreenRenderEnv(
        bddl_file_name=new_bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    new_env.reset()
    
    # Get object names from old environment
    print(f"\n3. Identifying objects to compare...")
    old_object_names = []
    for i in range(old_env.sim.model.nbody):
        name = old_env.sim.model.body_id2name(i)
        if name and name not in ['world', 'robot0_base', 'robot0_link0']:
            # Focus on main objects (bowls, plate, etc.)
            if any(keyword in name for keyword in ['bowl', 'plate', 'cookie', 'ramekin', 'book']):
                old_object_names.append(name)
    
    print(f"   Will compare {len(old_object_names)} key objects:")
    for name in old_object_names:
        print(f"     - {name}")
    
    # Compare positions for first N states
    print(f"\n4. Comparing positions for first {num_states_to_check} states...")
    print("=" * 80)
    
    all_differences = []
    
    for state_idx in range(min(num_states_to_check, len(old_states), len(new_states))):
        print(f"\n--- STATE {state_idx} ---")
        
        # Load old state
        old_env.sim.set_state_from_flattened(old_states[state_idx])
        old_env.sim.forward()
        
        # Load new state
        new_env.sim.set_state_from_flattened(new_states[state_idx])
        new_env.sim.forward()
        
        # Compare each object
        state_diffs = {}
        for obj_name in old_object_names:
            # Get position from old environment
            old_body_id = None
            for i in range(old_env.sim.model.nbody):
                if old_env.sim.model.body_id2name(i) == obj_name:
                    old_body_id = i
                    break
            
            # Get position from new environment
            new_body_id = None
            for i in range(new_env.sim.model.nbody):
                if new_env.sim.model.body_id2name(i) == obj_name:
                    new_body_id = i
                    break
            
            if old_body_id is not None and new_body_id is not None:
                old_pos = old_env.sim.data.body_xpos[old_body_id]
                new_pos = new_env.sim.data.body_xpos[new_body_id]
                
                old_quat = old_env.sim.data.body_xquat[old_body_id]
                new_quat = new_env.sim.data.body_xquat[new_body_id]
                
                # Calculate differences
                pos_diff = np.linalg.norm(old_pos - new_pos)
                quat_diff = np.linalg.norm(old_quat - new_quat)
                
                state_diffs[obj_name] = {
                    'pos_diff': pos_diff,
                    'quat_diff': quat_diff,
                    'old_pos': old_pos.copy(),
                    'new_pos': new_pos.copy(),
                    'old_quat': old_quat.copy(),
                    'new_quat': new_quat.copy()
                }
                
                # Print if significant difference
                if pos_diff > 0.001 or quat_diff > 0.001:
                    print(f"\n  ⚠️  {obj_name}:")
                    print(f"      Position diff: {pos_diff:.6f} m")
                    print(f"      Old pos: {old_pos}")
                    print(f"      New pos: {new_pos}")
                    if quat_diff > 0.001:
                        print(f"      Quaternion diff: {quat_diff:.6f}")
        
        all_differences.append(state_diffs)
        
        # Summary for this state
        max_pos_diff = max([d['pos_diff'] for d in state_diffs.values()])
        print(f"\n  Max position difference: {max_pos_diff:.6f} m")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for obj_name in old_object_names:
        pos_diffs = [state_diffs[obj_name]['pos_diff'] 
                     for state_diffs in all_differences 
                     if obj_name in state_diffs]
        
        if pos_diffs:
            avg_diff = np.mean(pos_diffs)
            max_diff = np.max(pos_diffs)
            
            status = "✅" if max_diff < 0.001 else "⚠️"
            print(f"{status} {obj_name:40s} avg: {avg_diff:.6f}m, max: {max_diff:.6f}m")
    
    old_env.close()
    new_env.close()
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("• Differences < 0.001m (1mm): Excellent - likely just floating point")
    print("• Differences 0.001-0.01m (1-10mm): Good - minor physics settling")
    print("• Differences > 0.01m (>10mm): May indicate position not preserved")
    print("=" * 80)


if __name__ == "__main__":
    old_bddl = "/workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl.old"
    new_bddl = "/workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
    old_init = "/workspace/lerobot/third_party/LIBERO/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bak"
    new_init = "/workspace/lerobot/third_party/LIBERO/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init"
    
    compare_positions(old_bddl, new_bddl, old_init, new_init, num_states_to_check=10)
