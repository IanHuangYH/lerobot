#!/usr/bin/env python3
"""
Preserve exact object positions when adding new objects to LIBERO task.

This script:
1. Creates OLD environment from OLD BDDL (without new objects)
2. Loads OLD init states and extracts object positions BY NAME
3. Creates NEW environment from NEW BDDL (with new objects)  
4. Sets old object positions by name in new environment
5. Saves new init states with preserved positions
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
import cv2
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def extract_object_positions(env, state_vector):
    """
    Extract object positions and orientations by name from a state.
    
    Args:
        env: LIBERO environment  
        state_vector: Flattened MuJoCo state
        
    Returns:
        dict: {object_name: {'pos': [x,y,z], 'quat': [qw,qx,qy,qz], 'vel': [vx,vy,vz,wx,wy,wz]}}
    """
    # Set the state in the environment
    env.sim.set_state_from_flattened(state_vector)
    env.sim.forward()
    
    object_states = {}
    
    # Extract positions for all bodies
    for body_id in range(env.sim.model.nbody):
        body_name = env.sim.model.body_id2name(body_id)
        
        # Skip world and robot base
        if not body_name or body_name in ['world', 'robot0_base', 'robot0_link0']:
            continue
            
        # Get position and orientation from sim data
        pos = env.sim.data.body_xpos[body_id].copy()
        quat = env.sim.data.body_xquat[body_id].copy()  # [w, x, y, z]
        
        # Also get velocity if available
        vel = None
        for jnt_id in range(env.sim.model.njnt):
            jnt_name = env.sim.model.joint_id2name(jnt_id)
            if body_name in jnt_name and 'joint' in jnt_name:
                # Get qvel for this joint
                qvel_addr = env.sim.model.jnt_dofadr[jnt_id]
                nv = env.sim.model.jnt_dof(jnt_id)
                vel = env.sim.data.qvel[qvel_addr:qvel_addr+nv].copy()
                break
        
        object_states[body_name] = {
            'pos': pos,
            'quat': quat,
            'vel': vel if vel is not None else np.zeros(6)
        }
    
    return object_states


def set_object_positions(env, object_positions_dict):
    """
    Set object positions in environment by name.
    
    Args:
        env: LIBERO environment
        object_positions_dict: dict from extract_object_positions
    """
    # Get current state
    state = env.sim.get_state()
    
    # For each object we want to preserve
    for obj_name, obj_data in object_positions_dict.items():
        # Find the joint for this object
        for jnt_id in range(env.sim.model.njnt):
            jnt_name = env.sim.model.joint_id2name(jnt_id)
            
            # Match object name in joint name
            if obj_name in jnt_name and 'joint' in jnt_name:
                # Get qpos address for this joint
                qpos_addr = env.sim.model.jnt_qposadr[jnt_id]
                qvel_addr = env.sim.model.jnt_dofadr[jnt_id]
                
                # Set position (3 values) and quaternion (4 values) for freejoint
                # MuJoCo stores quaternion as [w, x, y, z]
                state.qpos[qpos_addr:qpos_addr+3] = obj_data['pos']
                state.qpos[qpos_addr+3:qpos_addr+7] = obj_data['quat']
                
                # Set velocity (6 values: 3 linear + 3 angular)
                if len(obj_data['vel']) == 6:
                    state.qvel[qvel_addr:qvel_addr+6] = obj_data['vel']
                
                break
    
    # Apply the modified state
    env.sim.set_state(state)
    env.sim.forward()


def visualize_init_states(
    bddl_path: str,
    init_states_path: str,
    output_dir: str,
    num_scenes: int = 1,
):
    """
    Visualize initial states by rendering scenes.
    
    Args:
        bddl_path: Path to BDDL file
        init_states_path: Path to init states file
        output_dir: Directory to save rendered images
        num_scenes: Number of scenes to render
    """
    print(f"\n{'='*80}")
    print("RENDERING INITIAL SCENES")
    print(f"{'='*80}\n")
    
    # Load init states
    print(f"Loading init states from: {init_states_path}")
    init_states = torch.load(init_states_path, weights_only=False)
    print(f"Total states available: {len(init_states)}")
    
    # Create environment with visualization resolution (matches LiberoEnv defaults)
    print(f"Creating environment from: {bddl_path}")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=480,  # visualization_height from LiberoEnv
        camera_widths=640,   # visualization_width from LiberoEnv
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Render scenes
    num_to_render = min(num_scenes, len(init_states))
    print(f"\nRendering {num_to_render} scenes...")
    
    for i in range(num_to_render):
        # Set state and regenerate observations
        obs = env.regenerate_obs_from_state(init_states[i])
        
        # Save agentview camera (flip both H and W for visualization, matching LiberoEnv.render())
        agentview = obs.get('agentview_image', obs.get('image', None))
        if agentview is not None:
            # Flip both dimensions for correct orientation
            agentview_flipped = agentview[::-1, ::-1]
            img_path = output_path / f"scene_{i:03d}_agentview.png"
            # Convert RGB to BGR for cv2
            cv2.imwrite(str(img_path), cv2.cvtColor(agentview_flipped, cv2.COLOR_RGB2BGR))
            print(f"  ✓ Saved: {img_path.name}")
        
        # Save eye-in-hand camera
        eye_in_hand = obs.get('robot0_eye_in_hand_image', None)
        if eye_in_hand is not None:
            img_path = output_path / f"scene_{i:03d}_wrist.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(eye_in_hand, cv2.COLOR_RGB2BGR))
    
    env.close()
    print(f"\n✅ Rendered {num_to_render} scenes to: {output_path}")
    print(f"{'='*80}\n")


def preserve_positions_add_objects(
    old_bddl_path: str,
    new_bddl_path: str,
    old_init_path: str,
    output_init_path: str,
    num_states: int = None,
):
    """
    Generate new init states preserving exact object positions.
    
    Args:
        old_bddl_path: Path to BDDL before adding objects
        new_bddl_path: Path to BDDL after adding objects
        old_init_path: Path to old init states file
        output_init_path: Where to save new init states
        num_states: Number of states to generate (default: all from old_init)
    """
    print("=" * 80)
    print("PRESERVING OBJECT POSITIONS WHEN ADDING OBJECTS")
    print("=" * 80)
    
    # Load old init states
    print(f"\n1. Loading old init states from: {old_init_path}")
    old_states = torch.load(old_init_path, weights_only=False)
    print(f"   Shape: {old_states.shape}")
    
    if num_states is None:
        num_states = len(old_states)
    
    # Create OLD environment
    print(f"\n2. Creating OLD environment from: {old_bddl_path}")
    old_env = OffScreenRenderEnv(
        bddl_file_name=old_bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    old_env.reset()
    print(f"   Old state dimension: {old_env.sim.get_state().flatten().shape}")
    
    # Create NEW environment
    print(f"\n3. Creating NEW environment from: {new_bddl_path}")
    new_env = OffScreenRenderEnv(
        bddl_file_name=new_bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    new_env.reset()
    print(f"   New state dimension: {new_env.sim.get_state().flatten().shape}")
    
    # Extract object lists
    print(f"\n4. Analyzing objects in both environments...")
    old_objects = set()
    for i in range(old_env.sim.model.nbody):
        name = old_env.sim.model.body_id2name(i)
        if name and name not in ['world', 'robot0_base']:
            old_objects.add(name)
    
    new_objects = set()
    for i in range(new_env.sim.model.nbody):
        name = new_env.sim.model.body_id2name(i)
        if name and name not in ['world', 'robot0_base']:
            new_objects.add(name)
    
    added_objects = new_objects - old_objects
    removed_objects = old_objects - new_objects
    preserved_objects = old_objects & new_objects
    
    print(f"   Objects in OLD env: {len(old_objects)}")
    print(f"   Objects in NEW env: {len(new_objects)}")
    print(f"   ➕ Added objects: {sorted(added_objects) if added_objects else 'None'}")
    print(f"   ➖ Removed objects: {sorted(removed_objects) if removed_objects else 'None'}")
    print(f"   ✓ Preserved objects: {len(preserved_objects)} items")
    
    if removed_objects:
        print(f"\n   ⚠️  WARNING: {len(removed_objects)} object(s) removed/replaced:")
        for obj in sorted(removed_objects):
            print(f"      - {obj}")
        print(f"   → Their positions will NOT be copied to new environment")
    
    # Create joint name mappings
    print(f"\n5. Creating joint mappings...")
    old_joint_map = {}  # {object_name: (qpos_addr, qvel_addr, qpos_size, qvel_size)}
    for jid in range(old_env.sim.model.njnt):
        jname = old_env.sim.model.joint_id2name(jid)
        if 'joint' in jname:
            # Extract object name from joint name (e.g., "akita_black_bowl_1_joint0" -> "akita_black_bowl_1")
            obj_name = jname.replace('_joint0', '').replace('_joint', '')
            for body_id in range(old_env.sim.model.nbody):
                body_name = old_env.sim.model.body_id2name(body_id)
                if body_name and obj_name in body_name:
                    obj_name = body_name  # Use full body name
                    break
            
            qpos_addr = old_env.sim.model.jnt_qposadr[jid]
            qvel_addr = old_env.sim.model.jnt_dofadr[jid]
            jnt_type = old_env.sim.model.jnt_type[jid]
            
            # freejoint: qpos=7 (3 pos + 4 quat), qvel=6 (3 linear + 3 angular)
            if jnt_type == 0:  # Free joint
                old_joint_map[obj_name] = (qpos_addr, qvel_addr, 7, 6)
    
    new_joint_map = {}
    for jid in range(new_env.sim.model.njnt):
        jname = new_env.sim.model.joint_id2name(jid)
        if 'joint' in jname:
            obj_name = jname.replace('_joint0', '').replace('_joint', '')
            for body_id in range(new_env.sim.model.nbody):
                body_name = new_env.sim.model.body_id2name(body_id)
                if body_name and obj_name in body_name:
                    obj_name = body_name
                    break
            
            qpos_addr = new_env.sim.model.jnt_qposadr[jid]
            qvel_addr = new_env.sim.model.jnt_dofadr[jid]
            jnt_type = new_env.sim.model.jnt_type[jid]
            
            if jnt_type == 0:
                new_joint_map[obj_name] = (qpos_addr, qvel_addr, 7, 6)
    
    print(f"   Old env joints: {len(old_joint_map)}")
    print(f"   New env joints: {len(new_joint_map)}")
    
    # Generate new states
    print(f"\n6. Generating {num_states} new states with preserved positions...")
    new_states = []
    
    for i in range(num_states):
        idx = i % len(old_states)
        old_state = old_states[idx]
        
        # Set old state in old env
        old_env.sim.set_state_from_flattened(old_state)
        old_env.sim.forward()
        
        # If there are new objects, reset new env to place them, then copy old positions
        if added_objects:
            # Reset new env to randomly place new objects
            new_env.reset()
            new_state_obj = new_env.sim.get_state()
            
            # Copy qpos and qvel ONLY for objects that exist in BOTH environments
            copied_count = 0
            skipped_count = 0
            
            for obj_name in preserved_objects:
                # Safety check: ensure object exists in both joint maps
                if obj_name not in old_joint_map:
                    skipped_count += 1
                    continue
                
                if obj_name not in new_joint_map:
                    # This shouldn't happen (obj is in preserved_objects), but be safe
                    print(f"   ⚠️  WARNING: {obj_name} in preserved set but not in new joint map!")
                    skipped_count += 1
                    continue
                
                old_qpos_addr, old_qvel_addr, old_qpos_size, old_qvel_size = old_joint_map[obj_name]
                new_qpos_addr, new_qvel_addr, new_qpos_size, new_qvel_size = new_joint_map[obj_name]
                
                # Copy qpos (position + orientation)
                new_state_obj.qpos[new_qpos_addr:new_qpos_addr+old_qpos_size] = \
                    old_env.sim.data.qpos[old_qpos_addr:old_qpos_addr+old_qpos_size].copy()
                
                # Copy qvel (velocities)
                new_state_obj.qvel[new_qvel_addr:new_qvel_addr+old_qvel_size] = \
                    old_env.sim.data.qvel[old_qvel_addr:old_qvel_addr+old_qvel_size].copy()
                
                copied_count += 1
            
            # Log what was done (only for first state)
            if i == 0:
                print(f"   State 0: Copied {copied_count} objects, Skipped {skipped_count}")
            
            # Set state and forward
            new_env.sim.set_state(new_state_obj)
            new_env.sim.forward()
            new_state = new_env.sim.get_state().flatten()
        else:
            # No new objects - just use old state directly if dimensions match
            if old_state.shape == new_env.sim.get_state().flatten().shape:
                new_state = old_state.copy()
            else:
                # Dimensions differ even without new objects - copy what we can
                new_env.reset()
                new_state_obj = new_env.sim.get_state()
                
                for obj_name in preserved_objects:
                    # Safety check: only copy if object exists in both environments
                    if obj_name not in old_joint_map or obj_name not in new_joint_map:
                        continue
                    
                    old_qpos_addr, old_qvel_addr, old_qpos_size, old_qvel_size = old_joint_map[obj_name]
                    new_qpos_addr, new_qvel_addr, new_qpos_size, new_qvel_size = new_joint_map[obj_name]
                    
                    new_state_obj.qpos[new_qpos_addr:new_qpos_addr+old_qpos_size] = \
                        old_env.sim.data.qpos[old_qpos_addr:old_qpos_addr+old_qpos_size].copy()
                    
                    new_state_obj.qvel[new_qvel_addr:new_qvel_addr+old_qvel_size] = \
                        old_env.sim.data.qvel[old_qvel_addr:old_qvel_addr+old_qvel_size].copy()
                
                new_env.sim.set_state(new_state_obj)
                new_env.sim.forward()
                new_state = new_env.sim.get_state().flatten()
        
        new_states.append(new_state)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Generated state {i+1}/{num_states}")
    
    new_states = np.array(new_states)
    
    # Save new init states
    print(f"\n6. Saving new init states to: {output_init_path}")
    output_path = Path(output_init_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup if exists
    if output_path.exists():
        backup = output_path.with_suffix('.bak')
        if not backup.exists():
            import shutil
            shutil.copy2(output_path, backup)
            print(f"   Created backup: {backup}")
    
    torch.save(new_states, output_path)
    print(f"   Saved shape: {new_states.shape}")
    
    # Also save as .init if output is .pruned_init
    if output_path.suffix == '.pruned_init':
        init_path = output_path.with_suffix('.init')
        torch.save(new_states, init_path)
        print(f"   Also saved to: {init_path}")
    
    old_env.close()
    new_env.close()
    
    print("\n" + "=" * 80)
    print(f"✅ S✓ PRESERVED positions for {len(preserved_objects)} objects")
    if added_objects:
        print(f"  ➕ NEW random placements for: {sorted(added_objects)}")
    if removed_objects:
        print(f"  ➖ REMOVED from scene: {sorted(removed_objects)}")
    print(f"Generated {num_states} init states with:")
    print(f"  • PRESERVED positions for: {sorted(preserved_objects)}")
    print(f"  • NEW placements for: {sorted(added_objects)}")
    print("=" * 80)
    
    # Visualize the first few generated states
    visualize_init_states(
        bddl_path=new_bddl_path,
        init_states_path=str(output_path),
        output_dir=str(output_path.parent / "visualizations"),
    )
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preserve object positions when adding objects to LIBERO task"
    )
    parser.add_argument("--old_bddl", type=str, required=True,
                       help="Path to old BDDL file (before adding objects)")
    parser.add_argument("--new_bddl", type=str, required=True,
                       help="Path to new BDDL file (after adding objects)")
    parser.add_argument("--old_init", type=str, required=True,
                       help="Path to old init states file")
    parser.add_argument("--output_init", type=str, required=True,
                       help="Path where to save new init states")
    parser.add_argument("--num_states", type=int, default=None,
                       help="Number of states to generate (default: all)")
    
    args = parser.parse_args()
    
    preserve_positions_add_objects(
        args.old_bddl,
        args.new_bddl,
        args.old_init,
        args.output_init,
        args.num_states
    )
