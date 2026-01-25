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
import re
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def parse_bddl_init_placements(bddl_path):
    """
    Parse BDDL file to extract object -> region mappings from :init section.
    
    Returns:
        dict: {object_name: region_name} e.g., {'plate_1': 'main_table_plate_region'}
    """
    placements = {}
    
    with open(bddl_path, 'r') as f:
        content = f.read()
    
    # Find :init section
    init_match = re.search(r'\(:init\s+(.*?)\s+\)', content, re.DOTALL)
    if not init_match:
        return placements
    
    init_section = init_match.group(1)
    
    # Parse (On object_name region_name) statements
    # Example: (On plate_1 main_table_box_region)
    on_statements = re.findall(r'\(On\s+(\S+)\s+(\S+)\)', init_section)
    
    for obj_name, region_name in on_statements:
        placements[obj_name] = region_name
    
    return placements


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
    
    # Create environment with observation resolution (matches LiberoEnv defaults)
    print(f"Creating environment from: {bddl_path}")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=360,  # observation_height from LiberoEnv (default)
        camera_widths=360,   # observation_width from LiberoEnv (default)
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Render scenes
    num_to_render = min(num_scenes, len(init_states))
    print(f"\nRendering {num_to_render} scenes...")
    
    # Settle objects after loading each state (matching LiberoEnv.reset() behavior)
    dummy_action = np.zeros(7)  # LIBERO uses 7-dim actions
    
    for i in range(num_to_render):
        # Set state and let objects settle
        env.set_init_state(init_states[i])
        for _ in range(10):  # num_steps_wait=10 from LiberoEnv
            env.env.step(dummy_action)
        
        # Regenerate observations after settling
        obs = env.env._get_observations()
        
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
    
    # Create OLD environment (use low resolution for speed - camera doesn't affect physics)
    print(f"\n2. Creating OLD environment from: {old_bddl_path}")
    old_env = OffScreenRenderEnv(
        bddl_file_name=old_bddl_path,
        camera_heights=128,  # Low res for speed (camera doesn't affect physics)
        camera_widths=128,
    )
    old_env.reset()
    print(f"   Old state dimension: {old_env.sim.get_state().flatten().shape}")
    
    # Create NEW environment (use low resolution for speed - camera doesn't affect physics)
    print(f"\n3. Creating NEW environment from: {new_bddl_path}")
    new_env = OffScreenRenderEnv(
        bddl_file_name=new_bddl_path,
        camera_heights=128,  # Low res for speed (camera doesn't affect physics)
        camera_widths=128,
    )
    new_env.reset()
    print(f"   New state dimension: {new_env.sim.get_state().flatten().shape}")
    
    # Parse BDDL files to get object -> region mappings
    print(f"\n4. Parsing BDDL files to detect region changes...")
    old_placements = parse_bddl_init_placements(old_bddl_path)
    new_placements = parse_bddl_init_placements(new_bddl_path)
    
    print(f"   Old BDDL placements: {len(old_placements)} objects")
    print(f"   New BDDL placements: {len(new_placements)} objects")
    
    # Extract object lists
    print(f"\n5. Analyzing objects in both environments...")
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
    common_objects = old_objects & new_objects
    
    # Helper function to match BDDL names to MuJoCo body names
    def find_mujoco_body_name(bddl_name, mujoco_bodies):
        """
        Match BDDL object name (e.g., 'plate_1') to MuJoCo body name (e.g., 'plate_1_main').
        """
        # Direct match
        if bddl_name in mujoco_bodies:
            return bddl_name
        
        # Try with common MuJoCo suffixes
        for suffix in ['_main', '_body', '_base']:
            candidate = bddl_name + suffix
            if candidate in mujoco_bodies:
                return candidate
        
        # Partial match: find body that starts with bddl_name
        for body in mujoco_bodies:
            if body.startswith(bddl_name + '_'):
                return body
        
        return None
    
    # Detect which common objects have DIFFERENT regions (should get NEW positions)
    region_changed_objects = set()
    bddl_to_mujoco_mapping = {}  # Track BDDL name → MuJoCo body name
    
    for bddl_name in old_placements.keys():
        old_region = old_placements.get(bddl_name)
        new_region = new_placements.get(bddl_name)
        
        # Find the corresponding MuJoCo body name
        mujoco_name = find_mujoco_body_name(bddl_name, common_objects)
        
        if mujoco_name:
            bddl_to_mujoco_mapping[bddl_name] = mujoco_name
            
            if old_region and new_region and old_region != new_region:
                region_changed_objects.add(mujoco_name)  # Use MuJoCo name
    
    # Objects to preserve = common objects WITHOUT region changes
    # Only preserve objects that are in BDDL (have placement info)
    mujoco_names_in_bddl = set(bddl_to_mujoco_mapping.values())
    preserved_objects = mujoco_names_in_bddl - region_changed_objects
    
    print(f"   Objects in OLD env: {len(old_objects)}")
    print(f"   Objects in NEW env: {len(new_objects)}")
    print(f"   BDDL-declared objects: {len(bddl_to_mujoco_mapping)}")
    print(f"   ➕ Added objects: {sorted(added_objects) if added_objects else 'None'}")
    print(f"   ➖ Removed objects: {sorted(removed_objects) if removed_objects else 'None'}")
    print(f"   🔄 Region changed: {sorted(region_changed_objects) if region_changed_objects else 'None'}")
    print(f"   ✓ Preserved (same region): {len(preserved_objects)} items")
    
    if region_changed_objects:
        print(f"\n   📍 Region Changes Detected:")
        for mujoco_name in sorted(region_changed_objects):
            # Find the BDDL name for this MuJoCo body
            bddl_name = None
            for bname, mname in bddl_to_mujoco_mapping.items():
                if mname == mujoco_name:
                    bddl_name = bname
                    break
            
            if bddl_name:
                old_region = old_placements.get(bddl_name, 'unknown')
                new_region = new_placements.get(bddl_name, 'unknown')
                print(f"      {bddl_name} (MuJoCo: {mujoco_name}):")
                print(f"         OLD: {old_region}")
                print(f"         NEW: {new_region}")
        print(f"   → These {len(region_changed_objects)} object(s) will get NEW random positions")
    
    if removed_objects:
        print(f"\n   ⚠️  WARNING: {len(removed_objects)} object(s) removed/replaced:")
        for obj in sorted(removed_objects):
            print(f"      - {obj}")
        print(f"   → Their positions will NOT be copied to new environment")
    
    # Create joint name mappings
    print(f"\n6. Creating joint mappings...")
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
    print(f"\n7. Generating {num_states} new states with preserved positions...")
    new_states = []
    
    # Optimization: Get template state for new objects/region-changed objects ONCE
    template_qpos = None
    template_qvel = None
    if added_objects or region_changed_objects:
        print(f"   Creating template for new object placements...")
        new_env.reset()
        
        # Settle objects ONCE to get stable random positions for new/changed objects
        # This matches LiberoEnv.reset() behavior (num_steps_wait=10)
        dummy_action = np.zeros(7)
        for _ in range(10):
            new_env.env.step(dummy_action)
        
        template_state = new_env.sim.get_state()
        # Copy the position and velocity arrays AFTER settling
        template_qpos = template_state.qpos.copy()
        template_qvel = template_state.qvel.copy()
        print(f"   ✓ Template created with settled positions (will reuse for all {num_states} states)")
    
    # Pre-compute copying indices for efficiency
    copy_plan = []  # [(old_qpos_start, old_qpos_end, new_qpos_start, new_qpos_end, old_qvel_start, old_qvel_end, new_qvel_start, new_qvel_end)]
    
    for obj_name in preserved_objects:
        if obj_name in old_joint_map and obj_name in new_joint_map:
            old_qpos_addr, old_qvel_addr, old_qpos_size, old_qvel_size = old_joint_map[obj_name]
            new_qpos_addr, new_qvel_addr, new_qpos_size, new_qvel_size = new_joint_map[obj_name]
            
            copy_plan.append((
                old_qpos_addr, old_qpos_addr + old_qpos_size,
                new_qpos_addr, new_qpos_addr + new_qpos_size,
                old_qvel_addr, old_qvel_addr + old_qvel_size,
                new_qvel_addr, new_qvel_addr + new_qvel_size
            ))
    
    if copy_plan:
        print(f"   ✓ Copy plan created for {len(copy_plan)} objects")
    
    for i in range(num_states):
        idx = i % len(old_states)
        old_state = old_states[idx]
        
        # Set old state in old env
        old_env.sim.set_state_from_flattened(old_state)
        old_env.sim.forward()
        
        # If there are new objects OR region changes, use template and copy old positions
        if added_objects or region_changed_objects:
            # Get a fresh state object from new env
            new_state_obj = new_env.sim.get_state()
            
            # Start with template (contains random placements for new/changed objects)
            new_state_obj.qpos[:] = template_qpos
            new_state_obj.qvel[:] = template_qvel
            
            # Batch copy all preserved objects using pre-computed indices
            for old_qpos_s, old_qpos_e, new_qpos_s, new_qpos_e, old_qvel_s, old_qvel_e, new_qvel_s, new_qvel_e in copy_plan:
                new_state_obj.qpos[new_qpos_s:new_qpos_e] = old_env.sim.data.qpos[old_qpos_s:old_qpos_e]
                new_state_obj.qvel[new_qvel_s:new_qvel_e] = old_env.sim.data.qvel[old_qvel_s:old_qvel_e]
            
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
                
                for old_qpos_s, old_qpos_e, new_qpos_s, new_qpos_e, old_qvel_s, old_qvel_e, new_qvel_s, new_qvel_e in copy_plan:
                    new_state_obj.qpos[new_qpos_s:new_qpos_e] = old_env.sim.data.qpos[old_qpos_s:old_qpos_e]
                    new_state_obj.qvel[new_qvel_s:new_qvel_e] = old_env.sim.data.qvel[old_qvel_s:old_qvel_e]
                
                new_env.sim.set_state(new_state_obj)
                new_env.sim.forward()
                new_state = new_env.sim.get_state().flatten()
        
        new_states.append(new_state)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Generated state {i+1}/{num_states}")
    
    new_states = np.array(new_states)
    
    # Save new init states
    print(f"\n8. Saving new init states to: {output_init_path}")
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
    print("✅ SUCCESS!")
    print("=" * 80)
    print(f"Generated {num_states} init states with:")
    print(f"  ✓ PRESERVED positions for {len(preserved_objects)} objects:")
    if preserved_objects:
        for mujoco_name in sorted(preserved_objects):
            # Find BDDL name
            bddl_name = None
            for bname, mname in bddl_to_mujoco_mapping.items():
                if mname == mujoco_name:
                    bddl_name = bname
                    break
            if bddl_name:
                region = new_placements.get(bddl_name, 'unknown')
                print(f"      - {bddl_name} (MuJoCo: {mujoco_name}) @ {region}")
            else:
                print(f"      - {mujoco_name} @ unknown")
    
    if added_objects:
        print(f"\n  ➕ NEW random placements for {len(added_objects)} objects:")
        for obj in sorted(added_objects):
            print(f"      - {obj}")
    
    if region_changed_objects:
        print(f"\n  🔄 NEW random placements for {len(region_changed_objects)} objects (region changed):")
        for mujoco_name in sorted(region_changed_objects):
            # Find BDDL name
            bddl_name = None
            for bname, mname in bddl_to_mujoco_mapping.items():
                if mname == mujoco_name:
                    bddl_name = bname
                    break
            if bddl_name:
                old_region = old_placements.get(bddl_name, 'unknown')
                new_region = new_placements.get(bddl_name, 'unknown')
                print(f"      - {bddl_name} (MuJoCo: {mujoco_name}): {old_region} → {new_region}")
            else:
                print(f"      - {mujoco_name}")
    
    if removed_objects:
        print(f"\n  ➖ REMOVED from scene ({len(removed_objects)} objects):")
        for obj in sorted(removed_objects):
            print(f"      - {obj}")
    
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
