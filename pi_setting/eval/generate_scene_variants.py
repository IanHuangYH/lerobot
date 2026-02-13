#!/usr/bin/env python3
"""
Generate scene variants by swapping object positions in LIBERO tasks.

This script creates multiple BDDL and init file variants by swapping:
1. Target pickup object with non-interest objects
2. Target place location with non-interest objects  
3. Both target objects with pairs of non-interest objects

Each variant maintains the same physical init states but swaps which object is where.
"""

import argparse
import os
import re
from pathlib import Path
import torch
import numpy as np
import cv2
from libero.libero.envs import OffScreenRenderEnv


def parse_bddl_file(bddl_path):
    """
    Parse BDDL file to extract objects and regions.
    
    Returns:
        dict with keys: 'objects', 'obj_of_interest', 'init_placements', 'region_sizes', 'full_content'
    """
    with open(bddl_path, 'r') as f:
        content = f.read()
    
    # Extract objects section
    objects_match = re.search(r'\(:objects\s+(.*?)\s+\)', content, re.DOTALL)
    objects = {}
    if objects_match:
        objects_text = objects_match.group(1)
        # Parse lines like "alphabet_soup_1 - alphabet_soup"
        for line in objects_text.strip().split('\n'):
            line = line.strip()
            if line and '-' in line:
                obj_name, obj_type = line.split('-')
                objects[obj_name.strip()] = obj_type.strip()
    
    # Extract obj_of_interest section
    interest_match = re.search(r'\(:obj_of_interest\s+(.*?)\s+\)', content, re.DOTALL)
    obj_of_interest = []
    if interest_match:
        interest_text = interest_match.group(1)
        obj_of_interest = [obj.strip() for obj in interest_text.strip().split('\n') if obj.strip()]
    
    # Extract regions and calculate sizes
    regions_match = re.search(r'\(:regions\s+(.*?)\s+\)\s+\(:fixtures', content, re.DOTALL)
    region_sizes = {}
    if regions_match:
        regions_text = regions_match.group(1)
        # Parse region definitions with ranges
        region_blocks = re.findall(r'\((\w+).*?\(:ranges\s+\((.*?)\)\s+\)', regions_text, re.DOTALL)
        for region_name, ranges_text in region_blocks:
            # Parse range tuples like (-0.01 0.25 0.01 0.27)
            ranges = re.findall(r'\(([-\d\.e\s]+)\)', ranges_text)
            if ranges:
                # Take first range (most regions have one range)
                coords = [float(x) for x in ranges[0].split()]
                if len(coords) == 4:
                    x_min, y_min, x_max, y_max = coords
                    area = abs(x_max - x_min) * abs(y_max - y_min)
                    region_sizes[f'floor_{region_name}'] = area
    
    # Extract init placements
    init_match = re.search(r'\(:init\s+(.*?)\s+\)', content, re.DOTALL)
    init_placements = {}
    if init_match:
        init_section = init_match.group(1)
        # Parse (On object_name region_name)
        on_statements = re.findall(r'\(On\s+(\S+)\s+(\S+)\)', init_section)
        for obj_name, region_name in on_statements:
            init_placements[obj_name] = region_name
    
    return {
        'objects': objects,
        'obj_of_interest': obj_of_interest,
        'init_placements': init_placements,
        'region_sizes': region_sizes,
        'full_content': content
    }


def create_variant_bddl(original_bddl_path, output_bddl_path, object_swaps):
    """
    Create a variant BDDL file by swapping object→region assignments.
    
    Args:
        original_bddl_path: Path to original BDDL
        output_bddl_path: Path to save variant BDDL
        object_swaps: dict {obj1: region_of_obj2, obj2: region_of_obj1, ...}
                     Swaps the regions between objects
    """
    with open(original_bddl_path, 'r') as f:
        content = f.read()
    
    # Replace each (On object region) statement
    for obj_name, new_region in object_swaps.items():
        # Find and replace the specific On statement for this object
        pattern = rf'\(On\s+{re.escape(obj_name)}\s+\S+\)'
        replacement = f'(On {obj_name} {new_region})'
        content = re.sub(pattern, replacement, content)
    
    # Write variant BDDL
    with open(output_bddl_path, 'w') as f:
        f.write(content)
    
    print(f"   Created variant BDDL: {Path(output_bddl_path).name}")


def regenerate_init_states(variant_bddl_path, num_states=10, deterministic=False, base_seed=42):
    """
    Regenerate init states by resetting environment multiple times.
    
    This is safer than swapping positions, especially when objects have different sizes.
    The physics engine properly places objects within their assigned regions.
    
    Args:
        variant_bddl_path: Path to variant BDDL file
        num_states: Number of init states to generate (default: 10 for speed)
        deterministic: If True, use fixed random seeds for reproducibility (default: False)
        base_seed: Base seed for deterministic generation (default: 42)
    
    Returns:
        numpy array of init states (num_states, state_dim)
    """
    # Set deterministic seeds if requested
    if deterministic:
        import random
        np.random.seed(base_seed)
        random.seed(base_seed)
    
    # Create environment with the variant BDDL
    env = OffScreenRenderEnv(
        bddl_file_name=variant_bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    
    # Seed the environment if deterministic
    if deterministic and hasattr(env, 'seed'):
        env.seed(base_seed)
    
    # Generate init states by resetting
    init_states = []
    dummy_action = np.zeros(7)
    
    for i in range(num_states):
        # Set seed for this specific state if deterministic
        if deterministic:
            state_seed = base_seed + i
            np.random.seed(state_seed)
            if hasattr(env, 'seed'):
                env.seed(state_seed)
        
        # Reset environment (generates random placement within regions)
        env.reset()
        
        # Settle objects with reduced steps (5 instead of 10 for speed)
        for _ in range(5):
            env.env.step(dummy_action)
        
        # Get settled state
        state = env.sim.get_state().flatten()
        init_states.append(state)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"      Generated init state {i+1}/{num_states}")
    
    env.close()
    
    return np.array(init_states)


def visualize_variant(bddl_path, init_path, output_dir, variant_idx):
    """
    Visualize a single variant.
    Try all states to find one with maximum object visibility (using image variance as proxy).
    """
    # Load init states
    init_states = torch.load(init_path, weights_only=False)
    
    # Create environment
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=360,
        camera_widths=360,
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Try all states and pick the one with most visible content
    # Use image variance as a proxy for object visibility (more objects = more variance)
    best_state_idx = 0
    best_variance = -1
    
    for state_idx in range(len(init_states)):
        env.set_init_state(init_states[state_idx])
        dummy_action = np.zeros(7)
        for _ in range(5):  # Reduced settling steps for speed
            env.env.step(dummy_action)
        
        obs = env.env._get_observations()
        agentview = obs.get('agentview_image', obs.get('image', None))
        
        if agentview is not None:
            # Calculate variance - higher variance typically means more objects in view
            variance = np.var(agentview)
            if variance > best_variance:
                best_variance = variance
                best_state_idx = state_idx
    
    # Render the best state
    env.set_init_state(init_states[best_state_idx])
    dummy_action = np.zeros(7)
    for _ in range(5):
        env.env.step(dummy_action)
    
    obs = env.env._get_observations()
    
    # Save agentview image
    agentview = obs.get('agentview_image', obs.get('image', None))
    if agentview is not None:
        agentview_flipped = agentview[::-1, ::-1]
        img_path = output_path / f"variant_{variant_idx:03d}_agentview.png"
        cv2.imwrite(str(img_path), cv2.cvtColor(agentview_flipped, cv2.COLOR_RGB2BGR))
    
    env.close()
    return output_path


def find_mujoco_body_name(bddl_name, env):
    """Find MuJoCo body name corresponding to BDDL object name."""
    # Get all body names
    bodies = []
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i)
        if name and name not in ['world', 'robot0_base']:
            bodies.append(name)
    
    # Direct match
    if bddl_name in bodies:
        return bddl_name
    
    # Try with suffixes
    for suffix in ['_main', '_body', '_base']:
        candidate = bddl_name + suffix
        if candidate in bodies:
            return candidate
    
    # Partial match
    for body in bodies:
        if body.startswith(bddl_name + '_'):
            return body
    
    return None


def generate_scene_variants(
    bddl_path,
    init_path,
    n_target_object,
    n_target_location,
    n_both_object_target,
    output_dir=None,
    num_init_states=10,
    create_variant_0=True,
    deterministic=False
):
    """
    Generate scene variants by swapping object positions.
    
    Args:
        bddl_path: Path to original BDDL file
        init_path: Path to original init states file
        n_target_object: Number of variants swapping pickup target
        n_target_location: Number of variants swapping place target
        n_both_object_target: Number of variants swapping both
        output_dir: Directory for visualizations (default: same as bddl_path)
        num_init_states: Number of init states per variant (default: 10 for speed)
        create_variant_0: Create variant_0 as copy of original (default: True)
        deterministic: If True, use fixed random seeds for reproducibility (default: False)
    """
    print("=" * 80)
    print("GENERATING SCENE VARIANTS")
    print("=" * 80)
    
    # Parse BDDL
    print(f"\n1. Parsing BDDL file: {Path(bddl_path).name}")
    bddl_data = parse_bddl_file(bddl_path)
    
    objects = bddl_data['objects']
    obj_of_interest = bddl_data['obj_of_interest']
    init_placements = bddl_data['init_placements']
    region_sizes = bddl_data['region_sizes']
    
    print(f"   Total objects: {len(objects)}")
    print(f"   Objects of interest: {obj_of_interest}")
    print(f"   Region sizes: {len(region_sizes)} regions")
    
    # Identify target objects
    if len(obj_of_interest) < 2:
        raise ValueError(f"Need at least 2 objects of interest, found {len(obj_of_interest)}")
    
    pickup_target = obj_of_interest[0]  # First is pickup target (e.g., alphabet_soup_1)
    place_target = obj_of_interest[1]   # Second is place target (e.g., basket_1)
    
    print(f"   Pickup target: {pickup_target}")
    print(f"   Place target: {place_target}")
    
    # Get non-interest objects
    non_interest = [obj for obj in objects.keys() if obj not in obj_of_interest]
    print(f"   Non-interest objects ({len(non_interest)}): {non_interest}")
    
    # Validate sufficient objects
    if len(non_interest) < n_target_object:
        raise ValueError(f"Need {n_target_object} non-interest objects for n_target_object, only have {len(non_interest)}")
    if len(non_interest) < n_target_location:
        raise ValueError(f"Need {n_target_location} non-interest objects for n_target_location, only have {len(non_interest)}")
    if len(non_interest) < n_both_object_target * 2:
        raise ValueError(f"Need {n_both_object_target*2} non-interest objects for n_both_object_target, only have {len(non_interest)}")
    
    # Load init states
    print(f"\n2. Loading init states from: {Path(init_path).name}")
    init_states = torch.load(init_path, weights_only=False)
    print(f"   Shape: {init_states.shape}")
    
    # Create environment to map BDDL names to MuJoCo names
    print(f"\n3. Creating environment to map object names...")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    env.reset()
    
    # Build BDDL to MuJoCo mapping
    bddl_to_mujoco = {}
    for bddl_name in objects.keys():
        mujoco_name = find_mujoco_body_name(bddl_name, env)
        if mujoco_name:
            bddl_to_mujoco[bddl_name] = mujoco_name
    
    print(f"   Mapped {len(bddl_to_mujoco)} objects")
    
    # Prepare output paths
    bddl_dir = Path(bddl_path).parent
    init_dir = Path(init_path).parent
    base_name = Path(bddl_path).stem
    
    if output_dir is None:
        output_dir = init_dir / "variant_visualizations" / base_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create variant_0 (copy of original) if requested
    if create_variant_0:
        print(f"\n4. Creating variant_0 (original scene)")
        variant_0_bddl = bddl_dir / f"{base_name}_0.bddl"
        variant_0_init = init_dir / f"{base_name}_0.init"
        variant_0_pruned = init_dir / f"{base_name}_0.pruned_init"
        
        import shutil
        shutil.copy2(bddl_path, variant_0_bddl)
        shutil.copy2(init_path, variant_0_init)
        shutil.copy2(init_path, variant_0_pruned)
        
        print(f"   Created: {variant_0_bddl.name}")
        print(f"   Created: {variant_0_init.name}")
        print(f"   Created: {variant_0_pruned.name}")
    
    variant_idx = 1
    all_variants = []
    
    # Track used objects to avoid reuse within groups
    used_in_target_object = set()
    used_in_target_location = set()
    used_pairs_in_both = set()
    
    # Generate variants for n_target_object (swap pickup target)
    print(f"\n5. Generating {n_target_object} variants: swapping pickup target")
    for i in range(n_target_object):
        # Select non-interest object not used in this group
        available = [obj for obj in non_interest if obj not in used_in_target_object]
        if not available:
            raise ValueError(f"Ran out of non-interest objects for n_target_object at variant {i+1}")
        
        swap_obj = available[i]
        used_in_target_object.add(swap_obj)
        
        # Create BDDL with swapped regions
        pickup_region = init_placements[pickup_target]
        swap_region = init_placements[swap_obj]
        
        object_swaps = {
            pickup_target: swap_region,
            swap_obj: pickup_region
        }
        
        variant_bddl_path = bddl_dir / f"{base_name}_{variant_idx}.bddl"
        create_variant_bddl(bddl_path, variant_bddl_path, object_swaps)
        
        # Regenerate init states (safer than swapping, handles different object sizes)
        print(f"   Generating {len(init_states)} init states for variant {variant_idx}...")
        variant_init_states = regenerate_init_states(
            str(variant_bddl_path), 
            num_states=num_init_states,
            deterministic=deterministic,
            base_seed=100 + variant_idx  # Different seed per variant
        )
        
        variant_init_path = init_dir / f"{base_name}_{variant_idx}.init"
        torch.save(variant_init_states, variant_init_path)
        
        # Also save as .pruned_init
        variant_pruned_path = init_dir / f"{base_name}_{variant_idx}.pruned_init"
        torch.save(variant_init_states, variant_pruned_path)
        
        print(f"   Variant {variant_idx}: {pickup_target} ↔ {swap_obj}")
        
        all_variants.append({
            'idx': variant_idx,
            'type': 'target_object',
            'bddl': variant_bddl_path,
            'init': variant_init_path,
            'swap': f"{pickup_target} ↔ {swap_obj}"
        })
        
        variant_idx += 1
    
    # Generate variants for n_target_location (swap place target)
    print(f"\n6. Generating {n_target_location} variants: swapping place target")
    for i in range(n_target_location):
        # Select non-interest object not used in this group
        # Sort by region size (largest first) to prioritize visible regions for basket
        available = [obj for obj in non_interest if obj not in used_in_target_location]
        if not available:
            raise ValueError(f"Ran out of non-interest objects for n_target_location at variant {i+1}")
        
        # Sort by region size - prioritize larger regions for the basket
        available_sorted = sorted(available, 
                                 key=lambda obj: region_sizes.get(init_placements.get(obj, ''), 0), 
                                 reverse=True)
        swap_obj = available_sorted[i]
        used_in_target_location.add(swap_obj)
        
        # Create BDDL with swapped regions
        place_region = init_placements[place_target]
        swap_region = init_placements[swap_obj]
        
        object_swaps = {
            place_target: swap_region,
            swap_obj: place_region
        }
        
        variant_bddl_path = bddl_dir / f"{base_name}_{variant_idx}.bddl"
        create_variant_bddl(bddl_path, variant_bddl_path, object_swaps)
        
        # Regenerate init states (safer than swapping, handles different object sizes)
        print(f"   Generating {num_init_states} init states for variant {variant_idx}...")
        variant_init_states = regenerate_init_states(
            str(variant_bddl_path), 
            num_states=num_init_states,
            deterministic=deterministic,
            base_seed=100 + variant_idx  # Different seed per variant
        )
        
        variant_init_path = init_dir / f"{base_name}_{variant_idx}.init"
        torch.save(variant_init_states, variant_init_path)
        
        variant_pruned_path = init_dir / f"{base_name}_{variant_idx}.pruned_init"
        torch.save(variant_init_states, variant_pruned_path)
        
        print(f"   Variant {variant_idx}: {place_target} ↔ {swap_obj}")
        
        all_variants.append({
            'idx': variant_idx,
            'type': 'target_location',
            'bddl': variant_bddl_path,
            'init': variant_init_path,
            'swap': f"{place_target} ↔ {swap_obj}"
        })
        
        variant_idx += 1
    
    # Generate variants for n_both_object_target (swap both)
    print(f"\n7. Generating {n_both_object_target} variants: swapping both targets")
    for i in range(n_both_object_target):
        # Select two non-interest objects (pairs must be unique)
        # Sort by region size to prioritize larger regions for basket (swap_obj2)
        available = sorted(non_interest, 
                          key=lambda obj: region_sizes.get(init_placements.get(obj, ''), 0), 
                          reverse=True)
        
        # Try to find a pair not used before
        # Prioritize pairs where second object (for basket) has large region
        found_pair = False
        for j in range(len(available)):
            for k in range(j+1, len(available)):
                pair = tuple(sorted([available[j], available[k]]))
                if pair not in used_pairs_in_both:
                    # Assign larger region object to swap_obj2 (for basket)
                    obj_j_size = region_sizes.get(init_placements.get(available[j], ''), 0)
                    obj_k_size = region_sizes.get(init_placements.get(available[k], ''), 0)
                    if obj_k_size > obj_j_size:
                        swap_obj1, swap_obj2 = available[j], available[k]
                    else:
                        swap_obj1, swap_obj2 = available[k], available[j]
                    used_pairs_in_both.add(pair)
                    found_pair = True
                    break
            if found_pair:
                break
        
        if not found_pair:
            raise ValueError(f"Ran out of unique pairs for n_both_object_target at variant {i+1}")
        
        # Create BDDL with swapped regions
        pickup_region = init_placements[pickup_target]
        place_region = init_placements[place_target]
        swap1_region = init_placements[swap_obj1]
        swap2_region = init_placements[swap_obj2]
        
        object_swaps = {
            pickup_target: swap1_region,
            place_target: swap2_region,
            swap_obj1: pickup_region,
            swap_obj2: place_region
        }
        
        variant_bddl_path = bddl_dir / f"{base_name}_{variant_idx}.bddl"
        create_variant_bddl(bddl_path, variant_bddl_path, object_swaps)
        
        # Regenerate init states (safer than swapping, handles different object sizes)
        print(f"   Generating {num_init_states} init states for variant {variant_idx}...")
        variant_init_states = regenerate_init_states(
            str(variant_bddl_path), 
            num_states=num_init_states,
            deterministic=deterministic,
            base_seed=100 + variant_idx  # Different seed per variant
        )
        
        variant_init_path = init_dir / f"{base_name}_{variant_idx}.init"
        torch.save(variant_init_states, variant_init_path)
        
        variant_pruned_path = init_dir / f"{base_name}_{variant_idx}.pruned_init"
        torch.save(variant_init_states, variant_pruned_path)
        
        print(f"   Variant {variant_idx}: {pickup_target} ↔ {swap_obj1}, {place_target} ↔ {swap_obj2}")
        
        all_variants.append({
            'idx': variant_idx,
            'type': 'both_targets',
            'bddl': variant_bddl_path,
            'init': variant_init_path,
            'swap': f"{pickup_target} ↔ {swap_obj1}, {place_target} ↔ {swap_obj2}"
        })
        
        variant_idx += 1
    
    env.close()
    
    # Visualize all variants (including variant_0 if created)
    total_to_visualize = len(all_variants) + (1 if create_variant_0 else 0)
    print(f"\n8. Visualizing {total_to_visualize} variants...")
    
    # Visualize variant_0 first if it was created
    if create_variant_0:
        variant_0_bddl = bddl_dir / f"{base_name}_0.bddl"
        variant_0_init = init_dir / f"{base_name}_0.init"
        visualize_variant(
            str(variant_0_bddl),
            str(variant_0_init),
            str(output_dir),
            0
        )
        print(f"   ✓ Variant 0: original scene")
    
    # Visualize generated variants
    for variant in all_variants:
        visualize_variant(
            str(variant['bddl']),
            str(variant['init']),
            str(output_dir),
            variant['idx']
        )
        print(f"   ✓ Variant {variant['idx']}: {variant['swap']}")
    
    print(f"\n{'='*80}")
    print("✅ SUCCESS!")
    print(f"{'='*80}")
    print(f"Generated {len(all_variants)} variants:")
    print(f"  • {n_target_object} variants: swapping pickup target")
    print(f"  • {n_target_location} variants: swapping place target")
    print(f"  • {n_both_object_target} variants: swapping both targets")
    print(f"  • Deterministic: {'Yes' if deterministic else 'No (random each run)'}")
    print(f"\nFiles saved to:")
    print(f"  BDDL: {bddl_dir}")
    print(f"  Init: {init_dir}")
    print(f"  Visualizations: {output_dir}")
    print(f"{'='*80}\n")
    
    return all_variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scene variants by swapping object positions"
    )
    parser.add_argument("--bddl", type=str, required=True,
                       help="Path to original BDDL file")
    parser.add_argument("--init", type=str, required=True,
                       help="Path to original init states file")
    parser.add_argument("--n_target_object", type=int, required=True,
                       help="Number of variants swapping pickup target")
    parser.add_argument("--n_target_location", type=int, required=True,
                       help="Number of variants swapping place target")
    parser.add_argument("--n_both_object_target", type=int, required=True,
                       help="Number of variants swapping both targets")
    parser.add_argument("--num_init_states", type=int, default=10,
                       help="Number of init states per variant (default: 10, use 50 for full dataset)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory for visualizations (optional)")
    parser.add_argument("--create_variant_0", action="store_true", default=True,
                       help="Create variant_0 as copy of original (default: True)")
    parser.add_argument("--no_variant_0", action="store_false", dest="create_variant_0",
                       help="Don't create variant_0")
    parser.add_argument("--deterministic", action="store_true", default=False,
                       help="Use fixed random seeds for reproducible init states (default: False)")
    
    args = parser.parse_args()
    
    generate_scene_variants(
        args.bddl,
        args.init,
        args.n_target_object,
        args.n_target_location,
        args.n_both_object_target,
        args.output_dir,
        args.num_init_states,
        args.create_variant_0,
        args.deterministic
    )
