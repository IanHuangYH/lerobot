#!/usr/bin/env python3
"""
Evaluate a policy on LIBERO scene variants.

This script evaluates a policy on multiple scene variants of the same task,
where each variant has a different object layout (e.g., basket in different positions).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_variant_files(bddl_dir: Path, init_dir: Path, task_name: str):
    """
    Find all variant BDDL and init files for a task.
    
    Returns:
        List of tuples: [(variant_idx, bddl_path, init_path, pruned_init_path), ...]
    """
    variants = []
    
    # Find all variant BDDL files matching pattern: {task_name}_{number}.bddl
    # Exclude the base file (without number suffix)
    all_bddls = sorted(bddl_dir.glob(f"{task_name}_*.bddl"))
    
    for bddl_path in all_bddls:
        # Extract variant index from filename
        stem = bddl_path.stem  # e.g., "pick_up_the_alphabet_soup_and_place_it_in_the_basket_1"
        variant_suffix = stem.replace(f"{task_name}_", "")
        
        # Skip if it's the base file (no number suffix or has non-numeric suffix)
        if not variant_suffix or not variant_suffix.isdigit():
            continue
        
        try:
            variant_idx = int(variant_suffix)
        except ValueError:
            print(f"Warning: Could not parse variant index from {bddl_path.name}, skipping")
            continue
        
        # Check for corresponding init files
        init_path = init_dir / f"{task_name}_{variant_suffix}.init"
        pruned_init_path = init_dir / f"{task_name}_{variant_suffix}.pruned_init"
        
        if not init_path.exists():
            print(f"Warning: Init file not found for variant {variant_idx}: {init_path}")
            continue
        
        variants.append((variant_idx, bddl_path, init_path, pruned_init_path))
    
    # Sort by variant index
    variants.sort(key=lambda x: x[0])
    
    return variants


def run_evaluation_for_variant(
    variant_idx: int,
    bddl_path: Path,
    init_path: Path,
    pruned_init_path: Path,
    base_bddl_path: Path,
    base_init_path: Path,
    base_pruned_init_path: Path,
    variants_dir: Path,
    eval_config: dict,
    episode_number: int,
):
    """
    Run evaluation for a single variant by temporarily swapping files.
    
    Strategy:
    1. Backup original BDDL/init files
    2. Copy variant files to base location
    3. Run lerobot-eval with n_episodes=1
    4. Rename episode_00000 to episode_{episode_number} and move to final location
    5. Restore original files
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING VARIANT {variant_idx} (Episode {episode_number:05d})")
    print(f"{'='*80}")
    print(f"BDDL: {bddl_path.name}")
    print(f"Init: {init_path.name}")
    
    # Temporary output directory for this single evaluation
    temp_output_dir = variants_dir.parent / f".temp_eval_{variant_idx}"
    
    # Create backup directory
    backup_dir = base_bddl_path.parent / ".variant_eval_backup"
    backup_dir.mkdir(exist_ok=True)
    
    # Backup original files
    backup_bddl = backup_dir / base_bddl_path.name
    backup_init = backup_dir / base_init_path.name
    backup_pruned = backup_dir / base_pruned_init_path.name
    
    try:
        # Step 1: Backup original files if they exist
        if base_bddl_path.exists():
            shutil.copy2(base_bddl_path, backup_bddl)
        if base_init_path.exists():
            shutil.copy2(base_init_path, backup_init)
        if base_pruned_init_path.exists():
            shutil.copy2(base_pruned_init_path, backup_pruned)
        
        # Step 2: Copy variant files to base location
        shutil.copy2(bddl_path, base_bddl_path)
        shutil.copy2(init_path, base_init_path)
        if pruned_init_path.exists():
            shutil.copy2(pruned_init_path, base_pruned_init_path)
        
        # Step 3: Run lerobot-eval
        cmd = [
            "lerobot-eval",
            f"--env.type=libero",
            f"--env.task={eval_config['task_suite']}",
            f"--eval.batch_size=1",
            f"--eval.n_episodes=1",
            f"--policy.path={eval_config['policy_path']}",
            f"--policy.n_action_steps={eval_config['n_action_steps']}",
            f"--policy.device=cuda:{eval_config['policy_gpu_id']}",
            f"--policy.compile_model={eval_config['compile_model']}",
            f"--output_dir={temp_output_dir}",
            f"--env.max_parallel_tasks=1",
            f"--env.task_ids={eval_config['task_ids']}",
            f"--env.init_states={eval_config['use_init_states']}",
            f"--eval.save_attention_maps={eval_config['save_attention_maps']}",
        ]
        
        # Set CUDA_VISIBLE_DEVICES
        env = {**subprocess.os.environ, 'CUDA_VISIBLE_DEVICES': eval_config['cuda_devices']}
        
        print(f"\nRunning command:")
        print(" ".join(cmd))
        
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"Warning: Evaluation failed for variant {variant_idx}")
            return False
        
        # Step 4: Copy and rename results to final location
        print(f"\n   Checking temp output directory: {temp_output_dir}")
        print(f"   Temp dir exists: {temp_output_dir.exists()}")
        
        if temp_output_dir.exists():
            print(f"   Contents of temp dir:")
            for item in temp_output_dir.rglob("*"):
                if item.is_file():
                    print(f"      {item.relative_to(temp_output_dir)}")
        
        # The temp directory has structure: attention/libero_object_X/episode_00000_*.pt
        #                                    videos/libero_object_X/eval_episode_00000.mp4
        # We need to rename episode_00000 → episode_{episode_number:05d}
        
        task_key = f"libero_{eval_config['task_suite'].replace('libero_', '')}_{eval_config['task_ids'].strip('[]')}"
        print(f"   Task key: {task_key}")
        
        # Copy attention files
        temp_attention_dir = temp_output_dir / "attention" / task_key
        final_attention_dir = variants_dir / "attention" / task_key
        
        if temp_attention_dir.exists():
            final_attention_dir.mkdir(parents=True, exist_ok=True)
            attention_files = list(temp_attention_dir.glob("episode_*.pt"))
            print(f"   Found {len(attention_files)} attention files")
            for file in attention_files:
                # Extract episode number from original filename
                old_episode_num = file.stem.split('_')[1]
                new_name = file.name.replace(f"episode_{old_episode_num}", f"episode_{episode_number:05d}")
                shutil.copy2(file, final_attention_dir / new_name)
                print(f"   Copied attention: {new_name}")
        else:
            print(f"   Attention dir not found: {temp_attention_dir}")
        
        # Copy video files
        temp_videos_dir = temp_output_dir / "videos" / task_key
        final_videos_dir = variants_dir / "videos" / task_key
        
        if temp_videos_dir.exists():
            final_videos_dir.mkdir(parents=True, exist_ok=True)
            video_files = list(temp_videos_dir.glob("*.mp4"))
            print(f"   Found {len(video_files)} video files")
            for file in video_files:
                # Extract episode number from original filename
                # Could be "eval_episode_00000.mp4" or "episode_00000.mp4"
                if "eval_episode_" in file.name:
                    old_episode_num = file.name.split('_')[2].split('.')[0]
                    new_name = file.name.replace(f"episode_{old_episode_num}", f"episode_{episode_number:05d}")
                else:
                    old_episode_num = file.name.split('_')[1].split('.')[0]
                    new_name = file.name.replace(f"episode_{old_episode_num}", f"episode_{episode_number:05d}")
                shutil.copy2(file, final_videos_dir / new_name)
                print(f"   Copied video: {new_name}")
        else:
            print(f"   Videos dir not found: {temp_videos_dir}")
        
        # Copy eval_info.json if exists (merge if needed)
        temp_eval_info = temp_output_dir / "eval_info.json"
        final_eval_info = variants_dir / "eval_info.json"
        
        if temp_eval_info.exists():
            with open(temp_eval_info, 'r') as f:
                temp_info = json.load(f)
            
            # Update episode index and video paths in the temp info
            if 'per_task' in temp_info:
                for task in temp_info['per_task']:
                    if 'metrics' in task:
                        # Update video paths to point to final location
                        if 'video_paths' in task['metrics']:
                            updated_paths = []
                            for path in task['metrics']['video_paths']:
                                # Convert from temp path to final path
                                path_obj = Path(path)
                                if path_obj.exists():
                                    new_filename = f"eval_episode_{episode_number}.mp4"
                                    new_path = f"{variants_dir.name}/videos/{task_key}/{new_filename}"
                                    updated_paths.append(new_path)
                            task['metrics']['video_paths'] = updated_paths
            
            # Update overall video paths
            if 'overall' in temp_info and 'video_paths' in temp_info['overall']:
                updated_paths = []
                for path in temp_info['overall']['video_paths']:
                    path_obj = Path(path)
                    if path_obj.exists():
                        new_filename = f"eval_episode_{episode_number}.mp4"
                        new_path = f"{variants_dir.name}/videos/{task_key}/{new_filename}"
                        updated_paths.append(new_path)
                temp_info['overall']['video_paths'] = updated_paths
            
            # Update per_group video paths
            if 'per_group' in temp_info:
                for group_name, group_data in temp_info['per_group'].items():
                    if 'video_paths' in group_data:
                        updated_paths = []
                        for path in group_data['video_paths']:
                            path_obj = Path(path)
                            if path_obj.exists():
                                new_filename = f"eval_episode_{episode_number}.mp4"
                                new_path = f"{variants_dir.name}/videos/{task_key}/{new_filename}"
                                updated_paths.append(new_path)
                        group_data['video_paths'] = updated_paths
            
            # Merge with existing eval_info if it exists
            if final_eval_info.exists():
                with open(final_eval_info, 'r') as f:
                    final_info = json.load(f)
                
                # Merge per_task metrics
                if 'per_task' in temp_info and 'per_task' in final_info:
                    for i, temp_task in enumerate(temp_info['per_task']):
                        if i < len(final_info['per_task']):
                            final_task = final_info['per_task'][i]
                            # Merge metrics arrays
                            if 'metrics' in temp_task and 'metrics' in final_task:
                                for key in ['sum_rewards', 'max_rewards', 'successes', 'video_paths']:
                                    if key in temp_task['metrics']:
                                        if key in final_task['metrics']:
                                            final_task['metrics'][key].extend(temp_task['metrics'][key])
                                        else:
                                            final_task['metrics'][key] = temp_task['metrics'][key]
                
                # Merge per_group
                if 'per_group' in temp_info and 'per_group' in final_info:
                    for group_name, temp_group in temp_info['per_group'].items():
                        if group_name in final_info['per_group']:
                            final_group = final_info['per_group'][group_name]
                            # Accumulate arrays
                            for key in ['video_paths']:
                                if key in temp_group:
                                    if key in final_group:
                                        final_group[key].extend(temp_group[key])
                                    else:
                                        final_group[key] = temp_group[key]
                            # Recalculate aggregated metrics
                            n_episodes = final_group.get('n_episodes', 0) + temp_group.get('n_episodes', 1)
                            final_group['n_episodes'] = n_episodes
                
                # Merge overall
                if 'overall' in temp_info and 'overall' in final_info:
                    for key in ['video_paths']:
                        if key in temp_info['overall']:
                            if key in final_info['overall']:
                                final_info['overall'][key].extend(temp_info['overall'][key])
                            else:
                                final_info['overall'][key] = temp_info['overall'][key]
                    
                    # Update n_episodes
                    final_info['overall']['n_episodes'] = final_info['overall'].get('n_episodes', 0) + 1
                
                with open(final_eval_info, 'w') as f:
                    json.dump(final_info, f, indent=2)
            else:
                # First variant - just save the temp info
                with open(final_eval_info, 'w') as f:
                    json.dump(temp_info, f, indent=2)
        
        # Clean up temp directory
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
        
        print(f"\nResults saved to: {variants_dir}")
        
        return True
        
    finally:
        # Step 5: Restore original files
        if backup_bddl.exists():
            shutil.copy2(backup_bddl, base_bddl_path)
            backup_bddl.unlink()
        if backup_init.exists():
            shutil.copy2(backup_init, base_init_path)
            backup_init.unlink()
        if backup_pruned.exists():
            shutil.copy2(backup_pruned, base_pruned_init_path)
            backup_pruned.unlink()
        
        # Clean up backup directory if empty
        if backup_dir.exists() and not any(backup_dir.iterdir()):
            backup_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy on LIBERO scene variants")
    parser.add_argument("--task_suite", type=str, required=True, help="Task suite (e.g., libero_object)")
    parser.add_argument("--task_name", type=str, required=True, help="Task name (without variant suffix)")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID in LIBERO suite")
    parser.add_argument("--max_episodes", type=int, required=True, help="Maximum number of variants to evaluate")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to policy model")
    parser.add_argument("--policy_gpu_id", type=int, default=0, help="GPU ID for policy")
    parser.add_argument("--cuda_devices", type=str, default="0,1", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--n_action_steps", type=int, default=10, help="Number of action steps")
    parser.add_argument("--compile_model", type=str, default="false", help="Compile model")
    parser.add_argument("--use_init_states", type=str, default="true", help="Use init states")
    parser.add_argument("--save_attention_maps", type=str, default="true", help="Save attention maps")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--bddl_dir", type=str, required=True, help="Directory containing BDDL files")
    parser.add_argument("--init_dir", type=str, required=True, help="Directory containing init files")
    
    args = parser.parse_args()
    
    bddl_dir = Path(args.bddl_dir)
    init_dir = Path(args.init_dir)
    base_output_dir = Path(args.output_dir)
    
    # Create variants directory: {base_output_dir}/{task_suite}_{task_id}_variants/
    variants_dir = base_output_dir / f"{args.task_suite.replace('libero_', '')}_{args.task_id}_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all variant files
    print(f"\n{'='*80}")
    print(f"FINDING SCENE VARIANTS")
    print(f"{'='*80}")
    print(f"Task: {args.task_name}")
    print(f"BDDL dir: {bddl_dir}")
    print(f"Init dir: {init_dir}")
    
    variants = find_variant_files(bddl_dir, init_dir, args.task_name)
    
    if not variants:
        print(f"\nError: No variant files found for task '{args.task_name}'")
        print(f"Expected files like: {args.task_name}_1.bddl, {args.task_name}_2.bddl, etc.")
        sys.exit(1)
    
    print(f"\nFound {len(variants)} variants:")
    for variant_idx, bddl_path, init_path, _ in variants:
        print(f"  Variant {variant_idx:03d}: {bddl_path.name}")
    
    # Limit to max_episodes
    n_episodes = min(args.max_episodes, len(variants))
    print(f"\nEvaluating {n_episodes} variants (max_episodes={args.max_episodes})")
    
    # Base file paths (where lerobot-eval expects files)
    base_bddl_path = bddl_dir / f"{args.task_name}.bddl"
    base_init_path = init_dir / f"{args.task_name}.init"
    base_pruned_init_path = init_dir / f"{args.task_name}.pruned_init"
    
    # Evaluation config
    eval_config = {
        'task_suite': args.task_suite,
        'task_ids': f'[{args.task_id}]',
        'policy_path': args.policy_path,
        'policy_gpu_id': args.policy_gpu_id,
        'cuda_devices': args.cuda_devices,
        'n_action_steps': args.n_action_steps,
        'compile_model': args.compile_model,
        'use_init_states': args.use_init_states,
        'save_attention_maps': args.save_attention_maps,
    }
    
    # Run evaluation for each variant
    successful = 0
    failed = 0
    
    for i in range(n_episodes):
        variant_idx, bddl_path, init_path, pruned_init_path = variants[i]
        
        success = run_evaluation_for_variant(
            variant_idx=variant_idx,
            bddl_path=bddl_path,
            init_path=init_path,
            pruned_init_path=pruned_init_path,
            base_bddl_path=base_bddl_path,
            base_init_path=base_init_path,
            base_pruned_init_path=base_pruned_init_path,
            variants_dir=variants_dir,
            eval_config=eval_config,
            episode_number=i,  # Episode number matches variant index in evaluation
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total variants: {len(variants)}")
    print(f"Evaluated: {n_episodes}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved in: {variants_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
