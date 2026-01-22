#!/usr/bin/env python3
"""
Regenerate initial states for a modified LIBERO task.

When you add/remove objects from a BDDL file, the MuJoCo state dimensions change,
so you need to regenerate compatible initial states.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from libero.libero import benchmark, get_libero_path


def regenerate_init_states(suite_name: str, task_id: int, num_states: int = 10):
    """Generate new initial states for a modified task."""
    
    # Get the task suite
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.tasks[task_id]
    
    print(f"Task: {task.language}")
    print(f"Problem folder: {task.problem_folder}")
    
    # Create environment
    env = suite.get_env(task.problem_name, task.task_id, task.bddl_file)
    
    # Generate multiple initial states by resetting
    init_states = []
    for i in range(num_states):
        env.reset()
        # Get the current MuJoCo state
        state = env.sim.get_state().flatten()
        init_states.append(state)
        print(f"Generated state {i+1}/{num_states} - shape: {state.shape}")
    
    # Convert to numpy array
    init_states = np.array(init_states)
    
    # Save to the expected location
    init_states_dir = Path(get_libero_path("init_states")) / task.problem_folder
    init_states_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = init_states_dir / task.init_states_file
    
    # Backup old file if it exists
    if output_path.exists():
        backup_path = output_path.with_suffix(".bak")
        print(f"Backing up existing file to: {backup_path}")
        output_path.rename(backup_path)
    
    # Save new states
    torch.save(init_states, output_path)
    print(f"\n✅ Saved {num_states} initial states to: {output_path}")
    print(f"   State shape: {init_states.shape}")
    
    env.close()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate initial states for modified LIBERO task")
    parser.add_argument("--suite", type=str, default="libero_spatial", 
                        help="Task suite name")
    parser.add_argument("--task_id", type=int, default=0,
                        help="Task ID in the suite")
    parser.add_argument("--num_states", type=int, default=10,
                        help="Number of initial states to generate")
    
    args = parser.parse_args()
    
    regenerate_init_states(args.suite, args.task_id, args.num_states)
