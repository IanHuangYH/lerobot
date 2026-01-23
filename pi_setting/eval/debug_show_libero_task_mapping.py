#!/usr/bin/env python3
"""
Show the mapping between task IDs and BDDL/init files for LIBERO suites.

Usage:
    python show_libero_task_mapping.py                    # Show all suites
    python show_libero_task_mapping.py libero_spatial     # Show specific suite
    python show_libero_task_mapping.py libero_spatial,libero_object  # Show multiple suites
"""

import sys
from libero.libero import benchmark

def show_task_mapping(suite_name):
    """Show task ID to BDDL/init mapping for a suite."""
    print(f"\n{'='*100}")
    print(f"TASK MAPPING: {suite_name.upper()}")
    print(f"{'='*100}")
    
    # Get the benchmark
    bench_dict = benchmark.get_benchmark_dict()
    if suite_name not in bench_dict:
        print(f"[ERROR] Unknown suite '{suite_name}'")
        print(f"Available suites: {', '.join(sorted(bench_dict.keys()))}")
        return
    
    suite = bench_dict[suite_name]()
    
    print(f"\nTotal tasks: {suite.get_num_tasks()}\n")
    print(f"{'ID':<4} {'Task Name':<80} {'BDDL File':<40}")
    print("-" * 100)
    
    for i in range(suite.get_num_tasks()):
        task = suite.get_task(i)
        print(f"{i:<4} {task.name:<80}")
        print(f"     BDDL: {task.problem_folder}/{task.bddl_file}")
        print(f"     Init: {task.problem_folder}/{task.init_states_file}")
        print(f"     Lang: {task.language}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User specified suite(s)
        suites_arg = sys.argv[1]
        suites = [s.strip() for s in suites_arg.split(',')]
    else:
        # Show all suites
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
    
    for suite_name in suites:
        show_task_mapping(suite_name)
    
    print("\n" + "="*100)
    print("USAGE IN EVALUATION")
    print("="*100)
    print("\nTo evaluate specific tasks, use --env.task_ids parameter:")
    print("\nExample 1: Evaluate task 0 and 1 from libero_spatial:")
    print("  --env.task=libero_spatial --env.task_ids='[0,1]'")
    print("\nExample 2: Evaluate all tasks from libero_spatial (omit task_ids):")
    print("  --env.task=libero_spatial")
    print("\nExample 3: Evaluate multiple suites (task_ids apply to each suite):")
    print("  --env.task=libero_spatial,libero_object --env.task_ids='[0,1]'")
    print("  This will run:")
    print("    - libero_spatial task 0 and 1")
    print("    - libero_object task 0 and 1")
    print()
