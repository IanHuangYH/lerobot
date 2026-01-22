#!/usr/bin/env python3
"""Test script to verify LIBERO is using workspace version."""

import libero
from libero.libero import get_libero_path
import os

print("="*60)
print("LIBERO Workspace Configuration Test")
print("="*60)

# Check module location
module_path = libero.__file__
print(f"\n✓ Module path: {module_path}")

if "/workspace/lerobot/third_party/LIBERO" in str(module_path):
    print("  ✅ Using WORKSPACE version")
else:
    print("  ❌ Using CONDA version")

# Check BDDL files location
bddl_path = get_libero_path('bddl_files')
print(f"\n✓ BDDL files: {bddl_path}")
print(f"  Exists: {os.path.exists(bddl_path)}")

# List available task suites
if os.path.exists(bddl_path):
    suites = [d for d in os.listdir(bddl_path) if os.path.isdir(os.path.join(bddl_path, d))]
    print(f"\n✓ Available task suites: {suites}")
    
    # Count tasks in each suite
    for suite in suites:
        suite_path = os.path.join(bddl_path, suite)
        tasks = [f for f in os.listdir(suite_path) if f.endswith('.bddl')]
        print(f"  - {suite}: {len(tasks)} tasks")

print("\n" + "="*60)
print("✅ Test completed successfully!")
print("="*60)
