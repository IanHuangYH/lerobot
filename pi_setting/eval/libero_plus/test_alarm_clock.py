#!/usr/bin/env python3
"""Test script to verify alarm_clock object can be loaded in LIBERO environment."""

import sys
sys.path.insert(0, '/workspace/lerobot/third_party/LIBERO')

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.objects import OBJECTS_DICT

print("=" * 60)
print("STEP 1: Check if alarm_clock is registered")
print("=" * 60)
if 'alarm_clock' in OBJECTS_DICT:
    print(f"✓ alarm_clock registered: {OBJECTS_DICT['alarm_clock']}")
else:
    print("✗ alarm_clock NOT registered!")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 2: Try to instantiate AlarmClock object directly")
print("=" * 60)
print("(Skipping direct instantiation - will test via environment)")
print("Note: New LIBERO-plus objects have different XML structure")
print("      and may only work when loaded through BDDL environment")

print("\n" + "=" * 60)
print("STEP 3: Load BDDL file and create environment")
print("=" * 60)
bddl_file = "/workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/test_alarm_clock.bddl"
print(f"Loading: {bddl_file}")

try:
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)
    print("✓ Environment created successfully!")
    
    print("\n" + "=" * 60)
    print("STEP 4: Reset environment and check objects")
    print("=" * 60)
    obs = env.reset()
    print("✓ Environment reset successfully!")
    
    # Check if objects are in the environment
    print(f"\nObjects in environment:")
    if hasattr(env.env, 'objects_dict'):
        for obj_name, obj in env.env.objects_dict.items():
            print(f"  - {obj_name}: {type(obj).__name__}")
    
    print("\n" + "=" * 60)
    print("STEP 5: Check rendered images in observation")
    print("=" * 60)
    
    # Check available image observations
    image_keys = [k for k in obs.keys() if 'image' in k.lower()]
    print(f"Available image observations: {image_keys}")
    
    if 'agentview_image' in obs:
        print(f"✓ agentview_image shape: {obs['agentview_image'].shape}")
    if 'robot0_eye_in_hand_image' in obs:
        print(f"✓ robot0_eye_in_hand_image shape: {obs['robot0_eye_in_hand_image'].shape}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("The alarm_clock object is successfully registered and can be used in BDDL files.")
    
except Exception as e:
    print(f"\n✗ Failed to create/run environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
