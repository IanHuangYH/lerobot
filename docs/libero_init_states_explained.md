# LIBERO Initial States - How They Work

## Where Initial States Are Loaded From

Initial states are loaded in this flow:

```python
# 1. Path resolution
get_libero_path("init_states")
# Returns: /workspace/lerobot/third_party/LIBERO/libero/libero/init_files

# 2. Full path construction
init_states_path = (
    Path(get_libero_path("init_states"))
    / task_suite.tasks[i].problem_folder    # e.g., "libero_spatial"
    / task_suite.tasks[i].init_states_file  # e.g., "task_name.pruned_init"
)

# 3. Loading the file
init_states = torch.load(init_states_path)
```

**Example for task 0 in libero_spatial:**
```
/workspace/lerobot/third_party/LIBERO/libero/libero/init_files/
    libero_spatial/
        pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.init
        pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init  ← Used
        ... (other tasks)
```

## Where Initial State Files Are Saved

Files are stored in the LIBERO package under:
```
/workspace/lerobot/third_party/LIBERO/libero/libero/init_files/
    ├── libero_spatial/
    ├── libero_object/
    ├── libero_goal/
    ├── libero_10/
    └── libero_90/
```

Each folder contains `.init` and `.pruned_init` files (100 states each, shape `[100, N]` where N is the state dimension).

## What's Inside Initial State Files

Each file contains **100 MuJoCo simulation states** as a numpy array:

```python
init_states.shape  # (100, 92) for original spatial tasks
```

Each state (92-dimensional vector) contains:
- Robot joint positions (qpos)
- Robot joint velocities (qvel)
- Object positions and orientations
- Object velocities
- Gripper state
- All physics simulation parameters

**When you add `yellow_book`, the state becomes (100, 105)** because the book adds:
- 7 qpos values (3D position, 4D quaternion)
- 6 qvel values (3D linear velocity, 3D angular velocity)

## Why Use Pre-saved Initial States?

### Advantages ✅

1. **Reproducibility**: Same initial conditions across runs
   - Essential for fair benchmarking
   - Compare different policies on identical scenarios

2. **Consistency**: Multiple evaluation episodes start from known states
   - Can report "success on state 0-9" vs "state 10-19"
   - Isolate difficult vs easy initial conditions

3. **Speed**: No randomization overhead
   - Don't wait for physics to settle objects
   - Deterministic scene generation

4. **Research validity**: Published results can be reproduced
   - Papers can reference specific init state indices
   - Other researchers get same evaluation conditions

### Example Use Case
```python
# Evaluate on first 10 deterministic states
for i in range(10):
    env.reset()
    obs = env.set_init_state(init_states[i])  # Always same starting position
    # ... run policy
```

## Is There a Downside to Always Generating from BDDL?

### Using BDDL-only approach (`init_states=false`)

**Pros:**
- ✅ **Maximum flexibility**: Modify BDDL freely without regenerating states
- ✅ **Simple workflow**: No state management needed
- ✅ **Realistic variability**: Tests generalization to varied initial conditions
- ✅ **No dimension mismatch**: Always compatible with current BDDL

**Cons:**
- ❌ **Non-reproducible**: Different positions each reset
- ❌ **Non-deterministic results**: Success rate varies between runs
- ❌ **Harder to debug**: Can't replay exact same scenario
- ❌ **Incomparable**: Can't compare with published benchmarks
- ❌ **Unstable objects**: Need settling steps after reset (objects may float/collide initially)

### Comparison Table

| Aspect | Pre-saved States | BDDL-only Generation |
|--------|-----------------|---------------------|
| **Reproducibility** | Perfect | None |
| **Flexibility** | Must regenerate on BDDL changes | Immediate |
| **Benchmark validity** | Yes | No |
| **Setup complexity** | Medium | Low |
| **Generalization test** | Limited variety | High variety |
| **Debugging** | Easy (replay same state) | Hard (random each time) |

## Recommended Workflow

### For Development & Experimentation
Use `init_states=false`:
```bash
lerobot-eval --env.init_states=false ...
```
- Fast iteration on BDDL changes
- Test robustness to initialization variance

### For Benchmarking & Publication
Use pre-saved states:
```bash
# 1. Modify BDDL
vim /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/my_task.bddl

# 2. Regenerate init states
python scripts/regenerate_modified_task_states.py --suite libero_spatial --task_id 0 --num_states 100

# 3. Evaluate with deterministic states
lerobot-eval --env.init_states=true ...
```

### Hybrid Approach
1. **Development phase**: Use `init_states=false` for quick testing
2. **Final validation**: Generate 100 states, use `init_states=true` for official metrics
3. **Generalization test**: Run additional eval with `init_states=false` to test robustness

## How Init States Are Generated

The original LIBERO states were created from demonstrations:

```python
# Pseudocode from LIBERO
for demo in demonstrations:
    env.reset()
    # Play back demonstration actions
    for action in demo:
        env.step(action)
    # Sample states from the demonstration rollout
    states.append(env.sim.get_state())
```

This ensures states are:
- **Physically valid**: Objects settled naturally
- **Task-relevant**: Representative of actual task scenarios
- **Diverse**: Captured from different demonstration episodes

## Summary

**Where loaded from:**
```
/workspace/lerobot/third_party/LIBERO/libero/libero/init_files/{suite_name}/{task_name}.pruned_init
```

**Always generating from BDDL (`init_states=false`):**
- **Best for**: Development, BDDL experimentation, testing generalization
- **Drawback**: Non-reproducible results, can't compare to benchmarks

**Using pre-saved states (`init_states=true`):**
- **Best for**: Benchmarking, reproducible results, paper comparisons
- **Drawback**: Must regenerate when BDDL changes

**Recommendation:**
Use `init_states=false` during development, then generate proper states for final evaluation.
