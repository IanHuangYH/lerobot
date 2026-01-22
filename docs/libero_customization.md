# LIBERO Customization Guide for LeRobot

This guide explains how to add custom objects and modify LIBERO tasks in your workspace.

## Quick Start

### 1. Verify Workspace Setup

```bash
python /workspace/lerobot/test_libero_workspace.py
```

You should see "✅ Using WORKSPACE version".

### 2. Locate BDDL Task Files

All task definitions are in BDDL (Behavior Domain Definition Language) files:

```
/workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/
├── libero_spatial/   # 10 spatial reasoning tasks
├── libero_object/    # 10 object manipulation tasks  
├── libero_goal/      # 10 goal-conditioned tasks
├── libero_10/        # 10-task benchmark
└── libero_90/        # 90-task benchmark
```

## Adding Objects from Other Suites to Spatial Tasks

### Example: Add `milk` and `alphabet_soup` to a Spatial Task

1. **Find the source object definitions** in `libero_object/`:
   ```bash
   grep -A 5 ":objects" /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl
   ```

2. **Choose a spatial task to modify**:
   ```bash
   cd /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/
   cp pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl \
      pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_custom.bddl
   ```

3. **Edit the BDDL file** to add objects:

   **Before** (original):
   ```lisp
   (:objects
     akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
     cookies_1 - cookies
     glazed_rim_porcelain_ramekin_1 - glazed_rim_porcelain_ramekin
     plate_1 - plate
   )
   ```

   **After** (with new objects):
   ```lisp
   (:objects
     akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
     cookies_1 - cookies
     glazed_rim_porcelain_ramekin_1 - glazed_rim_porcelain_ramekin
     plate_1 - plate
     milk_1 - milk
     alphabet_soup_1 - alphabet_soup
   )
   ```

4. **Add initial positions** in `:init` section:
   ```lisp
   (:init
     (On akita_black_bowl_1 main_table_table_center)
     (On akita_black_bowl_2 main_table_next_to_plate_region)
     (On plate_1 main_table_plate_region)
     (On cookies_1 main_table_box_region)
     (On glazed_rim_porcelain_ramekin_1 main_table_ramekin_region)
     (On milk_1 main_table_next_to_box_region)
     (On alphabet_soup_1 main_table_next_to_ramekin_region)
     ...
   )
   ```

### Available Object Types

Check object types available in each suite:

```bash
# List all object types in object suite
grep "^  " /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_object/*.bddl | \
  grep " - " | cut -d "-" -f 2 | sort -u
```

Common objects:
- **libero_spatial**: `akita_black_bowl`, `plate`, `cookies`, `glazed_rim_porcelain_ramekin`, `wooden_cabinet`, `flat_stove`
- **libero_object**: `milk`, `alphabet_soup`, `butter`, `cream_cheese`, `salad_dressing`, `tomato_sauce`, `basket`
- **libero_goal**: Various task-specific objects

### Available Regions

Regions define where objects can be placed on the table:

```lisp
(:regions
  (table_center ...)       # Center of table
  (table_front ...)        # Front of table
  (plate_region ...)       # On/near the plate
  (next_to_plate_region ...) # Next to plate
  (box_region ...)         # Where the cookie box is
  (cabinet_region ...)     # Cabinet location
  (stove_region ...)       # Stove location
)
```

## Creating a New Custom Task

1. **Copy an existing task**:
   ```bash
   cd /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/
   cp pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl \
      my_custom_task.bddl
   ```

2. **Modify** the `:language`, `:objects`, `:init`, and `:goal` sections

3. **Test your task**:
   ```python
   from libero.libero import benchmark
   suite = benchmark.get_benchmark_dict()['libero_spatial']()
   # Your custom task should appear in the suite
   ```

## Testing Changes

After modifying BDDL files, test with:

```bash
cd /workspace/lerobot
python -c "
from libero.libero import benchmark, get_libero_path
print('BDDL path:', get_libero_path('bddl_files'))
suite = benchmark.get_benchmark_dict()['libero_spatial']()
print(f'Tasks in suite: {len(suite.tasks)}')
for i, task in enumerate(suite.tasks[:3]):
    print(f'{i}: {task.language}')
"
```

Or run your evaluation script:
```bash
bash /workspace/lerobot/pi_setting/eval/eval_libero_quick_test.sh
```

## Tips

1. **Keep backups**: Copy original BDDL files before modifying
2. **Consistent naming**: Use descriptive names for custom tasks
3. **Valid regions**: Only use regions defined in `:regions` section
4. **Object compatibility**: Ensure objects are compatible with the scene (e.g., don't add floor objects to tabletop scenes)
5. **Version control**: Commit your BDDL changes to git

## Troubleshooting

**Q: Changes not taking effect?**
- Verify you're editing files in `/workspace/lerobot/third_party/LIBERO/`
- Check `~/.libero/config.yaml` points to workspace
- Restart Python to reload BDDL files

**Q: Object not appearing in scene?**
- Check object type exists in robosuite's object library
- Verify initial position region is defined
- Check for typos in object names

**Q: Getting path warnings?**
- Run `/workspace/lerobot/scripts/setup_libero_workspace.sh`
- Check PYTHONPATH includes `/workspace/lerobot/third_party/LIBERO`

## Reference

- BDDL Language Spec: https://github.com/StanfordVL/bddl
- LIBERO Documentation: https://github.com/ARISE-Initiative/LIBERO
- Robosuite Objects: https://github.com/ARISE-Initiative/robosuite
