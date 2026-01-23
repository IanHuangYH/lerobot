# Object Replacement Test Guide

## How the Script Handles Different Scenarios

### Scenario 1: Adding Objects (Original Use Case)
**Example**: Add `yellow_book` to existing scene
- ✅ All original objects preserved at exact positions
- ✅ New object placed according to BDDL regions
```
Old BDDL: bowl1, bowl2, plate, cookies
New BDDL: bowl1, bowl2, plate, cookies, yellow_book
Result: ➕ Added: [yellow_book], ✓ Preserved: 4 objects
```

### Scenario 2: Replacing Objects
**Example**: Replace `cookies` with `red_mug`
- ✅ Objects that exist in both envs are preserved
- ✅ Removed object's state is NOT copied (skipped safely)
- ✅ New replacement object placed by BDDL regions
```
Old BDDL: bowl1, bowl2, plate, cookies
New BDDL: bowl1, bowl2, plate, red_mug
Result: ➕ Added: [red_mug], ➖ Removed: [cookies], ✓ Preserved: 3 objects
```

### Scenario 3: Multiple Replacements
**Example**: Replace `bowl2` with `white_bowl` AND `cookies` with `red_mug`
- ✅ All common objects preserved
- ✅ Multiple removed objects detected and skipped
- ✅ Multiple new objects placed
```
Old BDDL: bowl1, bowl2, plate, cookies
New BDDL: bowl1, white_bowl, plate, red_mug
Result: ➕ Added: [red_mug, white_bowl], ➖ Removed: [bowl2, cookies], ✓ Preserved: 2 objects
```

### Scenario 4: No Changes (Your Current Case)
**Example**: BDDL files are identical
- ✅ Just copy backup directly - no script needed
- OR run script: ➕ Added: None, ➖ Removed: None, ✓ Preserved: all
```
Old BDDL: bowl1, bowl2, plate, cookies
New BDDL: bowl1, bowl2, plate, cookies
Result: Just use: cp task.bak task.pruned_init
```

## Safety Features

1. **Explicit Detection**:
   - `removed_objects = old_objects - new_objects`
   - `added_objects = new_objects - old_objects`
   - `preserved_objects = old_objects & new_objects`

2. **Safety Checks During Copy**:
   ```python
   if obj_name not in old_joint_map:
       skipped_count += 1
       continue  # Skip safely if no joint in old env
   
   if obj_name not in new_joint_map:
       print("WARNING: ...")
       skipped_count += 1
       continue  # Skip safely if no joint in new env
   ```

3. **Clear Logging**:
   - Shows exactly what was added, removed, preserved
   - Counts copied vs skipped objects
   - Warns about any unexpected situations

## Usage Example - Replace cookies with red_mug

1. **Backup original BDDL**:
   ```bash
   cp task.bddl task.bddl.old
   ```

2. **Modify task.bddl**:
   ```diff
   (:objects
     akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
   - cookies_1 - cookies
   + red_mug_1 - red_mug
     glazed_rim_porcelain_ramekin_1 - glazed_rim_porcelain_ramekin
     plate_1 - plate
   )
   
   (:init
     (On akita_black_bowl_1 main_table_between_plate_ramekin_region)
     (On akita_black_bowl_2 main_table_next_to_ramekin_region)
     (On plate_1 main_table_plate_region)
   - (On cookies_1 main_table_box_region)
   + (On red_mug_1 main_table_box_region)
     (On glazed_rim_porcelain_ramekin_1 main_table_ramekin_region)
     ...
   )
   ```

3. **Run the script**:
   ```bash
   bash set_liberoenv_addobject.sh
   ```

4. **Expected output**:
   ```
   ➕ Added objects: ['red_mug_1_main']
   ➖ Removed objects: ['cookies_1_main']
   ✓ Preserved objects: 4 items
   
   ⚠️  WARNING: 1 object(s) removed/replaced:
      - cookies_1_main
   → Their positions will NOT be copied to new environment
   
   State 0: Copied 4 objects, Skipped 0
   
   ✅ SUCCESS!
   ✓ PRESERVED positions for 4 objects (bowls, plate, ramekin, etc.)
   ➕ NEW random placements for: ['red_mug_1_main']
   ➖ REMOVED from scene: ['cookies_1_main']
   ```

## Key Points

- ✅ **obj_of_interest**: The script preserves ALL objects that exist in both environments, regardless of whether they're obj_of_interest or not
- ✅ **Safety**: Never tries to copy state for objects that don't exist in new env
- ✅ **Flexibility**: Works for additions, replacements, or both
- ✅ **Explicit**: Always tells you exactly what changed
