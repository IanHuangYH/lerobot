# LIBERO Workspace Setup - Summary

## ✅ What Was Done

### 1. Installed LIBERO as Workspace Package

- Removed conda version of LIBERO
- Added `/workspace/lerobot/third_party/LIBERO` to `PYTHONPATH` in `~/.bashrc`
- Installed `gym==0.26.2` (required by LIBERO, not gymnasium)
- LIBERO is now loaded directly from your workspace

### 2. Updated Configuration

Updated `~/.libero/config.yaml` to point to workspace paths:
- BDDL files: `/workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files`
- Assets: `/workspace/lerobot/third_party/LIBERO/libero/libero/assets`
- Datasets: `/workspace/lerobot/third_party/LIBERO/datasets`
- Init states: `/workspace/lerobot/third_party/LIBERO/libero/libero/init_files`

### 3. Added Conda Auto-Activation

Added `conda activate lerobot` to `~/.bashrc` so the environment activates automatically.

## 📁 Files Created

1. **[/workspace/lerobot/test_libero_workspace.py](file:///workspace/lerobot/test_libero_workspace.py)**
   - Verification script to test LIBERO workspace configuration

2. **[/workspace/lerobot/third_party/README.md](file:///workspace/lerobot/third_party/README.md)**
   - Documentation for third-party packages

3. **[/workspace/lerobot/scripts/setup_libero_workspace.sh](file:///workspace/lerobot/scripts/setup_libero_workspace.sh)**
   - Automated setup script for fresh environments

4. **[/workspace/lerobot/docs/libero_customization.md](file:///workspace/lerobot/docs/libero_customization.md)**
   - Comprehensive guide for customizing LIBERO tasks

## ✅ How to Verify Everything Works

### Test 1: Check LIBERO Paths
```bash
python /workspace/lerobot/test_libero_workspace.py
```

Expected output:
```
✓ Module path: None
  ✅ Using WORKSPACE version
✓ BDDL files: /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files
  Exists: True
✓ Available task suites: ['libero_spatial', 'libero_goal', 'libero_object', 'libero_10', 'libero_90']
```

### Test 2: Run Your Eval Script
```bash
cd /workspace/lerobot
bash pi_setting/eval/eval_libero_quick_test.sh
```

This should now load BDDL files from your workspace.

## 🎯 How to Add Objects to Spatial Tasks

### Quick Example

1. **Navigate to BDDL files**:
   ```bash
   cd /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files/libero_spatial
   ```

2. **Edit a task file** (or copy and create new one):
   ```bash
   nano pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl
   ```

3. **Add objects** from `libero_object` suite:
   - Find objects in `../libero_object/*.bddl` files
   - Add to `:objects` section
   - Add initial positions to `:init` section

4. **Example modification**:
   ```lisp
   (:objects
     akita_black_bowl_1 - akita_black_bowl
     plate_1 - plate
     milk_1 - milk              # NEW from libero_object
     alphabet_soup_1 - alphabet_soup  # NEW from libero_object
   )
   
   (:init
     (On akita_black_bowl_1 main_table_table_center)
     (On plate_1 main_table_plate_region)
     (On milk_1 main_table_next_to_box_region)      # NEW
     (On alphabet_soup_1 main_table_next_to_ramekin_region)  # NEW
   )
   ```

5. **Test**: Run your evaluation script

## 📚 Documentation

- **[Customization Guide](file:///workspace/lerobot/docs/libero_customization.md)** - Detailed guide with examples
- **[Third-party README](file:///workspace/lerobot/third_party/README.md)** - Package management info
- **[Setup Script](file:///workspace/lerobot/scripts/setup_libero_workspace.sh)** - Automated setup for new environments

## 🔄 For New Terminal Sessions

Your `~/.bashrc` now automatically:
1. Activates the `lerobot` conda environment
2. Adds LIBERO to `PYTHONPATH`

Just open a new terminal and everything works!

## 🛠️ Troubleshooting

**Issue**: Changes to BDDL files not taking effect
- **Solution**: Make sure you're editing files in `/workspace/lerobot/third_party/LIBERO/`, not in conda site-packages

**Issue**: "Module not found" errors
- **Solution**: Run `source ~/.bashrc` or open a new terminal

**Issue**: Wrong paths in warnings
- **Solution**: Check `~/.libero/config.yaml` points to workspace paths

**Issue**: Need to reset everything
- **Solution**: Run `/workspace/lerobot/scripts/setup_libero_workspace.sh`

## 🎉 Next Steps

Now you can:
1. ✅ Modify LIBERO BDDL files directly in your workspace
2. ✅ Add custom objects to any task suite
3. ✅ Create new custom tasks
4. ✅ Version control all your changes with git
5. ✅ Share your customizations with your team

Happy experimenting with LIBERO! 🤖
