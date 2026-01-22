# Third-Party Dependencies

This directory contains third-party packages that we maintain locally for customization and development purposes.

## LIBERO

LIBERO (Lifelong bEnchmark for Robot Learning) is installed as an editable package from this workspace.

### Setup

The LIBERO package is configured to use workspace files instead of conda site-packages:

1. **Installation**: LIBERO is added to PYTHONPATH in `~/.bashrc`
   ```bash
   export PYTHONPATH="/workspace/lerobot/third_party/LIBERO:$PYTHONPATH"
   ```

2. **Dependencies**: LIBERO requires the old `gym` package (not `gymnasium`)
   ```bash
   pip install gym==0.26.2
   ```

3. **Configuration**: LIBERO paths are configured in `~/.libero/config.yaml` to point to workspace:
   ```yaml
   assets: /workspace/lerobot/third_party/LIBERO/libero/libero/assets
   bddl_files: /workspace/lerobot/third_party/LIBERO/libero/libero/bddl_files
   benchmark_root: /workspace/lerobot/third_party/LIBERO/libero/libero
   datasets: /workspace/lerobot/third_party/LIBERO/datasets
   init_states: /workspace/lerobot/third_party/LIBERO/libero/libero/init_files
   ```

### Customizing LIBERO Tasks

To add custom objects or modify tasks:

1. **Edit BDDL files**: Located in `LIBERO/libero/libero/bddl_files/`
   - `libero_spatial/` - Spatial reasoning tasks
   - `libero_object/` - Object manipulation tasks  
   - `libero_goal/` - Goal-conditioned tasks
   - `libero_10/` - 10-task benchmark
   - `libero_90/` - 90-task benchmark

2. **BDDL file structure**:
   ```lisp
   (:objects
     object_name_1 - object_type
     object_name_2 - object_type
   )
   
   (:init
     (On object_name_1 region_name)
   )
   
   (:goal
     (And (On object_1 object_2))
   )
   ```

3. **Add custom objects**: You can add objects from other task suites by:
   - Adding the object to the `:objects` section
   - Defining initial placement in `:init`
   - Optionally including in `:goal` if relevant

### Verification

To verify LIBERO is using workspace version:
```bash
python /workspace/lerobot/test_libero_workspace.py
```

Expected output should show:
- ✅ Using WORKSPACE version
- BDDL path pointing to `/workspace/lerobot/third_party/LIBERO/`

### Benefits of Workspace Installation

1. **Easy customization** - Modify BDDL files, add objects, create custom tasks
2. **Version control** - All changes tracked in git
3. **Reproducibility** - Team uses identical version with modifications
4. **No loss of changes** - Won't be overwritten on container rebuild
5. **Development workflow** - Instant changes without reinstallation
