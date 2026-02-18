#!/bin/bash
# Setup script for LIBERO third-party package in lerobot workspace

set -e  # Exit on error

echo "=================================================="
echo "LIBERO Workspace Setup"
echo "=================================================="

WORKSPACE_ROOT="/workspace/lerobot"
LIBERO_PATH="$WORKSPACE_ROOT/third_party/LIBERO"
CONFIG_FILE="$HOME/.libero/config.yaml"

# Check if LIBERO is cloned
if [ ! -d "$LIBERO_PATH" ]; then
    echo "❌ LIBERO not found at $LIBERO_PATH"
    echo "Please clone LIBERO first:"
    echo "  cd $WORKSPACE_ROOT/third_party"
    echo "  git clone https://github.com/ARISE-Initiative/LIBERO.git"
    exit 1
fi

echo "✓ LIBERO found at $LIBERO_PATH"

# Create datasets directory if it doesn't exist
mkdir -p "$LIBERO_PATH/datasets"
echo "✓ Created datasets directory"

# Create .libero config directory
mkdir -p "$HOME/.libero"
echo "✓ Created config directory"

# Update LIBERO config to point to workspace
cat > "$CONFIG_FILE" << EOF
assets: $LIBERO_PATH/libero/libero/assets
bddl_files: $LIBERO_PATH/libero/libero/bddl_files
benchmark_root: $LIBERO_PATH/libero/libero
datasets: $LIBERO_PATH/datasets
init_states: $LIBERO_PATH/libero/libero/init_files
EOF

echo "✓ Updated LIBERO config: $CONFIG_FILE"

# Add PYTHONPATH to .bashrc if not already there
if ! grep -q "LIBERO:.*PYTHONPATH" ~/.bashrc; then
    echo 'export PYTHONPATH="/workspace/lerobot/third_party/LIBERO:$PYTHONPATH"' >> ~/.bashrc
    echo "✓ Added LIBERO to PYTHONPATH in ~/.bashrc"
else
    echo "✓ PYTHONPATH already configured"
fi

# Also add conda activate to .bashrc if not there
if ! grep -q "conda activate lerobot" ~/.bashrc; then
    echo 'conda activate lerobot' >> ~/.bashrc
    echo "✓ Added conda activate lerobot to ~/.bashrc"
else
    echo "✓ Conda activation already configured"
fi

# Install required packages for LIBERO
echo ""
echo "Installing dependencies..."
pip install gym==0.26.2 pyyaml -q
echo "✓ Installed gym==0.26.2 (required by LIBERO)"
echo "✓ Installed pyyaml (required by LIBERO)"

echo ""
echo "=================================================="
echo "✅ Setup completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Open a new terminal or run: source ~/.bashrc"
echo "2. Verify setup: python $WORKSPACE_ROOT/test_libero_workspace.py"
echo ""
echo "To customize LIBERO tasks, edit BDDL files in:"
echo "  $LIBERO_PATH/libero/libero/bddl_files/"
echo ""
