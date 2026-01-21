# do this based on https://github.com/huggingface/lerobot/issues/2641
#!/usr/bin/env bash
set -euo pipefail
pip install -U pip setuptools wheel
pip uninstall -y transformers || true
pip install -e ".[pi]" --no-cache-dir
python -c "import transformers; print('transformers:', transformers.__file__)"