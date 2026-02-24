#!/usr/bin/env bash
set -e

DOCKER_IMG_NAME=lerobot_uq-dev:latest

CONTAINER_NAME="lerobot_uq-dev"
IMAGE_NAME=$DOCKER_IMG_NAME

PROJECT="lerobot_libero_uq"
WORKSPACE="$HOME/code/vla/$PROJECT/workspace"
OUTPUTS="$HOME/code/vla/$PROJECT/outputs"
CACHE_HF="$HOME/code/vla/$PROJECT/cache_hf"
CACHE_PIP="$HOME/code/vla/$PROJECT/cache_pip"
CACHE_RUNTIME="$HOME/code/vla/$PROJECT/cache_runtime"

# Create directories if they do not exist
mkdir -p "$WORKSPACE" "$OUTPUTS" "$CACHE_HF" "$CACHE_PIP" "$CACHE_RUNTIME"
mkdir -p "$CACHE_RUNTIME/triton" "$CACHE_RUNTIME/torchinductor" "$CACHE_RUNTIME/xdg"

# If container already exists, do not recreate it
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container ${CONTAINER_NAME} already exists."
  echo "Use: docker start ${CONTAINER_NAME}"
  exit 0
fi

docker run -it \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/cache/huggingface \
  -e TRANSFORMERS_CACHE=/cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/cache/huggingface/datasets \
  -e PIP_CACHE_DIR=/cache/pip \
  -e XDG_CACHE_HOME=/cache/runtime/xdg \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/runtime/torchinductor \
  -e TRITON_CACHE_DIR=/cache/runtime/triton \
  --mount type=bind,source="${WORKSPACE}",target=/workspace \
  --mount type=bind,source="${OUTPUTS}",target=/outputs \
  --mount type=bind,source="${CACHE_HF}",target=/cache/huggingface \
  --mount type=bind,source="${CACHE_PIP}",target=/cache/pip \
  --mount type=bind,source="${CACHE_RUNTIME}",target=/cache/runtime \
  -w /workspace \
  "${IMAGE_NAME}" \
/bin/bash
