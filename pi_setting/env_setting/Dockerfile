FROM huggingface/lerobot-gpu:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    htop \
    vim \
    tmux \
    ffmpeg \
    gedit \
    wget \
    ca-certificates \
    bzip2 \
 && rm -rf /var/lib/apt/lists/*

# Configure NVIDIA EGL vendor for multi-GPU support
# This allows EGL to detect all NVIDIA GPUs instead of just one
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{\n\
    "file_format_version" : "1.0.0",\n\
    "ICD" : {\n\
        "library_path" : "libEGL_nvidia.so.0"\n\
    }\n\
}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json


# Map Docker arch names to Miniforge naming if needed
RUN OS="$(uname)" \
 && ARCH="$(uname -m)" \
 && if [ "$ARCH" = "x86_64" ]; then ARCH="x86_64"; \
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then ARCH="aarch64"; \
    else echo "Unsupported arch: $ARCH" && exit 1; fi \
 && wget -O /tmp/Miniforge3.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${OS}-${ARCH}.sh" \
 && bash /tmp/Miniforge3.sh -b -p /opt/conda \
 && rm -f /tmp/Miniforge3.sh 

ENV PATH=/opt/conda/bin:$PATH


ARG USERNAME=go68xoq
ARG UID=4254787
ARG GID=3000000

# Create group and user; handle the case where group/user might already exist
RUN if ! getent group ${GID} >/dev/null; then groupadd -g ${GID} ${USERNAME}; fi \
 && if ! id -u ${USERNAME} >/dev/null 2>&1; then useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}; fi

# Optional: ensure mount targets exist and are writable
RUN mkdir -p /workspace /outputs /cache/huggingface /cache/pip /cache/runtime \
 && chown -R ${UID}:${GID} /workspace /outputs /cache /opt/conda || true

# set cache environment variables for pytorch
ENV XDG_CACHE_HOME=/cache/runtime/xdg
ENV TORCHINDUCTOR_CACHE_DIR=/cache/runtime/torchinductor
ENV TRITON_CACHE_DIR=/cache/runtime/triton



USER ${UID}:${GID}
ENV HOME=/home/${USERNAME}
WORKDIR /workspace
