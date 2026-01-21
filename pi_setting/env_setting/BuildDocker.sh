docker build \
    --build-arg USERNAME=$(id -un) \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t lerobot-dev:latest .