DOCKER_IMG_NAME=lerobot_uq-dev:latest
docker build \
    --build-arg USERNAME=$(id -un) \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t $DOCKER_IMG_NAME .