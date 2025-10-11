#!/bin/bash

# Set image name and container name from arguments or use defaults
IMAGE="${1:-robobase:humble}"
CONTAINER_NAME="${2:-humble_dev}"

# Use third argument as host repo path, or default to current directory
HOST_REPO_PATH="${3:-$(pwd)}"
CONTAINER_REPO_PATH="/root/robows/src/robobase"

# Shift to allow passing extra -v options as further arguments
shift 3 || true

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already exists."
    
    # Check if container is running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is running. Executing into it..."
        docker exec -it "${CONTAINER_NAME}" bash
    else
        echo "Container is stopped. Starting and executing into it..."
        docker start "${CONTAINER_NAME}"
        docker exec -it "${CONTAINER_NAME}" bash
    fi
    exit 0
fi

echo "Creating new container ${CONTAINER_NAME}..."

# X11 and user environment (host must provide XAUTH/XAUTHORITY)
XSOCK="/tmp/.X11-unix"
XAUTH="${XAUTHORITY:-$HOME/.Xauthority}"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_OPTS="--runtime nvidia --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all"
else
    GPU_OPTS=""
fi

# --privileged gives access to all host devices (USB, etc.)
docker run -v "${HOST_REPO_PATH}:${CONTAINER_REPO_PATH}" \
    -it \
    --privileged \
    --net=host \
    --ipc=host \
    -v /dev/shm:/dev/shm \
    -v "${XSOCK}:${XSOCK}" \
    -v /dev/dri:/dev/dri \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY="${XAUTH}" \
    -v "${XAUTH}:${XAUTH}" \
    --name "${CONTAINER_NAME}" \
    ${GPU_OPTS} \
    "$@" \
    "${IMAGE}"
