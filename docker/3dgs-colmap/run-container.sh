#!/usr/bin/env bash
set -euo pipefail

# Run COLMAP preprocessing container.
# USAGE: DATASET_PATH=/path/to/dataset ./run-container.sh
#
# ENVIRONMENT:
#   DATASET_PATH   Host path to dataset (required)
#   CONTAINER_MEM  Memory limit (default: 12g)
#   DETACHED       Run in detached mode (default: false)

if [[ -z "${DATASET_PATH:-}" ]]; then
    echo "ERROR: DATASET_PATH is not set."
    echo "Usage: DATASET_PATH=/path/to/dataset $0"
    exit 1
fi

DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"
DETACHED="${DETACHED:-false}"

echo "=========================================="
echo "  COLMAP Preprocessing"
echo "=========================================="
echo "  Dataset        : $DATASET_PATH"
echo "  Container mem  : ${CONTAINER_MEM:-12g}"
echo "  Detached       : $DETACHED"
echo "=========================================="
echo ""

if [[ "$DETACHED" == "true" ]]; then
    # Remove any existing container with the same name (from previous failed runs)
    docker rm -f 3dgs-colmap >/dev/null 2>&1 || true
    
    # Pass host UID/GID so files created inside the container are owned by
    # the host user — no sudo or chown needed on the host side.
    HOST_UID="$(id -u)"
    HOST_GID="$(id -g)"
    
    echo "Starting container in detached mode..."
    docker run -d \
      --name 3dgs-colmap \
      --user "${HOST_UID}:${HOST_GID}" \
      --gpus all \
      --entrypoint "" \
      -m "${CONTAINER_MEM:-12g}" \
      -v "$DATASET_PATH":/dataset:rw \
      -v "$(pwd)":/workspace:rw \
      -w /workspace \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      3dgs-colmap:latest \
      bash convert.sh -s /dataset --resize --magick_executable ""
    
    CONTAINER_ID=$(docker ps -q -f name=3dgs-colmap)
    echo "Container started: $CONTAINER_ID"
    echo "Attaching to logs (Ctrl+C to detach, container continues in background)..."
    echo "To stop: docker stop 3dgs-colmap"
    echo ""
    
    # Trap Ctrl+C so user can detach from logs without stopping container
    trap 'echo -e "\nDetached from logs. Container continues running..."' INT
    
    # Attach to logs in background
    docker logs -f 3dgs-colmap 2>&1 &
    LOGS_PID=$!
    
    # Wait for container to finish
    docker wait 3dgs-colmap >/dev/null 2>&1
    
    # Kill logs process if still running
    kill $LOGS_PID 2>/dev/null || true
    wait $LOGS_PID 2>/dev/null || true
    
    # Clean up the container
    docker rm -f 3dgs-colmap >/dev/null 2>&1 || true
    
    # Remove the trap
    trap - INT
else
    # Pass host UID/GID so files created inside the container are owned by
    # the host user — no sudo or chown needed on the host side.
    HOST_UID="$(id -u)"
    HOST_GID="$(id -g)"
    
    docker run --rm \
      --name 3dgs-colmap \
      --user "${HOST_UID}:${HOST_GID}" \
      --gpus all \
      --entrypoint "" \
      -it \
      -m "${CONTAINER_MEM:-12g}" \
      -v "$DATASET_PATH":/dataset:rw \
      -v "$(pwd)":/workspace:rw \
      -w /workspace \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      3dgs-colmap:latest \
      bash convert.sh -s /dataset --resize --magick_executable ""
fi
