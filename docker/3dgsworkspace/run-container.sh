#!/usr/bin/env bash
set -euo pipefail

# Run 3DGS training container.
# USAGE: DATASET_PATH=/path/to/dataset ./run-container.sh
#
# ENVIRONMENT:
#   DATASET_PATH   Host path to dataset (required)
#   OUTPUT_PATH    Host path for training output (default: ./output)
#   CONTAINER_MEM  Memory limit (default: 12g)
#   DETACHED       Run in detached mode (default: false)

if [[ -z "${DATASET_PATH:-}" ]]; then
    echo "ERROR: DATASET_PATH is not set."
    echo "Usage: DATASET_PATH=/path/to/dataset $0"
    exit 1
fi

DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
DETACHED="${DETACHED:-false}"

echo "=========================================="
echo "  3D Gaussian Splatting Training"
echo "=========================================="
echo "  Dataset        : $DATASET_PATH"
echo "  Output         : $OUTPUT_PATH"
echo "  Container mem  : ${CONTAINER_MEM:-12g}"
echo "  Detached       : $DETACHED"
echo "=========================================="
echo ""

# Pass host UID/GID so files created inside the container are owned by
# the host user — no sudo or chown needed on the host side.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

if [[ "$DETACHED" == "true" ]]; then
    # Remove any existing container with the same name (from previous failed runs)
    docker rm -f 3dgs-workspace-train >/dev/null 2>&1 || true
    
    echo "Starting container in detached mode..."
    docker run -d \
      --name 3dgs-workspace-train \
      --user "${HOST_UID}:${HOST_GID}" \
      --gpus all \
      -m "${CONTAINER_MEM:-12g}" \
      -v "$DATASET_PATH":/dataset:rw \
      -v "$OUTPUT_PATH":/output:rw \
      -w /dataset \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      3dgsworkspace:latest \
      bash -c 'python3 $GS_PATH/train.py -s /dataset -m /output'
    
    CONTAINER_ID=$(docker ps -q -f name=3dgs-workspace-train)
    echo "Container started: $CONTAINER_ID"
    echo "Attaching to logs (Ctrl+C to detach, container continues in background)..."
    echo "To stop: docker stop 3dgs-workspace-train"
    echo ""
    
    # Trap Ctrl+C so user can detach from logs without stopping container
    trap 'echo -e "\nDetached from logs. Container continues running..."' INT
    
    # Attach to logs in background
    docker logs -f 3dgs-workspace-train 2>&1 &
    LOGS_PID=$!
    
    # Wait for container to finish
    docker wait 3dgs-workspace-train >/dev/null 2>&1
    
    # Kill logs process if still running
    kill $LOGS_PID 2>/dev/null || true
    wait $LOGS_PID 2>/dev/null || true
    
    # Clean up the container
    docker rm -f 3dgs-workspace-train >/dev/null 2>&1 || true
    
    # Remove the trap
    trap - INT
else
    docker run --rm \
      --name 3dgs-workspace-train \
      --user "${HOST_UID}:${HOST_GID}" \
      --gpus all \
      -it \
      -m "${CONTAINER_MEM:-12g}" \
      -v "$DATASET_PATH":/dataset:rw \
      -v "$OUTPUT_PATH":/output:rw \
      -w /dataset \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      3dgsworkspace:latest \
      bash -c 'python3 $GS_PATH/train.py -s /dataset -m /output'
fi
