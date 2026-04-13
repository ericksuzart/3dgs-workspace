#!/usr/bin/env bash
set -euo pipefail

#
# Run COLMAP's panorama_sfm.py example inside the 3dgs-colmap container.
# After the pipeline completes, flattens the output so all rendered
# perspective images live in a single folder with the virtual-camera
# folder name as a prefix (e.g.  pano_camera0_panorama_img.jpg).
#
# USAGE:
#   DATASET_PATH=/path/to/panoramas ./docker/3dgs-colmap/run-panorama-sfm.sh
#
# The DATASET_PATH should contain 360° equirectangular panorama images.
# Output is written to: $DATASET_PATH/output/
#
# OPTIONS (all optional, via environment):
#   MATCHER          Feature matching strategy (default: sequential)
#                    Choices: sequential, exhaustive, vocabtree, spatial
#   PANO_RENDER_TYPE How to render perspective views (default: overlapping)
#                    Choices: overlapping, non-overlapping
#   CONTAINER_MEM    Memory limit for container (default: 12g)

if [[ -z "${DATASET_PATH:-}" ]]; then
    echo "ERROR: DATASET_PATH is not set."
    echo "Usage: DATASET_PATH=/path/to/panoramas $0"
    exit 1
fi

if [[ ! -d "$DATASET_PATH" ]]; then
    echo "ERROR: DATASET_PATH '$DATASET_PATH' does not exist."
    exit 1
fi

DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"

MATCHER="${MATCHER:-sequential}"
PANO_RENDER_TYPE="${PANO_RENDER_TYPE:-overlapping}"
DETACHED="${DETACHED:-false}"

# Pass host UID/GID so files created inside the container are owned by
# the host user — no sudo or chown needed on the host side.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

echo "=========================================="
echo "  COLMAP Panorama SfM"
echo "=========================================="
echo "  Dataset        : $DATASET_PATH"
echo "  Output         : $DATASET_PATH/output/"
echo "  Matcher        : $MATCHER"
echo "  Render type    : $PANO_RENDER_TYPE"
echo "  Container mem  : ${CONTAINER_MEM:-12g}"
echo "  Detached       : $DETACHED"
echo "=========================================="
echo ""

# Build the command to run inside the container
# Note: The Dockerfile has ENTRYPOINT [ "bash" ], so we need to override it
# with --entrypoint to run our custom command properly
if [[ "$DETACHED" == "true" ]]; then
    # Remove any existing container with the same name (from previous failed runs)
    docker rm -f 3dgs-colmap-panorama >/dev/null 2>&1 || true
    
    echo "Starting container in detached mode..."
    docker run -d \
      --name 3dgs-colmap-panorama \
      --user "${HOST_UID}:${HOST_GID}" \
      --gpus all \
      --entrypoint "" \
      -m "${CONTAINER_MEM:-12g}" \
      -v "$DATASET_PATH":/dataset:rw \
      -w /dataset \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      3dgs-colmap:latest \
      bash -c "python3 /colmap/python/examples/panorama_sfm.py \
        --input_image_path /dataset/panorama \
        --output_path /dataset/output \
        --matcher $MATCHER \
        --pano_render_type $PANO_RENDER_TYPE"
    
    CONTAINER_ID=$(docker ps -q -f name=3dgs-colmap-panorama)
    echo "Container started: $CONTAINER_ID"
    echo "Attaching to logs (Ctrl+C to detach, container continues in background)..."
    echo "To stop: docker stop 3dgs-colmap-panorama"
    echo ""
    
    # Attach to logs in background
    docker logs -f 3dgs-colmap-panorama 2>&1 &
    LOGS_PID=$!
    
    # Wait for container to finish
    EXIT_CODE=$(docker wait 3dgs-colmap-panorama 2>/dev/null || echo "1")
    
    # Kill logs process if still running
    kill $LOGS_PID 2>/dev/null || true
    wait $LOGS_PID 2>/dev/null || true
    
    # Clean up the container
    docker rm -f 3dgs-colmap-panorama >/dev/null 2>&1 || true
    
    if [[ "$EXIT_CODE" != "0" ]]; then
        echo ""
        echo "ERROR: Container exited with code $EXIT_CODE"
        exit 1
    fi
else
    docker run --rm \
      --name 3dgs-colmap-panorama \
      --user "${HOST_UID}:${HOST_GID}" \
      --gpus all \
      --entrypoint "" \
      -it \
      -m "${CONTAINER_MEM:-12g}" \
      -v "$DATASET_PATH":/dataset:rw \
      -w /dataset \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      3dgs-colmap:latest \
      bash -c "python3 /colmap/python/examples/panorama_sfm.py \
        --input_image_path /dataset/panorama \
        --output_path /dataset/output \
        --matcher $MATCHER \
        --pano_render_type $PANO_RENDER_TYPE"
fi

# flatten the image directory
# Copies files from subdirectories into the parent, prefixing with the
# subdirectory name.  Then removes the now-empty subdirectories.
#   images/pano_camera0/panorama_img.jpg → images/pano_camera0_panorama_img.jpg
#
IMAGES_DIR="$DATASET_PATH/output/images"
MASKS_DIR="$DATASET_PATH/output/masks"

flatten_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        return
    fi
    local subdirs=()
    for entry in "$dir"/*/; do
        [[ -d "$entry" ]] && subdirs+=("$entry")
    done
    [[ ${#subdirs[@]} -eq 0 ]] && return

    echo "Flattening $dir (${#subdirs[@]} subdirectories) …"
    for subdir in "${subdirs[@]}"; do
        local prefix
        prefix="$(basename "$subdir")"
        find "$subdir" -maxdepth 1 -type f | while IFS= read -r file; do
            local name
            name="$(basename "$file")"
            cp "$file" "${dir}/${prefix}_${name}"
        done
    done
    find "$dir" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
}

flatten_dir "$IMAGES_DIR"
flatten_dir "$MASKS_DIR"

# Summary
echo ""
echo "=========================================="
echo "  Pipeline complete"
echo "=========================================="
if [[ -d "$IMAGES_DIR" ]]; then
    img_count=$(find "$IMAGES_DIR" -maxdepth 1 -type f | wc -l)
    echo "  Rendered images : $img_count  →  $IMAGES_DIR/"
fi
if [[ -d "$MASKS_DIR" ]]; then
    mask_count=$(find "$MASKS_DIR" -maxdepth 1 -type f | wc -l)
    echo "  Masks           : $mask_count  →  $MASKS_DIR/"
fi
echo "  Database        : $DATASET_PATH/output/database.db"
echo "  Sparse model    : $DATASET_PATH/output/sparse/"
echo "=========================================="
