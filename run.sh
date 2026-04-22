#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 3DGS Workspace - Unified CLI
# =============================================================================
#
# A single entry point for Gaussian Splatting scene reconstruction.
# Supports both 360° panorama and regular photo workflows.
#
# Usage:
#   ./run.sh <command> [options]
#
# Commands:
#   panorama   Process 360° equirectangular images (panorama/ → images/)
#   colmap     Run COLMAP SfM on regular photos (input/ → images/ + sparse/0/)
#   train      Run 3D Gaussian Splatting training (images/ → output/)
#   inpaint    Interactive 360° image cleanup
#   pipeline   Full pipeline: (panorama|photos) → train
#
# Options:
#   --dataset PATH   Dataset path (required)
#   --mode MODE      Pipeline mode: panorama | photos (required for pipeline)
#   --output PATH    Training output directory (default: dataset/output)
#   --resolution N   Training resolution (1=full, 2=half, default: 1)
#   --pull           Force pull latest images from Docker Hub
#   --help           Show this help
#
# Dataset structure:
#   Panorama:  dataset/panorama/*.jpg  →  run panorama  →  dataset/images/
#   Photos:    dataset/input/*.jpg     →  run colmap    →  dataset/images/ + sparse/0/
#   Training:  dataset/images/ + sparse/0/  →  run train  →  dataset/output/
#
# Docker images (local-first, pulled from Hub if missing):
#   3dgs-colmap      - COLMAP + Panorama SfM
#   3dgsworkspace    - 3DGS Training
#   lama-inpaint     - Inpainting
# =============================================================================

DOCKER_HUB_ORG="ericksuzart"
DEFAULT_MEM="15g"

# Defaults
COMMAND=""
DATASET_PATH=""
OUTPUT_PATH=""
RESOLUTION="4"
MODE=""
FORCE_PULL="false"

# Helpers

usage() {
    cat <<EOF
3DGS Workspace - Unified CLI
========================================

Usage:
  ./run.sh <command> [options]

Commands:
  panorama   Process 360° equirectangular images (panorama/ → images/)
  colmap     Run COLMAP SfM on regular photos (input/ → images/ + sparse/0/)
  train      Run 3D Gaussian Splatting training (images/ → output/)
  inpaint    Interactive 360° image cleanup
  pipeline   Full pipeline: (panorama|photos) → train

Options:
  --dataset PATH   Dataset path (required)
  --mode MODE      Pipeline mode: panorama | photos (required for pipeline)
  --output PATH    Training output directory (default: dataset/output)
  --resolution N   Training resolution (1=full, 2=half, default: 1)
  --pull           Force pull latest images from Docker Hub
  --help           Show this help

Dataset structure:
  Panorama:  dataset/panorama/*.jpg  →  run panorama  →  dataset/images/
  Photos:    dataset/input/*.jpg     →  run colmap    →  dataset/images/ + sparse/0/
  Training:  dataset/images/ + sparse/0/  →  run train  →  dataset/output/

Examples:
  ./run.sh panorama --dataset /data/scene
  ./run.sh colmap   --dataset /data/scene
  ./run.sh train    --dataset /data/scene
  ./run.sh train    --dataset /data/scene --resolution 2
  ./run.sh pipeline --dataset /data/scene --mode panorama
  ./run.sh pipeline --dataset /data/scene --mode photos
  ./run.sh inpaint  --dataset /data/360

Run in background:
  screen -dmS 3dgs ./run.sh pipeline --dataset /data/scene --mode panorama
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

ensure_dataset() {
    [[ -n "$DATASET_PATH" ]] || die "--dataset is required"
    [[ -d "$DATASET_PATH" ]] || die "Dataset path does not exist: $DATASET_PATH"
    DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"
}

host_user() {
    echo "$(id -u):$(id -g)"
}

# Resolve Docker image: local first, fallback to Hub
resolve_image() {
    local name="$1"
    local local_img="${name}:latest"
    local hub_img="${DOCKER_HUB_ORG}/${name}"

    if [[ "$FORCE_PULL" == "true" ]]; then
        docker pull "$hub_img"
        echo "$hub_img"
        return
    fi

    # Prefer local if available
    if docker image inspect "$local_img" >/dev/null 2>&1; then
        echo "$local_img"
        return
    fi

    # Try pulling from Hub
    echo "Pulling ${hub_img} from Docker Hub..."
    docker pull "$hub_img"
    echo "$hub_img"
}

# Run a container interactively (foreground)
run_container() {
    docker run --rm -it \
        --user "$(host_user)" \
        --gpus all \
        -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
        --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        "$@"
}

# Commands

cmd_panorama() {
    ensure_dataset
    [[ -d "$DATASET_PATH/panorama" ]] || die "No panorama/ directory found in $DATASET_PATH"

    local image
    image=$(resolve_image "3dgs-colmap")

    echo "=========================================="
    echo "  Panorama SfM"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Image:   $image"
    echo "=========================================="
    echo ""

    run_container \
        --entrypoint "" \
        -v "$DATASET_PATH:/dataset:rw" \
        -w /dataset \
        "$image" \
        bash -c "python3 /colmap/python/examples/panorama_sfm.py \
            --input_image_path /dataset/panorama \
            --output_path /dataset \
            --matcher exhaustive \
            --pano_render_type overlapping"

    # Remove obstruction cameras (0-3) from images and COLMAP sparse files
    for i in 0 1 2 3; do
        rm -rf "$DATASET_PATH/images/pano_camera${i}" 2>/dev/null || true
    done

    # Clean COLMAP sparse/0/ to remove references to deleted images
    docker run --rm \
        --user "$(host_user)" \
        --entrypoint /usr/bin/python3 \
        -v "$DATASET_PATH:/dataset:rw" \
        -w /dataset \
        "$image" \
        /usr/local/bin/clean_colmap_sparse.py /dataset "_camera0,_camera1,_camera2,_camera3" || true

    # Verify sparse reconstruction succeeded
    if [[ ! -f "$DATASET_PATH/sparse/0/images.bin" ]] && [[ ! -f "$DATASET_PATH/sparse/0/images.txt" ]]; then
        die "Panorama SfM failed to create a sparse model. Try with more panorama images (at least 3 with good overlap)."
    fi

    local count
    count=$(find "$DATASET_PATH/images" -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo ""
    echo "=========================================="
    echo "  Panorama complete"
    echo "  Cameras: $((count - 1)) → $DATASET_PATH/images/"
    echo "  Sparse:  $DATASET_PATH/sparse/"
    echo "=========================================="
}

cmd_colmap() {
    ensure_dataset
    [[ -d "$DATASET_PATH/input" ]] || die "No input/ directory found in $DATASET_PATH"

    local image
    image=$(resolve_image "3dgs-colmap")

    echo "=========================================="
    echo "  COLMAP Preprocessing"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Image:   $image"
    echo "=========================================="
    echo ""

    run_container \
        --entrypoint "" \
        -v "$DATASET_PATH:/dataset:rw" \
        -v "$(pwd):/workspace:rw" \
        -w /workspace \
        "$image" \
        bash convert.sh -s /dataset --resize --magick_executable ""

    # Verify sparse reconstruction succeeded
    if [[ ! -f "$DATASET_PATH/sparse/0/images.bin" ]] && [[ ! -f "$DATASET_PATH/sparse/0/images.txt" ]]; then
        die "COLMAP failed to create a sparse model. Check image overlap and quality."
    fi

    echo ""
    echo "=========================================="
    echo "  COLMAP complete"
    echo "  Images: $DATASET_PATH/images/"
    echo "  Sparse: $DATASET_PATH/sparse/0/"
    echo "=========================================="
}

cmd_train() {
    ensure_dataset
    [[ -d "$DATASET_PATH/images" ]] || die "No images/ directory found in $DATASET_PATH"
    [[ -d "$DATASET_PATH/sparse/0" ]] || die "No sparse/0/ directory found. Run panorama or colmap first."

    local output="${OUTPUT_PATH:-$DATASET_PATH/output}"
    mkdir -p "$output"
    chown "$(id -u):$(id -g)" "$output" 2>/dev/null || true

    local image
    image=$(resolve_image "3dgsworkspace")

    echo "=========================================="
    echo "  3D Gaussian Splatting Training"
    echo "=========================================="
    echo "  Dataset:  $DATASET_PATH"
    echo "  Output:   $output"
    echo "  Image:    $image"
    echo "  Res:      $RESOLUTION"
    echo "=========================================="
    echo ""

    run_container \
        -v "$DATASET_PATH:/dataset:rw" \
        -v "$output:/output:rw" \
        -w /dataset \
        -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024 \
        "$image" \
        bash -c "python3 \$GS_PATH/train.py -s /dataset -m /output -r $RESOLUTION"

    echo ""
    echo "=========================================="
    echo "  Training complete"
    echo "  Output: $output"
    echo "=========================================="
}

cmd_inpaint() {
    ensure_dataset

    local image
    image=$(resolve_image "lama-inpaint")

    echo "=========================================="
    echo "  Interactive Inpainting"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Image:   $image"
    echo "=========================================="
    echo ""

    local display_arg=()
    if [[ -n "${DISPLAY:-}" ]]; then
        display_arg=(-e "DISPLAY=$DISPLAY")
    fi

    docker run --rm -it \
        --gpus all \
        --user "$(host_user)" \
        -v "$DATASET_PATH:/data:rw" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
        "${display_arg[@]}" \
        -e DATASET_PATH=/data \
        -e MODE=full \
        -e INPAINT_BACKEND=sd2 \
        "$image"
}

cmd_pipeline() {
    ensure_dataset
    [[ -n "$MODE" ]] || die "--mode is required for pipeline (panorama|photos)"
    [[ "$MODE" == "panorama" || "$MODE" == "photos" ]] || die "Invalid mode: $MODE (must be panorama or photos)"

    local output="${OUTPUT_PATH:-$DATASET_PATH/output}"

    echo "=========================================="
    echo "  Full Pipeline"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Mode:    $MODE"
    echo "  Output:  $output"
    echo "=========================================="
    echo ""

    if [[ "$MODE" == "panorama" ]]; then
        echo ">>> Step 1/2: Panorama SfM"
        cmd_panorama
        echo ""
    elif [[ "$MODE" == "photos" ]]; then
        echo ">>> Step 1/2: COLMAP"
        cmd_colmap
        echo ""
    fi

    echo ">>> Step 2/2: Training"
    cmd_train
    echo ""
    echo ">>> Pipeline complete!"
    echo ">>> Output: $output"
}

# Argument parsing

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            panorama|colmap|train|inpaint|pipeline)
                COMMAND="$1"; shift ;;
            --dataset)
                [[ $# -ge 2 ]] || die "--dataset requires a value"
                DATASET_PATH="$2"; shift 2 ;;
            --mode)
                [[ $# -ge 2 ]] || die "--mode requires a value (panorama|photos)"
                MODE="$2"; shift 2 ;;
            --output)
                [[ $# -ge 2 ]] || die "--output requires a value"
                OUTPUT_PATH="$2"; shift 2 ;;
            --resolution)
                [[ $# -ge 2 ]] || die "--resolution requires a value"
                RESOLUTION="$2"; shift 2 ;;
            --pull)
                FORCE_PULL="true"; shift ;;
            --help|-h)
                usage; exit 0 ;;
            *)
                die "Unknown option: $1" ;;
        esac
    done
}

# Main

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

# First arg is command (unless it's a flag)
case "$1" in
    panorama|colmap|train|inpaint|pipeline|--help|-h)
        if [[ "$1" == --help || "$1" == -h ]]; then
            usage; exit 0
        fi
        COMMAND="$1"; shift
        parse_args "$@"
        ;;
    *)
        die "Unknown command: $1"
        ;;
esac

# Check Docker
command -v docker >/dev/null 2>&1 || die "docker is not installed"
docker info >/dev/null 2>&1 || die "Cannot connect to Docker daemon"

# Dispatch
case "$COMMAND" in
    panorama) cmd_panorama ;;
    colmap)   cmd_colmap ;;
    train)    cmd_train ;;
    inpaint)  cmd_inpaint ;;
    pipeline) cmd_pipeline ;;
esac
