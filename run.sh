#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 3DGS Workspace - Unified CLI
# =============================================================================
#
# A single entry point for all 3DGS workspace operations.
# Works equivalently to running Docker Hub images directly.
#
# Usage:
#   ./run.sh <command> [options]
#
# Commands:
#   colmap     Run COLMAP preprocessing (SfM)
#   train     Run 3DGS training
#   inpaint   Run interactive 360° image cleanup
#   pipeline  Run full pipeline (panorama -> colmap -> train)
#
# Options:
#   --dataset PATH   Dataset path (required)
#   --output PATH   Output path (default: ./output)
#   --image IMAGE   Docker image to use
#   --local        Use local image only (don't pull)
#   --pull         Force pull from Docker Hub
#   --detach      Run in detached mode
#   --help        Show this help
#
# Examples:
#   ./run.sh train --dataset /data/scene
#   ./run.sh colmap --dataset /data/scene
#   ./run.sh pipeline --dataset /data/panorama
#
# Docker Hub Images:
#   ericksuzart/3dgsworkspace   - Training
#   ericksuzart/3dgs-colmap     - COLMAP
#   ericksuzart/lama-inpaint    - Inpainting
# =============================================================================

# Configuration
DOCKER_HUB_ORG="ericksuzart"
DEFAULT_MEM="15g"

# Image mappings
declare -A IMAGES=(
    ["colmap"]="3dgs-colmap"
    ["train"]="3dgsworkspace"
    ["inpaint"]="lama-inpaint"
    ["panorama"]="3dgs-colmap"
    ["pipeline"]="3dgs-colmap"
)

# Default values
COMMAND=""
DATASET_PATH=""
OUTPUT_PATH=""
RESOLUTION=""
USE_LOCAL="${LOCAL:-false}"
FORCE_PULL="${PULL:-false}"
DETACHED="${DETACHED:-true}"

show_help() {
    cat <<EOF
3DGS Workspace - Unified CLI
========================================

Usage:
  ./run.sh <command> [options]

Commands:
  colmap     Run COLMAP preprocessing (SfM)
  train     Run 3DGS training
  panorama  Run Panorama SfM (360° images -> COLMAP input)
  inpaint   Run interactive 360° image cleanup
  pipeline  Run full pipeline: panorama -> colmap -> train

Options:
  --dataset PATH   Dataset path (required for colmap, train, panorama, pipeline)
  --output PATH   Output directory (default: DATASET_PATH/output)
  --resolution N  Native resolution (1 = full resolution, 2 = half, 4 = quarter, etc.)
  --image NAME   Docker image to use (default: auto-resolve)
  --local       Prefer local image, don't pull from Hub
  --pull        Force pull from Docker Hub
  --detach      Run in detached mode (background)
  --help        Show this help

Environment Variables:
  DATASET_PATH    Dataset path (alternative to --dataset)
  OUTPUT_PATH     Output path (alternative to --output)
  CONTAINER_MEM    Memory limit (default: 12g)
  LOCAL            Prefer local image (true/false)
  PULL             Force pull from Hub (true/false)

Examples:
  # Panorama SfM (360° images -> COLMAP input)
  ./run.sh panorama --dataset /data/panorama

  # COLMAP preprocessing (input/ -> images/)
  ./run.sh colmap --dataset /data/my-scene

  # Training (full resolution)
  ./run.sh train --dataset /data/my-scene --resolution 1

  # Training (half resolution - faster)
  ./run.sh train --dataset /data/my-scene --resolution 2

  # Full pipeline (panorama -> colmap -> train)
  ./run.sh pipeline --dataset /data/panorama

  # Inpaint (interactive 360° cleanup)
  ./run.sh inpaint --dataset /data/360

  # Force pull new image from Hub
  ./run.sh train --dataset /data/my-scene --pull

Docker Images (auto-resolved):
  ${DOCKER_HUB_ORG}/3dgsworkspace   Training
  ${DOCKER_HUB_ORG}/3dgs-colmap    COLMAP
  ${DOCKER_HUB_ORG}/lama-inpaint  Inpainting
EOF
}

resolve_image() {
    local image_key="$1"
    local image_name="${IMAGES[$image_key]}"
    local dockerhub="${DOCKER_HUB_ORG}/${image_name}"
    local local="${image_name}:latest"

    if [[ "$FORCE_PULL" == "true" ]]; then
        echo "$dockerhub"
    elif [[ "$USE_LOCAL" == "true" ]]; then
        if docker image inspect "$local" >/dev/null 2>&1; then
            echo "$local"
        else
            echo "$dockerhub"
        fi
    else
        # Default: prefer local, fallback to hub
        if docker image inspect "$local" >/dev/null 2>&1; then
            echo "$local"
        elif docker image inspect "$dockerhub" >/dev/null 2>&1; then
            echo "$dockerhub"
        else
            echo "$dockerhub"
        fi
    fi
}

check_dataset() {
    local required="$1"
    if [[ -z "$DATASET_PATH" ]]; then
        echo "ERROR: --dataset is required for $required"
        echo "Usage: ./run.sh $COMMAND --dataset /path/to/dataset"
        exit 1
    fi
    if [[ ! -d "$DATASET_PATH" ]]; then
        echo "ERROR: Dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
    DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"
}

check_input_dir() {
    local stage="$1"
    if [[ ! -d "$DATASET_PATH/input" ]]; then
        echo "ERROR: $stage requires $DATASET_PATH/input/ (raw images)"
        echo "Place raw images in $DATASET_PATH/input/ first."
        exit 1
    fi
}

check_colmap_output() {
    if [[ ! -d "$DATASET_PATH/images" ]] || [[ ! -d "$DATASET_PATH/sparse/0" ]]; then
        echo "ERROR: Training requires $DATASET_PATH/images/ and $DATASET_PATH/sparse/0/"
        echo "Run 'make colmap' or './run.sh colmap' first."
        exit 1
    fi
}

get_host_user() {
    echo "$(id -u):$(id -g)"
}

run_colmap() {
    check_dataset "colmap"
    check_input_dir "colmap"

    local image
    image=$(resolve_image "colmap")

    echo "=========================================="
    echo "  COLMAP Preprocessing"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Image:   $image"
    echo "  Memory:  ${CONTAINER_MEM:-$DEFAULT_MEM}"
    echo "  Logs:    docker logs -f 3dgs-colmap"
    echo "=========================================="
    echo ""

    # Remove any existing container
    docker rm -f 3dgs-colmap 2>/dev/null || true

    if [[ "$DETACHED" == "true" ]]; then
        docker run -d \
            --name "3dgs-colmap" \
            --user "$(get_host_user)" \
            --gpus all \
            --entrypoint "" \
            -v "$DATASET_PATH:/dataset:rw" \
            -v "$(pwd):/workspace:rw" \
            -w /workspace \
            -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
            --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            "$image" \
            bash convert.sh -s /dataset --resize --magick_executable ""

        # Follow logs until completion
        echo "Following COLMAP logs (Ctrl+C to detach, container continues)..."
        docker logs -f 3dgs-colmap 2>&1 &
        local logs_pid=$!
        trap 'kill $logs_pid 2>/dev/null || true' INT
        
        docker wait 3dgs-colmap >/dev/null 2>&1
        local exit_code=$?
        kill $logs_pid 2>/dev/null || true
        wait $logs_pid 2>/dev/null || true
        docker rm -f 3dgs-colmap >/dev/null 2>&1 || true
        trap - INT

        if [[ "$exit_code" != "0" ]]; then
            echo "ERROR: COLMAP failed with exit code $exit_code"
            exit 1
        fi
    else
        docker run --rm -it \
            --name "3dgs-colmap" \
            --user "$(get_host_user)" \
            --gpus all \
            --entrypoint "" \
            -v "$DATASET_PATH:/dataset:rw" \
            -v "$(pwd):/workspace:rw" \
            -w /workspace \
            -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
            --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            "$image" \
            bash convert.sh -s /dataset --resize --magick_executable ""
    fi
}

run_train() {
    check_dataset "train"

    # Also allow running if images/ + sparse/0 exist (already processed)
    if [[ ! -d "$DATASET_PATH/images" ]] && [[ ! -d "$DATASET_PATH/sparse/0" ]]; then
        if [[ -d "$DATASET_PATH/input" ]]; then
            echo "ERROR: Data not processed. Run colmap first."
            echo "  ./run.sh colmap --dataset $DATASET_PATH"
            exit 1
        fi
    fi

    local image
    image=$(resolve_image "train")

    OUTPUT_PATH="${OUTPUT_PATH:-$DATASET_PATH/output}"

    echo "=========================================="
    echo "  3D Gaussian Splatting Training"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Output:  $OUTPUT_PATH"
    echo "  Image:   $image"
    echo "  Memory: ${CONTAINER_MEM:-$DEFAULT_MEM}"
    echo "  Logs:    docker logs -f 3dgs-workspace-train"
    echo "=========================================="
    echo ""

    # Remove any existing container
    docker rm -f 3dgs-workspace-train 2>/dev/null || true

    if [[ "$DETACHED" == "true" ]]; then
        docker run -d \
            --name "3dgs-workspace-train" \
            --user "$(get_host_user)" \
            --gpus all \
            -v "$DATASET_PATH:/dataset:rw" \
            -v "$OUTPUT_PATH:/output:rw" \
            -w /dataset \
            -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
            --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            "$image" \
            bash -c "python3 \$GS_PATH/train.py -s /dataset -m /output${RESOLUTION:+ -r $RESOLUTION}"

        # Follow logs until completion
        echo "Following Training logs (Ctrl+C to detach, container continues)..."
        docker logs -f 3dgs-workspace-train 2>&1 &
        local logs_pid=$!
        trap 'kill $logs_pid 2>/dev/null || true' INT
        
        docker wait 3dgs-workspace-train >/dev/null 2>&1
        local exit_code=$?
        kill $logs_pid 2>/dev/null || true
        wait $logs_pid 2>/dev/null || true
        docker rm -f 3dgs-workspace-train >/dev/null 2>&1 || true
        trap - INT

        if [[ "$exit_code" != "0" ]]; then
            echo "ERROR: Training failed with exit code $exit_code"
            exit 1
        fi
    else
        docker run --rm -it \
            --name "3dgs-workspace-train" \
            --user "$(get_host_user)" \
            --gpus all \
            -v "$DATASET_PATH:/dataset:rw" \
            -v "$OUTPUT_PATH:/output:rw" \
            -w /dataset \
            -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
            --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            "$image" \
            bash -c "python3 \$GS_PATH/train.py -s /dataset -m /output${RESOLUTION:+ -r $RESOLUTION}"
    fi
}

run_panorama() {
    check_dataset "panorama"

    # Check for panorama images
    local panorama_dir="$DATASET_PATH/panorama"
    if [[ ! -d "$panorama_dir" ]]; then
        if [[ -n "$(ls -A "$DATASET_PATH"/*.jpg "$DATASET_PATH"/*.png 2>/dev/null)" ]]; then
            panorama_dir="$DATASET_PATH"
        else
            echo "ERROR: No images found in $DATASET_PATH/panorama/"
            exit 1
        fi
    fi

    # Panorama outputs to $DATASET_PATH, then we rename images/ to input/
    local panorama_output="$DATASET_PATH"

    echo "=========================================="
    echo "  Panorama SfM"
    echo "=========================================="
    echo "  Input:   $panorama_dir"
    echo "  Output:  $panorama_output"
    echo "  Matcher: spatial"
    echo "  Render:  overlapping"
    echo "  Logs:    docker logs -f 3dgs-panorama"
    echo "=========================================="
    echo ""

    local image
    image=$(resolve_image "panorama")

    # Remove any existing container
    docker rm -f 3dgs-panorama 2>/dev/null || true

    # Run in detached mode
    docker run -d \
        --name "3dgs-panorama" \
        --user "$(get_host_user)" \
        --gpus all \
        --entrypoint "" \
        -v "$DATASET_PATH:/dataset:rw" \
        -w /dataset \
        -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
        --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        "$image" \
        bash -c "python3 /colmap/python/examples/panorama_sfm.py \
            --input_image_path /dataset/panorama \
            --output_path /dataset \
            --matcher spatial \
            --pano_render_type overlapping"

    # Follow logs until completion
    echo "Following Panorama logs (Ctrl+C to detach, container continues)..."
    docker logs -f 3dgs-panorama 2>&1 &
    local logs_pid=$!
    trap 'kill $logs_pid 2>/dev/null || true' INT
    
    docker wait 3dgs-panorama >/dev/null 2>&1
    local exit_code=$?
    kill $logs_pid 2>/dev/null || true
    wait $logs_pid 2>/dev/null || true
    docker rm -f 3dgs-panorama >/dev/null 2>&1 || true
    trap - INT

    if [[ "$exit_code" != "0" ]]; then
        echo "ERROR: Panorama failed with exit code $exit_code"
        exit 1
    fi

    # Rename images/ to input/
    if [[ -d "$panorama_output/images" ]]; then
        if [[ -d "$panorama_output/input" ]]; then
            rm -rf "$panorama_output/input"
        fi
        mv "$panorama_output/images" "$panorama_output/input"
    fi

    # Flatten images first
    flatten_images() {
        local dir="$1"
        if [[ ! -d "$dir" ]]; then return; fi
        for subdir in "$dir"/*/; do
            [[ -d "$subdir" ]] || continue
            local prefix
            prefix="$(basename "$subdir")"
            for f in "$subdir"/*; do
                [[ -f "$f" ]] || continue
                local name
                name="$(basename "$f")"
                cp "$f" "${dir}/${prefix}_${name}"
            done
        done
        find "$dir" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
    }
    flatten_images "$panorama_output/input" 2>/dev/null || true
    flatten_images "$panorama_output/masks" 2>/dev/null || true

    # Remove images from excluded camera indices (0-3 typically have obstruction)
    # These were flattened with prefix pano_camera0_, pano_camera1_, etc.
    for i in 0 1 2 3; do
        rm -f "$panorama_output/input/pano_camera${i}_"*.jpg 2>/dev/null || true
        rm -f "$panorama_output/input/pano_camera${i}_"*.png 2>/dev/null || true
    done

    # Summary
    echo ""
    echo "=========================================="
    echo "  Panorama complete"
    echo "=========================================="
    if [[ -d "$panorama_output/input" ]]; then
        img_count=$(find "$panorama_output/input" -maxdepth 1 -type f | wc -l)
        echo "  Images: $img_count -> $panorama_output/input/"
    fi
    if [[ -d "$panorama_output/masks" ]]; then
        mask_count=$(find "$panorama_output/masks" -maxdepth 1 -type f | wc -l)
        echo "  Masks:  $mask_count -> $panorama_output/masks/"
    fi
    echo "  Database: $panorama_output/database.db"
    echo "  Sparse:   $panorama_output/sparse/"
    echo "=========================================="
}

run_inpaint() {
    if [[ -z "$DATASET_PATH" ]]; then
        DATASET_PATH="${DATASET_PATH:-./data}"
    fi
    if [[ ! -d "$DATASET_PATH" ]]; then
        echo "ERROR: Dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
    DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"

    local image
    image=$(resolve_image "inpaint")

    echo "=========================================="
    echo "  Interactive Inpainting"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Image:   $image"
    echo "  Memory: ${CONTAINER_MEM:-$DEFAULT_MEM}"
    echo "=========================================="
    echo ""

    # X11 forwarding
    local display_arg=()
    if [[ -n "${DISPLAY:-}" ]]; then
        display_arg=("-e" "DISPLAY=$DISPLAY")
    fi

    docker run --rm -it \
        --gpus all \
        --user "$(get_host_user)" \
        -v "$DATASET_PATH:/data:rw" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
        "${display_arg[@]}" \
        -e DATASET_PATH=/data \
        -e MODE=full \
        -e INPAINT_BACKEND=sd2 \
        "$image"
}

run_pipeline() {
    check_dataset "pipeline"

    # Check for panorama images
    local panorama_dir="$DATASET_PATH/panorama"
    if [[ ! -d "$panorama_dir" ]]; then
        if [[ -n "$(ls -A "$DATASET_PATH"/*.jpg "$DATASET_PATH"/*.png 2>/dev/null)" ]]; then
            panorama_dir="$DATASET_PATH"
        elif [[ -z "$(ls -A "$DATASET_PATH"/* 2>/dev/null)" ]]; then
            echo "ERROR: No images found in $DATASET_PATH"
            exit 1
        fi
    fi

    # Set output paths
    local colmap_input="$DATASET_PATH/input"    # Created from panorama images/
    local colmap_images="$DATASET_PATH/images"    # COLMAP output
    local colmap_sparse="$DATASET_PATH/sparse/0" # COLMAP sparse
    local train_output="$DATASET_PATH/output"   # Training output

    echo "=========================================="
    echo "  Full Pipeline"
    echo "  (Panorama -> COLMAP -> Training)"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "=========================================="
    echo ""
    echo "  Pipeline structure:"
    echo "    $DATASET_PATH/panorama/  -> INPUT (raw 360° images)"
    echo "    $DATASET_PATH/        -> PANORAMA OUTPUT"
    echo "    $DATASET_PATH/input/   -> (images/ renamed after panorama)"
    echo "    $DATASET_PATH/images/  -> COLMAP OUTPUT"
    echo "    $DATASET_PATH/sparse/0/ -> COLMAP SPARSE"
    echo "    $DATASET_PATH/output/ -> TRAINING OUTPUT"
    echo ""

    local image
    image=$(resolve_image "colmap")

    # PanoramaSfM (detached with log following)
    echo ">>> Panorama SfM (starting in background)..."
    echo "    Input:  $panorama_dir"
    echo "    Output: $DATASET_PATH"
    echo "    Logs:   docker logs -f 3dgs-panorama-pipeline"
    echo ""

    # Remove any existing container
    docker rm -f 3dgs-panorama-pipeline 2>/dev/null || true

    docker run -d \
        --name "3dgs-panorama-pipeline" \
        --user "$(get_host_user)" \
        --gpus all \
        --entrypoint "" \
        -v "$DATASET_PATH:/dataset:rw" \
        -w /dataset \
        -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
        --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        "$image" \
        bash -c "python3 /colmap/python/examples/panorama_sfm.py \
            --input_image_path /dataset/panorama \
            --output_path /dataset \
            --matcher spatial \
            --pano_render_type overlapping"

    # Follow logs until completion
    echo "Following Panorama logs (Ctrl+C to detach, container continues)..."
    docker logs -f 3dgs-panorama-pipeline 2>&1 &
    local logs_pid=$!
    trap 'kill $logs_pid 2>/dev/null || true' INT
    
    # Wait for container to finish
    docker wait 3dgs-panorama-pipeline >/dev/null 2>&1
    local exit_code=$?
    kill $logs_pid 2>/dev/null || true
    wait $logs_pid 2>/dev/null || true
    docker rm -f 3dgs-panorama-pipeline >/dev/null 2>&1 || true
    trap - INT

    if [[ "$exit_code" != "0" ]]; then
        echo "ERROR: Panorama stage failed with exit code $exit_code"
        exit 1
    fi

    # Rename images/ to input/
    if [[ -d "$DATASET_PATH/images" ]]; then
        if [[ -d "$colmap_input" ]]; then
            rm -rf "$colmap_input"
        fi
        mv "$DATASET_PATH/images" "$colmap_input"
    fi

    # Flatten panorama output (images from subdirs to root)
    flatten_images() {
        local dir="$1"
        if [[ ! -d "$dir" ]]; then return; fi
        for subdir in "$dir"/*/; do
            [[ -d "$subdir" ]] || continue
            local prefix
            prefix="$(basename "$subdir")"
            for f in "$subdir"/*; do
                [[ -f "$f" ]] || continue
                local name
                name="$(basename "$f")"
                cp "$f" "${dir}/${prefix}_${name}"
            done
        done
        find "$dir" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
    }
    flatten_images "$colmap_input" 2>/dev/null || true
    flatten_images "$DATASET_PATH/masks" 2>/dev/null || true

    # Remove images from excluded camera indices (0-3 typically have obstruction)
    for i in 0 1 2 3; do
        rm -f "$colmap_input/pano_camera${i}_"*.jpg 2>/dev/null || true
        rm -f "$colmap_input/pano_camera${i}_"*.png 2>/dev/null || true
    done

    # Check panorama output exists
    if [[ ! -d "$colmap_input" ]]; then
        echo "ERROR: Panorama did not produce output at $colmap_input"
        exit 1
    fi

    echo ""
    echo ">>> Running COLMAP (starting in background)..."
    echo "    Input:  $DATASET_PATH"
    echo "    Output: $colmap_images"
    echo "    Logs:   docker logs -f 3dgs-colmap-pipeline"
    echo ""

    # COLMAP (detached with log following)
    docker rm -f 3dgs-colmap-pipeline 2>/dev/null || true

    docker run -d \
        --name "3dgs-colmap-pipeline" \
        --user "$(get_host_user)" \
        --gpus all \
        --entrypoint "" \
        -v "$DATASET_PATH:/dataset:rw" \
        -v "$(pwd):/workspace:rw" \
        -w /workspace \
        -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
        --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        "$image" \
        bash convert.sh -s /dataset --resize --magick_executable ""

    # Follow logs until completion
    echo "Following COLMAP logs (Ctrl+C to detach, container continues)..."
    docker logs -f 3dgs-colmap-pipeline 2>&1 &
    logs_pid=$!
    trap 'kill $logs_pid 2>/dev/null || true' INT
    
    docker wait 3dgs-colmap-pipeline >/dev/null 2>&1
    exit_code=$?
    kill $logs_pid 2>/dev/null || true
    wait $logs_pid 2>/dev/null || true
    docker rm -f 3dgs-colmap-pipeline >/dev/null 2>&1 || true
    trap - INT

    if [[ "$exit_code" != "0" ]]; then
        echo "ERROR: COLMAP stage failed with exit code $exit_code"
        exit 1
    fi

    # Check colmap output exists
    if [[ ! -d "$colmap_images" ]]; then
        echo "ERROR: COLMAP did not produce output at $colmap_images"
        exit 1
    fi

    echo ""
    echo ">>> Running 3DGS Training (starting in background)..."
    echo "    Input:  $colmap_images"
    echo "    Output: $train_output"
    echo "    Logs:   docker logs -f 3dgs-workspace-train"
    echo ""

    # Training (detached with log following)
    image=$(resolve_image "train")
    docker rm -f 3dgs-workspace-train 2>/dev/null || true

    docker run -d \
        --name "3dgs-workspace-train" \
        --user "$(get_host_user)" \
        --gpus all \
        -v "$DATASET_PATH:/dataset:rw" \
        -v "$train_output:/output:rw" \
        -w /dataset \
        -m "${CONTAINER_MEM:-$DEFAULT_MEM}" \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        "$image" \
        bash -c "python3 \$GS_PATH/train.py -s /dataset -m /output${RESOLUTION:+ -r $RESOLUTION}"

    # Follow logs until completion
    echo "Following Training logs (Ctrl+C to detach, container continues)..."
    docker logs -f 3dgs-workspace-train 2>&1 &
    logs_pid=$!
    trap 'kill $logs_pid 2>/dev/null || true' INT
    
    docker wait 3dgs-workspace-train >/dev/null 2>&1
    exit_code=$?
    kill $logs_pid 2>/dev/null || true
    wait $logs_pid 2>/dev/null || true
    docker rm -f 3dgs-workspace-train >/dev/null 2>&1 || true
    trap - INT

    echo ""
    echo ">>> Pipeline complete!"
    echo ">>> Output in: $train_output"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            colmap|train|panorama|inpaint|pipeline)
                COMMAND="$1"
                shift
                ;;
            --dataset)
                DATASET_PATH="$2"
                shift 2
                ;;
            --output)
                OUTPUT_PATH="$2"
                shift 2
                ;;
            --resolution)
                RESOLUTION="$2"
                shift 2
                ;;
            --image)
                # Override image (advanced)
                shift 2
                ;;
            --local)
                USE_LOCAL="true"
                shift
                ;;
            --pull)
                FORCE_PULL="true"
                shift
                ;;
            --detach)
                DETACHED="true"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use ./run.sh --help for usage"
                exit 1
                ;;
        esac
    done
}

main() {
    # Check for help flag first
    if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_help
        exit 0
    fi

    # Parse command
    COMMAND="$1"
    if [[ ! -v "IMAGES[$COMMAND]" ]]; then
        echo "Unknown command: $COMMAND"
        echo "Valid commands: colmap, train, inpaint, pipeline"
        exit 1
    fi

    shift

    # Parse remaining args
    parse_args "$@"

    # Check docker is available
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker is not installed"
        exit 1
    fi

    # Check for NVIDIA GPU
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Cannot connect to Docker daemon"
        echo "Is Docker running?"
        exit 1
    fi

    # Run the command
    case "$COMMAND" in
        colmap)
            run_colmap
            ;;
        train)
            run_train
            ;;
        panorama)
            run_panorama
            ;;
        inpaint)
            run_inpaint
            ;;
        pipeline)
            run_pipeline
            ;;
    esac
}

main "$@"