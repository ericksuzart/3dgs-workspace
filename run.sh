#!/usr/bin/env bash
set -euo pipefail


# A single entry point for Gaussian Splatting scene reconstruction.
# Supports both 360° panorama and regular photo workflows.
#
# Usage:
#   ./run.sh <command> [options]
#
# Commands:
#   keyframes  Extract keyframes from a video file
#   panorama   Process 360° equirectangular images (panorama/ -> images/)
#   colmap     Run COLMAP SfM on regular photos (input/ -> images/ + sparse/0/)
#   train      Run 3D Gaussian Splatting training (images/ -> output/)
#   inpaint    Interactive 360° image cleanup
#   pipeline   Full pipeline: (panorama|photos) -> train
#
# Options:
#   --dataset PATH        Dataset path (required)
#   --video PATH          Video file for keyframe extraction (required for keyframes)
#   --360                 Mark video as 360° (saves to panorama/, otherwise input/)
#   --mode MODE           Pipeline mode: panorama | photos (required for pipeline)
#   --output PATH         Training output directory (default: dataset/output)
#   --resolution N        Training resolution (1=full, 2=half, default: 1)
#   --yaw-steps N         Virtual cameras per pitch level (default: 8, reduces GPU mem)
#   --remove-bottom N     Remove first N cameras (for tripod/rig, default: =yaw-steps, 0=none)
#   --pull                Force pull latest images from Docker Hub
#   --help                Show this help
#
# Dataset structure:
#   Video:     video.mp4               ->  run keyframes ->  dataset/{panorama|input}/*.jpg
#   Panorama:  dataset/panorama/*.jpg  ->  run panorama  ->  dataset/images/
#   Photos:    dataset/input/*.jpg     ->  run colmap    ->  dataset/images/ + sparse/0/
#   Training:  dataset/images/ + sparse/0/  ->  run train  ->  dataset/output/
#
# Docker images (local-first, pulled from Hub if missing):
#   3dgs-colmap      - COLMAP + Panorama SfM
#   3dgsworkspace    - 3DGS Training
#   lama-inpaint     - Inpainting
# =============================================================================

DOCKER_HUB_ORG="ericksuzart"
DEFAULT_MEM="120g"

# Defaults
COMMAND=""
DATASET_PATH=""
OUTPUT_PATH=""
RESOLUTION="1"
MODE=""
YAW_STEPS=""
BOTTOM_REMOVE=""
FORCE_PULL="false"
VIDEO_PATH=""
IS_360="false"

# Helpers

usage() {
    cat <<EOF
3DGS Workspace - Unified CLI
========================================

Usage:
  ./run.sh <command> [options]

Commands:
  keyframes  Extract keyframes from a video file
  panorama   Process 360° equirectangular images (panorama/ -> images/)
  colmap     Run COLMAP SfM on regular photos (input/ -> images/ + sparse/0/)
  train      Run 3D Gaussian Splatting training (images/ -> output/)
  inpaint    Interactive 360° image cleanup
  pipeline   Full pipeline: (panorama|photos) -> train

Options:
  --dataset PATH        Dataset path (required)
  --video PATH          Video file for keyframe extraction (required for keyframes)
  --360                 Mark video as 360° (saves to panorama/, otherwise input/)
  --mode MODE           Pipeline mode: panorama | photos (required for pipeline)
  --output PATH         Training output directory (default: dataset/output)
  --resolution N        Training resolution (1=full, 2=half, default: 1)
  --yaw-steps N         Virtual cameras per pitch level (default: 8, reduces GPU mem)
  --remove-bottom N     Remove first N cameras (for tripod/rig, default: =yaw-steps, 0=none)
  --pull                Force pull latest images from Docker Hub
  --help                Show this help

Dataset structure:
  Video:     video.mp4               ->  run keyframes ->  dataset/{panorama|input}/*.jpg
  Panorama:  dataset/panorama/*.jpg  ->  run panorama  ->  dataset/images/
  Photos:    dataset/input/*.jpg     ->  run colmap    ->  dataset/images/ + sparse/0/
  Training:  dataset/images/ + sparse/0/  ->  run train  ->  dataset/output/

Examples:
  ./run.sh keyframes --dataset /data/scene --video /path/to/video.mp4
  ./run.sh keyframes --dataset /data/scene --video /path/to/360.mp4 --360
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
        docker pull "$hub_img" >&2
        echo "$hub_img"
        return
    fi

    # Prefer local if available
    if docker image inspect "$local_img" >/dev/null 2>&1; then
        echo "$local_img"
        return
    fi

    # Try pulling from Hub
    echo "Pulling ${hub_img} from Docker Hub..." >&2
    docker pull "$hub_img" >&2
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

cmd_keyframes() {
    ensure_dataset
    [[ -n "$VIDEO_PATH" ]] || die "--video is required for keyframes"
    [[ -f "$VIDEO_PATH" ]] || die "Video file does not exist: $VIDEO_PATH"
    command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is not installed"

    # Determine output directory based on --360 flag
    local out_dir
    if [[ "$IS_360" == "true" ]]; then
        out_dir="$DATASET_PATH/panorama"
    else
        out_dir="$DATASET_PATH/input"
    fi
    mkdir -p "$out_dir"

    # Resolve absolute path for video file
    VIDEO_PATH="$(cd "$(dirname "$VIDEO_PATH")" && pwd)/$(basename "$VIDEO_PATH")"

    echo "=========================================="
    echo "  Keyframe Extraction"
    echo "=========================================="
    echo "  Video:   $VIDEO_PATH"
    echo "  Output:  $out_dir/"
    echo "  360°:    $IS_360"
    echo "=========================================="
    echo ""

    ffmpeg -skip_frame nokey -i "$VIDEO_PATH" -vsync vfr "$out_dir/panorama%04d.jpg"

    local count
    count=$(find "$out_dir" -maxdepth 1 -name "panorama*.jpg" -type f 2>/dev/null | wc -l)
    echo ""
    echo "=========================================="
    echo "  Keyframe extraction complete"
    echo "  Extracted: $count frames"
    echo "  Output:    $out_dir/"
    echo "=========================================="
}

cmd_panorama() {
    ensure_dataset
    [[ -d "$DATASET_PATH/panorama" ]] || die "No panorama/ directory found in $DATASET_PATH"

    local image
    image=$(resolve_image "3dgs-colmap")

    echo "=========================================="
    echo "  Panorama SfM (intra-rig matching)"
    echo "=========================================="
    echo "  Dataset: $DATASET_PATH"
    echo "  Image:   $image"
    echo "=========================================="
    echo ""

    # Build panorama_sfm.py args
    pano_args=(
        --input_image_path /dataset/panorama
        --output_path /dataset
        --matcher vocabtree
        --pano_render_type overlapping
    )
    if [[ -n "$YAW_STEPS" ]]; then
        pano_args+=(--num_steps_yaw "$YAW_STEPS")
    fi

    run_container \
        --entrypoint "" \
        -v "$DATASET_PATH:/dataset:rw" \
        -w /dataset \
        "$image" \
        /usr/local/bin/panorama_sfm.py "${pano_args[@]}"

    # Determine how many cameras to remove (default: yaw steps, i.e. one whole pitch level)
    # With N yaw steps × 3 pitches, the first N cameras (pitch=-35°) capture the tripod/rig.
    local bottom_remove="${BOTTOM_REMOVE:-${YAW_STEPS:-8}}"
    # Clean both possible naming conventions (pano_camera and _camera).
    remove_cams=()
    for ((i=0; i<bottom_remove; i++)); do
        remove_cams+=("pano_camera$i")
        rm -rf "$DATASET_PATH/images/pano_camera${i}" 2>/dev/null || true
        rm -rf "$DATASET_PATH/images/_camera${i}" 2>/dev/null || true
    done

    # Clean the COLMAP sparse model to remove references to the deleted cameras
    if (( bottom_remove > 0 )); then
        IFS=,; clean_prefixes="${remove_cams[*]}"; unset IFS
        docker run --rm \
            --user "$(host_user)" \
            --entrypoint /usr/bin/python3 \
            -v "$DATASET_PATH:/dataset:rw" \
            -w /dataset \
            "$image" \
            /usr/local/bin/clean_colmap_sparse.py /dataset "$clean_prefixes"
    fi

    # Verify sparse reconstruction succeeded
    if [[ ! -f "$DATASET_PATH/sparse/0/images.bin" ]] && [[ ! -f "$DATASET_PATH/sparse/0/images.txt" ]]; then
        die "Panorama SfM failed to create a sparse model. Try with more panorama images (at least 3 with good overlap)."
    fi

    # Resize images (same as before)
    cmd_resize "$image"

    local count
    count=$(find "$DATASET_PATH/images" -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo ""
    echo "=========================================="
    echo "  Panorama complete (intra-rig enabled)"
    echo "  Cameras: $((count - 1)) -> $DATASET_PATH/images/"
    echo "  Sparse:  $DATASET_PATH/sparse/"
    echo "=========================================="
}


cmd_resize() {
    local image="$1"

    echo ""
    echo "Resizing images (50%, 25%, 12.5%)..."
    echo "  Running inside Docker container..."

    # shellcheck disable=SC2016
    run_container \
        --entrypoint "" \
        -v "$DATASET_PATH:/dataset:rw" \
        -w /dataset \
        "$image" \
        bash -c '
        set -euo pipefail

        IMAGES_DIR="/dataset/images"
        TOTAL_FILES=$(find "$IMAGES_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)
        job_pids=()

        if [[ $TOTAL_FILES -eq 0 ]]; then
            echo "  No images found to resize."
            exit 0
        fi

        echo "  Found $TOTAL_FILES images to resize."

        MAX_JOBS=$(nproc 2>/dev/null || echo 4)
        PROCESSED_COUNT=0

        # Process each camera subdirectory
        for camera_dir in "$IMAGES_DIR"/pano_camera*/; do
            if [[ ! -d "$camera_dir" ]]; then
                continue
            fi

            camera_name=$(basename "$camera_dir")
            echo "  Processing $camera_name..."

            # Create corresponding directories
            mkdir -p "/dataset/images_2/$camera_name"
            mkdir -p "/dataset/images_4/$camera_name"
            mkdir -p "/dataset/images_8/$camera_name"

            # Process images in this camera directory
            while IFS= read -r -d "" source_file; do
                file=$(basename "$source_file")

                # Process in background
                (
                    # 50% resize (images_2)
                    cp "$source_file" "/dataset/images_2/$camera_name/$file"
                    mogrify -resize 50% "/dataset/images_2/$camera_name/$file"

                    # 25% resize (images_4)
                    cp "$source_file" "/dataset/images_4/$camera_name/$file"
                    mogrify -resize 25% "/dataset/images_4/$camera_name/$file"

                    # 12.5% resize (images_8)
                    cp "$source_file" "/dataset/images_8/$camera_name/$file"
                    mogrify -resize 12.5% "/dataset/images_8/$camera_name/$file"
                ) &
                job_pids+=($!)

                # Limit parallel jobs
                if ((${#job_pids[@]} >= MAX_JOBS)); then
                    wait "${job_pids[0]}"
                    job_pids=("${job_pids[@]:1}")
                fi

                # Update progress
                PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
                percentage=$(( (PROCESSED_COUNT * 100) / TOTAL_FILES ))
                bar_length=40
                filled_length=$(( (PROCESSED_COUNT * bar_length) / TOTAL_FILES ))
                bar=$(printf "%0.s#" $(seq 1 $filled_length))
                empty=$(printf "%0.s-" $(seq 1 $((bar_length - filled_length))))
                printf "    Progress: [%s%s] %d%% (%d/%d) \r" "$bar" "$empty" "$percentage" "$PROCESSED_COUNT" "$TOTAL_FILES"
            done < <(find "$camera_dir" -maxdepth 1 -type f -print0)
        done

        # Wait for remaining jobs
        for pid in "${job_pids[@]}"; do
            wait "$pid"
        done

        printf "\n"
        echo "  Image resizing complete."
        '
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
            --video)
                [[ $# -ge 2 ]] || die "--video requires a value"
                VIDEO_PATH="$2"; shift 2 ;;
            --360)
                IS_360="true"; shift ;;
            --mode)
                [[ $# -ge 2 ]] || die "--mode requires a value (panorama|photos)"
                MODE="$2"; shift 2 ;;
            --output)
                [[ $# -ge 2 ]] || die "--output requires a value"
                OUTPUT_PATH="$2"; shift 2 ;;
            --resolution)
                [[ $# -ge 2 ]] || die "--resolution requires a value"
                RESOLUTION="$2"; shift 2 ;;
            --yaw-steps)
                [[ $# -ge 2 ]] || die "--yaw-steps requires a value"
                YAW_STEPS="$2"; shift 2 ;;
            --remove-bottom)
                [[ $# -ge 2 ]] || die "--remove-bottom requires a value"
                BOTTOM_REMOVE="$2"; shift 2 ;;
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
    keyframes|panorama|colmap|train|inpaint|pipeline|--help|-h)
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
    keyframes) cmd_keyframes ;;
    panorama) cmd_panorama ;;
    colmap)   cmd_colmap ;;
    train)    cmd_train ;;
    inpaint)  cmd_inpaint ;;
    pipeline) cmd_pipeline ;;
esac
