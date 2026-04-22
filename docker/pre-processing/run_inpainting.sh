#!/usr/bin/env bash
# =============================================================================
# run_inpaint.sh — Build and launch the interactive SAM + LaMa container.
#
# USAGE:
#   DATASET_PATH=/path/to/images ./run_inpaint.sh
#
# MODES:
#   MODE=full     — Interactive masking + inpainting (default)
#   MODE=crop     — Crop equirectangular frames only (rows 128:2128), no inpainting
#   MODE=inpaint  — Interactive masking + inpainting on already-cropped images
#
# OPTIONAL ENV OVERRIDES:
#   MODE             Processing mode                               (default: full)
#   NADIR_FRACTION   (ignored — masking is now interactive via SAM)
#   DILATION_ITER    Mask dilation iterations                      (default: 10)
#   BLUR_KERNEL      Feathering blur kernel size                   (default: 21)
#   JPEG_QUALITY     Output JPEG quality [1-95]                    (default: 95)
#   DISPLAY_SCALE    Display window scale [0.1-1.0]                (default: 0.5)
#                      Lower if your screen is too small for the images.
#                      Insta360 X5 frames are 11008x5504 — 0.25 recommended
#                      for typical 1080p monitors.
#   MODELS_DIR       Host path for SAM weight cache                (default: ./models)
#                      SAM vit_h weights (~2.5 GB) are downloaded here on
#                      first run and reused on all subsequent runs.
#   IMAGE_NAME       Docker image tag                              (default: lama-inpaint:latest)
#   SKIP_BUILD       Set to "1" to skip docker build               (default: unset)
#
# EXAMPLES:
#   # Basic run (full mode: mask + inpaint)
#   DATASET_PATH=~/captures/scene1 ./run_inpaint.sh
#
#   # Crop-only mode (no masking, no inpainting)
#   DATASET_PATH=~/captures/scene1 MODE=crop ./run_inpaint.sh
#
#   # Inpaint-only mode (on already-cropped images)
#   DATASET_PATH=~/captures/scene1/output MODE=inpaint ./run_inpaint.sh
#
#   # Scale window down for a 1080p monitor + custom dilation
#   DATASET_PATH=~/captures/scene1 DISPLAY_SCALE=0.25 DILATION_ITER=15 ./run_inpaint.sh
#
#   # Use a shared models cache in a different location
#   DATASET_PATH=~/captures/scene1 MODELS_DIR=~/sam-models ./run_inpaint.sh
#
#   # Skip rebuild (image already built)
#   SKIP_BUILD=1 DATASET_PATH=~/captures/scene1 ./run_inpaint.sh
# =============================================================================
set -euo pipefail

# Colours
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Config
IMAGE_NAME="${IMAGE_NAME:-lama-inpaint:latest}"
MODE="${MODE:-full}"
DILATION_ITER="${DILATION_ITER:-10}"
BLUR_KERNEL="${BLUR_KERNEL:-21}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
DISPLAY_SCALE="${DISPLAY_SCALE:-0.5}"

# Validate DATASET_PATH
if [[ -z "${DATASET_PATH:-}" ]]; then
    error "DATASET_PATH is not set.\n  Example: DATASET_PATH=/path/to/images ./run_inpaint.sh"
fi
if [[ ! -d "$DATASET_PATH" ]]; then
    error "DATASET_PATH '$DATASET_PATH' does not exist or is not a directory."
fi
DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"

# X11 / Display setup
OS="$(uname -s)"
DOCKER_OPTS=()

if [[ "$OS" == "Darwin" ]]; then
    # macOS — XQuartz
    if [[ -z "${DISPLAY:-}" ]]; then
        export DISPLAY=":0"
        warn "DISPLAY not set — defaulting to :0 (XQuartz). See script header if this fails."
    fi
    # XQuartz uses a different socket path
    XSOCK="/tmp/.X11-unix"
    DOCKER_OPTS+=(-e DISPLAY="host.docker.internal:0")
    # Allow connections from localhost
    xhost +127.0.0.1 2>/dev/null || warn "xhost failed — XQuartz may not be running."
else
    # Linux / WSL
    if [[ -z "${DISPLAY:-}" ]]; then
        error "DISPLAY is not set. Make sure you are running in an X11 session."
    fi
    XSOCK="/tmp/.X11-unix"
    
    # Check if we are using SSH X11 forwarding (usually DISPLAY=localhost:10.0)
    if [[ "$DISPLAY" == *"localhost:"* || "$DISPLAY" == *"127.0.0.1:"* ]]; then
        # Force TCP connection: X11 clients try to use missing Unix sockets for "localhost"
        SAFE_DISPLAY="${DISPLAY/localhost/127.0.0.1}"
        
        info "SSH X11 Forwarding detected. Configuring host network and Xauthority..."
        DOCKER_OPTS+=(
            --network host
            --ipc host
            -e DISPLAY="$SAFE_DISPLAY"
            -e QT_X11_NO_MITSHM=1
            -v "$HOME/.Xauthority:/root/.Xauthority:ro"
            -e XAUTHORITY="/root/.Xauthority"
        )
    else
        DOCKER_OPTS+=(-e DISPLAY="$DISPLAY")
        xhost +local:docker 2>/dev/null \
            || warn "xhost +local:docker failed — the OpenCV window may not open."
    fi
fi

# Build
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    warn "SKIP_BUILD=1 — skipping docker build."
else
    info "Building Docker image '${IMAGE_NAME}' ..."
    docker build \
        --tag  "$IMAGE_NAME" \
        --file "$SCRIPT_DIR/Dockerfile" \
        "$SCRIPT_DIR"
    success "Image built: $IMAGE_NAME"
fi

# GPU detection
GPU_FLAGS=()
if command -v nvidia-smi &>/dev/null \
   && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    info "NVIDIA GPU detected: $GPU_NAME — enabling GPU passthrough."
    GPU_FLAGS=(--gpus all)
else
    warn "No NVIDIA GPU detected (or nvidia-smi unavailable) — running on CPU."
    warn "SAM inference on CPU is slow. Expect ~10-30 s per mask on large images."
fi

# Summary
echo ""
echo -e "${BOLD}${CYAN}┌─ Run configuration ─────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│${NC}  Mode           : $MODE"
echo -e "${CYAN}│${NC}  Dataset path   : $DATASET_PATH"
echo -e "${CYAN}│${NC}  Output folder  : $DATASET_PATH/output/"
echo -e "${CYAN}│${NC}  Display scale  : $DISPLAY_SCALE"
echo -e "${CYAN}│${NC}  Dilation iter  : $DILATION_ITER"
echo -e "${CYAN}│${NC}  Blur kernel    : $BLUR_KERNEL"
echo -e "${CYAN}│${NC}  JPEG quality   : $JPEG_QUALITY"
echo -e "${CYAN}│${NC}  GPU flags      : ${GPU_FLAGS[*]:-none (CPU mode)}"
echo -e "${CYAN}│${NC}  DISPLAY        : ${DISPLAY:-unset}"
echo -e "${CYAN}└─────────────────────────────────────────────────────────────────┘${NC}"
echo ""

# Run
info "Starting container ..."
docker run --rm \
    "${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"}" \
    --volume "${DATASET_PATH}:/data" \
    --volume "${XSOCK}:${XSOCK}" \
    "${DOCKER_OPTS[@]}" \
    --env DATASET_PATH=/data \
    --env MODE="$MODE" \
    --env DILATION_ITER="$DILATION_ITER" \
    --env BLUR_KERNEL="$BLUR_KERNEL" \
    --env JPEG_QUALITY="$JPEG_QUALITY" \
    --env DISPLAY_SCALE="$DISPLAY_SCALE" \
    "$IMAGE_NAME"

success "All done. Results are in: $DATASET_PATH/output/"
