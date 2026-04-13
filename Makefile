# =============================================================================
# 3DGS Workspace — Makefile
#
# USAGE:
#   make build-all        Build all Docker images
#   make build-train      Build training image
#   make build-colmap     Build COLMAP image
#   make build-inpaint    Build inpainting image
#
#   make colmap           Run COLMAP preprocessing
#   make train            Start 3DGS training
#   make inpaint          Run interactive inpainting
#
#   make pipeline         Run panorama -> colmap -> train (full workflow)
#   make pipeline-panorama   Run panorama stage only
#   make pipeline-colmap     Run colmap stage only
#   make pipeline-train      Run training stage only
#
#   make stop             Stop all running containers
#   make logs             View logs from running containers
#
#   make lint             Run shellcheck + ruff
#   make clean            Remove build artifacts
#
# REQUIRED ENVIRONMENT:
#   DATASET_PATH=/path/to/your/datasets   (for run targets)
#
# DETACHED MODE (default: true):
#   Containers run in detached mode by default. Your terminal shows logs
#   but Ctrl+C detaches without stopping the container. Use DETACHED=false
#   for interactive mode. See PIPELINE.md for details.
# =============================================================================

.PHONY: build-all build-train build-colmap build-inpaint \
		colmap train inpaint crop panorama \
		pipeline \
		stop stop-colmap stop-panorama stop-train \
		logs logs-colmap logs-panorama logs-train \
		lint lint-sh lint-py \
		clean help

# Configuration
DOCKER      := docker
COMPOSE     := docker compose
DATASET     ?= $(error DATASET_PATH is required. Usage: make colmap DATASET_PATH=/path/to/data)
OUTPUT_PATH ?= ./output
CONTAINER_MEM ?= 12g
MATCHER       ?= sequential
PANO_RENDER_TYPE ?= overlapping
DETACHED      ?= true

# Build targets
build-all: build-train build-colmap build-inpaint

build-train:
	@echo "Building 3DGS training image..."
	$(DOCKER) build -t 3dgsworkspace:latest -f docker/3dgsworkspace/Dockerfile docker/3dgsworkspace

build-colmap:
	@echo "Building COLMAP image..."
	$(DOCKER) build -t 3dgs-colmap:latest -f docker/3dgs-colmap/Dockerfile docker/3dgs-colmap

build-inpaint:
	@echo "Building inpainting image..."
	$(DOCKER) build -t lama-inpaint:latest -f docker/pre-processing/Dockerfile docker/pre-processing

# Run targets
colmap:
	@if [ ! -d "$(DATASET_PATH)/input" ]; then \
		echo "ERROR: $(DATASET_PATH)/input does not exist."; \
		echo "Place raw images in $(DATASET_PATH)/input/ first."; \
		exit 1; \
	fi
	@echo "Running COLMAP preprocessing..."
	DATASET_PATH="$(DATASET_PATH)" \
	CONTAINER_MEM="$(CONTAINER_MEM)" \
	DETACHED="$(DETACHED)" \
	./docker/3dgs-colmap/run-container.sh

train:
	@if [ ! -d "$(DATASET_PATH)/images" ]; then \
		echo "ERROR: $(DATASET_PATH)/images does not exist."; \
		echo "Run 'make colmap' or 'make panorama' first."; \
		exit 1; \
	fi
	@echo "Starting 3D Gaussian Splatting training..."
	DATASET_PATH="$(DATASET_PATH)" \
	OUTPUT_PATH="$(OUTPUT_PATH)" \
	CONTAINER_MEM="$(CONTAINER_MEM)" \
	DETACHED="$(DETACHED)" \
	./docker/3dgsworkspace/run-container.sh

inpaint:
	@echo "Starting interactive inpainting..."
	$(COMPOSE) run --rm inpaint

crop:
	@echo "Cropping equirectangular frames..."
	$(COMPOSE) run --rm -e MODE=crop inpaint

panorama:
	@if [ ! -d "$(DATASET_PATH)" ]; then \
		echo "ERROR: $(DATASET_PATH) does not exist."; \
		exit 1; \
	fi
	@echo "Running COLMAP Panorama SfM..."
	DATASET_PATH="$(DATASET_PATH)" \
	MATCHER="$(MATCHER)" \
	PANO_RENDER_TYPE="$(PANO_RENDER_TYPE)" \
	CONTAINER_MEM="$(CONTAINER_MEM)" \
	DETACHED="$(DETACHED)" \
	./docker/3dgs-colmap/run-panorama-sfm.sh

# Full pipeline: panorama -> colmap -> train
# This target runs all stages sequentially, ensuring output from one
# stage matches the input requirements of the next.
#
# Panorama output structure:
#   $(DATASET_PATH)/output/images/    ← Rendered perspective images
#   $(DATASET_PATH)/output/masks/     ← Generated masks
#   $(DATASET_PATH)/output/database.db
#   $(DATASET_PATH)/output/sparse/
#
# After flattening, images are in:
#   $(DATASET_PATH)/output/images/    ← Flattened image files
#
# Colmap expects:
#   $(DATASET_PATH)/input/            ← Raw images (for standard colmap)
#   OR uses panorama output directly
#
# Training expects:
#   $(DATASET_PATH)/images/           ← Undistorted images
#   $(DATASET_PATH)/sparse/0/         ← COLMAP sparse model

pipeline: pipeline-panorama pipeline-colmap pipeline-train

pipeline-panorama:
	@if [ ! -d "$(DATASET_PATH)" ]; then \
		echo "ERROR: $(DATASET_PATH) does not exist."; \
		exit 1; \
	fi
	@if [ -z "$$(ls -A "$(DATASET_PATH)"/*.jpg "$(DATASET_PATH)"/*.png 2>/dev/null)" ] && \
	    [ -z "$$(ls -Ad "$(DATASET_PATH)"/* 2>/dev/null | head -1)" ]; then \
		echo "ERROR: $(DATASET_PATH) is empty or contains no images."; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "  Pipeline Stage 1: Panorama SfM"
	@echo "=========================================="
	DATASET_PATH="$(DATASET_PATH)" \
	MATCHER="$(MATCHER)" \
	PANO_RENDER_TYPE="$(PANO_RENDER_TYPE)" \
	CONTAINER_MEM="$(CONTAINER_MEM)" \
	DETACHED="$(DETACHED)" \
	./docker/3dgs-colmap/run-panorama-sfm.sh
	@echo ""
	@echo "Panorama SfM complete. Output in $(DATASET_PATH)/output/"
	@echo ""

pipeline-colmap:
	@if [ ! -d "$(DATASET_PATH)/output/images" ]; then \
		echo "ERROR: $(DATASET_PATH)/output/images does not exist."; \
		echo "Run 'make pipeline-panorama' first."; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "  Pipeline Stage 2: COLMAP Processing"
	@echo "=========================================="
	@# Create input symlink for colmap if needed
	@# Panorama outputs to output/images/, colmap reads from input/
	@if [ -L "$(DATASET_PATH)/input" ] || [ -d "$(DATASET_PATH)/input" ]; then \
		echo "Removing existing input/ directory or symlink..."; \
		rm -rf "$(DATASET_PATH)/input"; \
	fi
	@echo "Creating symlink: input/ -> output/images/"
	@ln -s output/images "$(DATASET_PATH)/input"
	@# Now run colmap which will read from input/ (symlinked to output/images/)
	DATASET_PATH="$(DATASET_PATH)" \
	CONTAINER_MEM="$(CONTAINER_MEM)" \
	DETACHED="$(DETACHED)" \
	./docker/3dgs-colmap/run-container.sh
	@echo ""
	@echo "COLMAP processing complete."
	@echo ""

pipeline-train:
	@# Training expects images/ and sparse/0/
	@# After colmap, we have images/ (from colmap undistortion) and sparse/0/
	@if [ ! -d "$(DATASET_PATH)/images" ]; then \
		echo "ERROR: $(DATASET_PATH)/images does not exist."; \
		echo "Run 'make pipeline-colmap' first."; \
		exit 1; \
	fi
	@if [ ! -d "$(DATASET_PATH)/sparse/0" ]; then \
		echo "ERROR: $(DATASET_PATH)/sparse/0/ does not exist."; \
		echo "Run 'make pipeline-colmap' first."; \
		exit 1; \
	fi
	@echo "=========================================="
	@echo "  Pipeline Stage 3: 3DGS Training"
	@echo "=========================================="
	DATASET_PATH="$(DATASET_PATH)" \
	OUTPUT_PATH="$(OUTPUT_PATH)" \
	CONTAINER_MEM="$(CONTAINER_MEM)" \
	DETACHED="$(DETACHED)" \
	./docker/3dgsworkspace/run-container.sh
	@echo ""
	@echo "Training complete. Output in $(OUTPUT_PATH)/"
	@echo ""

# Stop detached containers
stop: stop-colmap stop-panorama stop-train

stop-colmap:
	@echo "Stopping COLMAP container..."
	@$(DOCKER) stop 3dgs-colmap 2>/dev/null && echo "Stopped." || echo "Not running."

stop-panorama:
	@echo "Stopping Panorama container..."
	@$(DOCKER) stop 3dgs-colmap-panorama 2>/dev/null && echo "Stopped." || echo "Not running."

stop-train:
	@echo "Stopping Training container..."
	@$(DOCKER) stop 3dgs-workspace-train 2>/dev/null && echo "Stopped." || echo "Not running."

# View logs from detached containers
logs: logs-colmap logs-panorama logs-train

logs-colmap:
	@echo "=== COLMAP Logs ==="
	@$(DOCKER) logs -f 3dgs-colmap 2>/dev/null || echo "Container not running."

logs-panorama:
	@echo "=== Panorama Logs ==="
	@$(DOCKER) logs -f 3dgs-colmap-panorama 2>/dev/null || echo "Container not running."

logs-train:
	@echo "=== Training Logs ==="
	@$(DOCKER) logs -f 3dgs-workspace-train 2>/dev/null || echo "Container not running."

# Linting
lint: lint-sh lint-py

lint-sh:
	@echo "Running shellcheck..."
	@command -v shellcheck >/dev/null 2>&1 || { echo "Install shellcheck: https://github.com/koalaman/shellcheck#installing"; exit 1; }
	shellcheck docker/build_and_push.sh \
		docker/3dgs-colmap/run-container.sh \
		docker/3dgs-colmap/convert.sh \
		docker/3dgsworkspace/run-container.sh \
		docker/pre-processing/run_inpainting.sh

lint-py:
	@echo "Running ruff..."
	@command -v ruff >/dev/null 2>&1 || { echo "Install ruff: pip install ruff"; exit 1; }
	ruff check docker/pre-processing/src/

# Cleanup
clean:
	@echo "Removing build artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

# Help
help:
	@echo "3DGS Workspace — Available targets:"
	@echo ""
	@echo "  Build:"
	@echo "    build-all       Build all Docker images"
	@echo "    build-train     Build training image only"
	@echo "    build-colmap    Build COLMAP image only"
	@echo "    build-inpaint   Build inpainting image only"
	@echo ""
	@echo "  Run (requires DATASET_PATH):"
	@echo "    colmap          Run COLMAP preprocessing (DATASET_PATH/input/ must exist)"
	@echo "    panorama        Run Panorama SfM (DATASET_PATH with 360° images)"
	@echo "    crop            Crop equirectangular frames only (no inpainting)"
	@echo "    train           Start 3DGS training (DATASET_PATH with images/ + sparse/0/)"
	@echo "    inpaint         Run interactive inpainting (masking + inpainting)"
	@echo ""
	@echo "  Pipeline (full workflow):"
	@echo "    pipeline        Run panorama -> colmap -> train sequentially"
	@echo "    pipeline-panorama   Run only panorama stage"
	@echo "    pipeline-colmap     Run only colmap stage (after panorama)"
	@echo "    pipeline-train      Run only training stage (after colmap)"
	@echo ""
	@echo "  Container Management (detached mode):"
	@echo "    stop            Stop all running containers"
	@echo "    stop-colmap     Stop COLMAP container"
	@echo "    stop-panorama   Stop Panorama container"
	@echo "    stop-train      Stop Training container"
	@echo "    logs            View logs from all containers"
	@echo "    logs-colmap     View COLMAP container logs"
	@echo "    logs-panorama   View Panorama container logs"
	@echo "    logs-train      View Training container logs"
	@echo ""
	@echo "  Quality:"
	@echo "    lint            Run all linters (shellcheck + ruff)"
	@echo "    lint-sh         Run shellcheck only"
	@echo "    lint-py         Run ruff only"
	@echo ""
	@echo "  Other:"
	@echo "    clean           Remove build artifacts"
	@echo "    help            Show this message"
	@echo ""
	@echo "  Environment Variables:"
	@echo "    DATASET_PATH    Path to dataset directory (required)"
	@echo "    OUTPUT_PATH     Path for training output (default: ./output)"
	@echo "    CONTAINER_MEM   Memory limit for containers (default: 12g)"
	@echo "    MATCHER         Feature matching strategy (default: sequential)"
	@echo "    PANO_RENDER_TYPE  Panorama render type (default: overlapping)"
	@echo "    DETACHED        Run in detached mode (default: true)"
