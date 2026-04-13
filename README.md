# Devcontainer workspace for 3D Gaussian Splatting

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/ericksuzart/3dgs-workspace)

This workspace provides a dockerized environment for running the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) project. It simplifies the setup process by encapsulating all dependencies and configs within Docker & devcontainer.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **[Docker](https://www.docker.com/get-started):** A platform for developing, shipping, and running applications in containers.
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):** A toolkit for building and running GPU-accelerated Docker containers.
- **[Visual Studio Code](https://code.visualstudio.com/):** An open source-code editor developed by Microsoft.
  - **[Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers):** It allows you to use a Docker container as a development environment.

## System Requirements

- **NVIDIA GPU** with CUDA Compute Capability 7.0+
- **Minimum `8GB` GPU memory**, `24GB+` preferred

## Quick Start

### Option 1: Devcontainer (Recommended)

1. **Clone and setup:**

   ```bash
   git clone https://github.com/ericksuzart/3dgs-workspace.git
   cd 3dgs-workspace
   export DATASET_PATH=/path/to/your/datasets
   code .
   ```

   > **Note**: Replace `/path/to/your/datasets` with the actual path on your machine where your datasets are stored. This path will be mounted inside the Docker container as `/datasets` by default.

2. **Click "Reopen in Container" when prompted**

   If you don't see the popup, run `Dev Containers: Reopen in Container` from the command palette (`F1`).

3. **Verify GPU Access in Container:**

   ```bash
   nvidia-smi
   python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **Run Training:**

   ```bash
   python3 $GS_PATH/train.py -s $DATASET_PATH/<your-dataset>
   ```

### Option 2: Docker Compose

```bash
export DATASET_PATH=/path/to/your/datasets

# Preprocess with COLMAP
docker compose --profile colmap run --rm colmap -s /dataset --resize

# Training (interactive shell)
docker compose --profile training run --rm training

# Or run training directly
docker compose --profile training run --rm training python3 $GS_PATH/train.py -s /dataset
```

### Option 3: Docker Hub Images

Pull and run directly from Docker Hub:

**COLMAP Panorama SfM:**
```bash
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$DATASET_PATH":/dataset \
  -e DATASET_PATH=/dataset \
  --entrypoint python3 \
  ericksuzart/3dgs-colmap:latest \
  /colmap/python/examples/panorama_sfm.py --input_image_path /dataset/panorama --output_path /dataset/output
```

**3DGS Training:**
```bash
docker run --rm -it \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$DATASET_PATH":/dataset \
  -v ./output:/output \
  ericksuzart/3dgsworkspace:latest \
  python3 /opt/gaussian-splatting/train.py -s /dataset
```

> **Note**: The `--user "$(id -u):$(id -g)"` flag ensures output files have correct ownership on your host system.

### Option 4: Standalone Docker

See the [Makefile](Makefile) for common targets, or use the scripts in `docker/*/run-container.sh`.

## Complete Pipeline

A typical 3DGS workflow involves these steps:

### Step 1: COLMAP Preprocessing

Convert raw images into camera poses and a sparse point cloud using COLMAP.

**Input structure:**
```
<dataset>/
└── input/              ← Place your raw images here (.jpg, .png)
```

**Run COLMAP:**
```bash
# Using docker compose
docker compose --profile colmap run --rm colmap -s /dataset --resize

# Or using the script directly
DATASET_PATH=/path/to/dataset docker/3dgs-colmap/run-container.sh

# Or with docker run directly
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$DATASET_PATH":/dataset \
  ericksuzart/3dgs-colmap:latest \
  -c "convert.sh -s /dataset --resize"
```

**Output structure:**
```
<dataset>/
├── images/             ← Undistorted images (use these for training)
├── sparse/0/           ← COLMAP sparse reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── images_2/           ← 50% resized images (if --resize used)
├── images_4/           ← 25% resized images (if --resize used)
└── images_8/           ← 12.5% resized images (if --resize used)
```

**COLMAP options:**
| Flag                         | Description                            | Default       |
| ---------------------------- | -------------------------------------- | ------------- |
| `-s, --source_path`          | Path to dataset (required)             | —             |
| `--resize`                   | Create multi-resolution image pyramids | Off           |
| `--no_gpu`                   | Disable GPU for SIFT                   | GPU enabled   |
| `--skip_matching`            | Skip feature extraction/matching       | Off           |
| `--camera <model>`           | Camera model                           | `OPENCV`      |
| `--colmap_executable <path>` | Custom COLMAP path                     | `colmap`      |
| `--magick_executable <path>` | ImageMagick path (for resize)          | `magick` (v7) |

### Step 2: (Optional) Interactive Inpainting

Remove tripods, people, or unwanted objects from 360° equirectangular images.

```bash
# Using docker compose
docker compose --profile inpaint run --rm inpaint

# Or using the script
DATASET_PATH=/path/to/images ./docker/pre-processing/run_inpainting.sh
```

**Workflow:**
1. **Phase 1 — Interactive masking:** Paint over unwanted objects with your mouse
2. **Phase 2 — Batch inpainting:** SD2/LaMa fills in the masked regions

**Controls:**
- **Left-click + drag:** Paint mask
- **A:** Accept mask & next image
- **R:** Reset mask
- **S:** Skip image
- **Q:** Quit

### Step 3: Training

```bash
# Using docker compose
docker compose --profile training run --rm training python3 $GS_PATH/train.py -s /dataset

# Or with docker run
docker run --rm -it --gpus all --user "$(id -u):$(id -g)" \
  -v "$DATASET_PATH":/dataset -v ./output:/output \
  ericksuzart/3dgsworkspace:latest \
  python3 /opt/gaussian-splatting/train.py -s /dataset

# Or inside the devcontainer
python3 $GS_PATH/train.py -s $DATASET_PATH/<your-dataset>
```

**Output:**
```
output/
└── <your-scene-run>/
    ├── point_cloud/
    │   ├── iteration_7000/
    │   │   └── point_cloud.ply
    │   └── iteration_30000/
    │       └── point_cloud.ply
    ├── cameras.json
    └── ... (training logs, checkpoints)
```

## Makefile Targets

```bash
make build-all        # Build all Docker images
make build-train      # Build training image
make build-colmap     # Build COLMAP image
make build-inpaint    # Build inpainting image

make colmap           # Run COLMAP preprocessing
make train            # Start training shell
make inpaint          # Run interactive inpainting

make lint             # Run shellcheck + ruff
make clean            # Remove build artifacts
```

## Environment Variables

| Variable          | Description                      | Default    |
| ----------------- | -------------------------------- | ---------- |
| `DATASET_PATH`    | Host path to dataset             | —          |
| `OUTPUT_PATH`     | Host path for training output    | `./output` |
| `CONTAINER_MEM`   | Memory limit for containers      | `12g`      |
| `DISPLAY_SCALE`   | Inpainting window scale          | `0.5`      |
| `DILATION_ITER`   | Mask dilation iterations         | `10`       |
| `INPAINT_BACKEND` | Inpainting engine (`sd2`/`lama`) | `sd2`      |

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
