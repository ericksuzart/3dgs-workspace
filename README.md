# 3DGS Workspace

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/ericksuzart/3dgs-workspace)

This workspace provides a Dockerized environment for running the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) project. It simplifies the setup process by encapsulating all dependencies and configuration within Docker.

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

### Option 1: Unified CLI (Recommended)

The `run.sh` script provides a single entry point for all operations.

1. **Clone and setup:**

   ```bash
   git clone https://github.com/ericksuzart/3dgs-workspace.git
   cd 3dgs-workspace
   ```

2. **Run a command:**

   ```bash
   # Process 360° panorama images
   ./run.sh panorama --dataset /path/to/scene

   # Run COLMAP on regular photos
   ./run.sh colmap --dataset /path/to/scene

   # Train a 3DGS model
   ./run.sh train --dataset /path/to/scene

   # Full pipeline: panorama → train
   ./run.sh pipeline --dataset /path/to/scene --mode panorama

   # Full pipeline: photos → colmap → train
   ./run.sh pipeline --dataset /path/to/scene --mode photos

   # Interactive inpainting (remove tripods, unwanted objects)
   ./run.sh inpaint --dataset /path/to/360

   # Help
   ./run.sh --help
   ```

   > **Note**: Replace `/path/to/scene` with the actual path on your machine where your datasets are stored.

3. **Run in background (optional):**

   ```bash
   screen -dmS 3dgs ./run.sh pipeline --dataset /path/to/scene --mode panorama
   ```

### Option 2: Devcontainer

1. **Clone and set dataset path:**

   ```bash
   git clone https://github.com/ericksuzart/3dgs-workspace.git
   cd 3dgs-workspace
   export DATASET_PATH=/path/to/your/datasets
   code .
   ```

   > **Note**: Replace `/path/to/your/datasets` with the actual path on your machine where your datasets are stored. This path will be mounted inside the Docker container as `/dataset`.

2. **Click "Reopen in Container" when prompted**

   If you don't see the popup, run `Dev Containers: Reopen in Container` from the command palette (`F1`).

3. **Verify GPU Access in Container:**

   ```bash
   nvidia-smi
   python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **Run Training:**

   ```bash
   python3 $GS_PATH/train.py -s /dataset
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
  /colmap/python/examples/panorama_sfm.py --input_image_path /dataset/panorama --output_path /dataset
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

## Complete Pipeline

A typical 3DGS workflow involves these steps:

### Step 1: Process Images

Choose one approach based on your image source:

**360° Panorama Images:**

```
<dataset>/
└── panorama/             ← Place your equirectangular images here (.jpg, .png)
```

```bash
./run.sh panorama --dataset /path/to/scene
```

Output: `images/` + `sparse/` — ready for training.

**Regular Photos:**

```
<dataset>/
└── input/                ← Place your raw images here (.jpg, .png)
```

```bash
./run.sh colmap --dataset /path/to/scene
```

Output: `images/` + `sparse/0/` — ready for training.

### Step 2: (Optional) Interactive Inpainting

Remove tripods, people, or unwanted objects from 360° equirectangular images.

```bash
./run.sh inpaint --dataset /path/to/360
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
# Using the unified CLI
./run.sh train --dataset /path/to/scene

# With half resolution
./run.sh train --dataset /path/to/scene --resolution 2

# Or inside the devcontainer
python3 $GS_PATH/train.py -s /dataset
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

## run.sh Reference

| Command    | Description                               | Required Inputs                       |
| ---------- | ----------------------------------------- | ------------------------------------- |
| `panorama` | Process 360° equirectangular images       | `--dataset` + `panorama/`             |
| `colmap`   | Run COLMAP SfM on regular photos          | `--dataset` + `input/`                |
| `train`    | Run 3D Gaussian Splatting training        | `--dataset` + `images/` + `sparse/0/` |
| `inpaint`  | Interactive 360° image cleanup            | `--dataset`                           |
| `pipeline` | Full pipeline: (panorama\|photos) → train | `--dataset` + `--mode`                |

**Options:**

| Flag             | Description                                     | Default          |
| ---------------- | ----------------------------------------------- | ---------------- |
| `--dataset PATH` | Dataset path (required)                         | —                |
| `--mode MODE`    | Pipeline mode: `panorama` \| `photos`           | —                |
| `--output PATH`  | Training output directory                       | `dataset/output` |
| `--resolution N` | Training resolution (1=full, 2=half, 4=quarter) | `4`              |
| `--pull`         | Force pull latest images from Docker Hub        | Off              |
| `--help`         | Show help                                       | —                |

## Environment Variables

| Variable          | Description                      | Default                |
| ----------------- | -------------------------------- | ---------------------- |
| `DATASET_PATH`    | Host path to dataset             | (required)             |
| `OUTPUT_PATH`     | Host path for training output    | `$DATASET_PATH/output` |
| `CONTAINER_MEM`   | Memory limit for containers      | `15g`                  |
| `DISPLAY_SCALE`   | Inpainting window scale          | `0.5`                  |
| `DILATION_ITER`   | Mask dilation iterations         | `10`                   |
| `INPAINT_BACKEND` | Inpainting engine (`sd2`/`lama`) | `sd2`                  |

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
