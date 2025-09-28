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

1. **Clone and setup:**

   ```bash
   git clone https://github.com/ericksuzart/3dgs-workspace.git
   cd 3dgs-workspace
   export DATASET_PATH=/path/to/your/datasets
   code .
    ```

    > **Note**: Replace `/path/to/your/datasets` with the actual path on your machine where your datasets are stored. This path will be mounted inside the Docker container as `/datasets` by default. You can change this path in the `devcontainer.json` file if needed. You can make this change permanent by adding the export command to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`).
    >
    > ```bash
    > echo 'export DATASET_PATH=/path/to/your/datasets' >> ~/.bashrc
    > source ~/.bashrc
    > ```

2. **Click "Reopen in Container" when prompted**

    <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c516eb43-6d14-48f7-855e-c0859720e62d" />

    If you don't see the popup, you can manually open the command palette and run the command `Dev Containers: Reopen in Container`.
    - Press `F1` to open the command palette.
    - Type `Dev Containers: Reopen in Container` and select it.

    VSCode will build the Docker image and start a container based on the configuration in the `.devcontainer` folder. This may take a few minutes, especially the first time. Make sure you have a stable internet connection, as it will need to download several dependencies.

3. **Verify GPU Access in Container**

    After opening in the container, verify GPU access:

    ```bash
    nvidia-smi
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

4. **Run Training:**

   ```bash
   python3 $GS_PATH/train.py -s $DATASET_PATH/<your-dataset>
   ```

    You can view all available options in the [3D Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting).

    After successful training, you'll find the output in:

    ```bash
    output/
    └── <your-scene-run>/
        ├── point_cloud
        │   ├── iteration_7000
        │   │  └── point_cloud.ply
        │   └── iteration_30000
        │      └── point_cloud.ply
        ├── cameras.json
        └── other output files...
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
