FROM nvcr.io/nvidia/pytorch:22.06-py3

ARG USERNAME=3dgs
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    WORKSPACE=/workspace \
    GS_PATH=/opt/gaussian-splatting

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl nano git libx11-6 libgl1-mesa-glx g++ \
    libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev \
    libxxf86vm-dev colmap imagemagick ffmpeg meshlab && \
    rm -rf /var/lib/apt/lists/*

# Clone Gaussian Splatting to a persistent location
RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git ${GS_PATH}

# Install Python dependencies
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install tqdm plyfile

# Build the submodules
RUN cd ${GS_PATH} && \
    pip install submodules/diff-gaussian-rasterization/ && \
    pip install submodules/simple-knn/

# Create a non-root user to use
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN sudo usermod --append --groups video $USERNAME

CMD ["sleep", "infinity"]
