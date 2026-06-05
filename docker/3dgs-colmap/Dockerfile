ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.6.0

#
# Docker builder stage.
# Builds COLMAP C++ library and pycolmap Python bindings.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG COLMAP_GIT_COMMIT=4.0.4
ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building Ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qt6-base-dev \
        libqt6opengl6-dev \
        libqt6openglwidgets6 \
        qt6-svg-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libmkl-full-dev \
        libopenimageio-dev \
        libopenimageio2.4t64 \
        openimageio-tools \
        libopencv-dev \
        python3-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Clone COLMAP repository.
RUN git clone https://github.com/colmap/colmap.git /colmap-src

RUN cd /colmap-src && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD

# Build and install COLMAP C++ library.
RUN mkdir -p /colmap-src/build && \
    cd /colmap-src/build && \
    cmake .. \
        -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DBLA_VENDOR=Generic \
        -DCOLMAP_OPENMP_ENABLED=OFF && \
    ninja install && \
    rm -rf /colmap-src/build

# Build and install pycolmap Python bindings system-wide.
# PIP_BREAK_SYSTEM_PACKAGES is needed on Ubuntu 24.04+ which enforces PEP 668.
# CMAKE_PREFIX_PATH points pycolmap's CMake to the COLMAP installation.
ENV CMAKE_PREFIX_PATH=/colmap-install
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN cd /colmap-src && pip3 install .

#
# Docker runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# Minimal dependencies to run COLMAP binary and pycolmap.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 \
        libc6 \
        libomp5 \
        libopengl0 \
        libmetis5 \
        libceres4t64 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v6t64 \
        libqt6core6 \
        libqt6gui6 \
        libqt6widgets6 \
        libqt6openglwidgets6 \
        libcurl4 \
        libssl3t64 \
        libmkl-locale \
        libmkl-intel-lp64 \
        libmkl-intel-thread \
        imagemagick \
        libopenimageio2.4t64 \
        libmkl-core \
        python3 \
        python3-pip \
        python3-scipy \
        python3-pil \
        python3-tqdm \
        python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy COLMAP binaries.
COPY --from=builder /colmap-install/ /usr/local/

# Copy ONNX Runtime libraries (installed as apt dependency in builder, needed by pycolmap).
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

# Copy pycolmap system-wide installation.
COPY --from=builder /usr/local/lib/python3.12/dist-packages/pycolmap \
    /usr/local/lib/python3.12/dist-packages/pycolmap/
COPY --from=builder /usr/local/lib/python3.12/dist-packages/pycolmap-*.dist-info \
    /usr/local/lib/python3.12/dist-packages/

# Copy COLMAP Python examples and scripts.
COPY --from=builder /colmap-src/python /colmap/python

# Copy our conversion script.
COPY convert.sh /usr/local/bin/convert.sh
RUN chmod +x /usr/local/bin/convert.sh

# Copy our custom scripts.
COPY clean_colmap_sparse.py /usr/local/bin/clean_colmap_sparse.py
RUN chmod +x /usr/local/bin/clean_colmap_sparse.py
COPY panorama_sfm.py /usr/local/bin/panorama_sfm.py
RUN chmod +x /usr/local/bin/panorama_sfm.py

# User management is handled at runtime via docker run --user in run.sh
WORKDIR /workspace

ENTRYPOINT [ "bash" ]
