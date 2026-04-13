# =============================================================================
# Base: PyTorch 2.1.2 + CUDA 12.1 + cuDNN 8 runtime.
# Works on CPU too — CUDA is auto-detected at runtime.
# =============================================================================
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL description="Interactive 360 tripod removal: SAM point-click masking + LaMa inpainting"

# =============================================================================
# System dependencies
#   - libgl1, libglib2, libsm6, libxext6, libxrender1 : OpenCV core
#   - libgtk-3-0, libx11-6, libxcb1                   : OpenCV HighGUI / X11 window
#   - libdbus-1-3                                      : GTK runtime dependency
#   - wget                                             : utility
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk-3-0 \
    libx11-6 \
    libxcb1 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python dependencies
#   Installed from pinned requirements.txt for reproducibility.
#   See requirements.txt for individual package descriptions.
# =============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

    # =============================================================================
# Pre-download LaMa model weights into the image (~200 MB).
# Stored in /root/.cache — no extra volume needed for LaMa.
#
# SAM vit_h weights (~2.5 GB) are intentionally NOT baked in.
# They are downloaded at first run into the /models volume so they
# persist across container restarts without inflating the image size.
# =============================================================================
RUN python - <<'PYEOF'
from simple_lama_inpainting import SimpleLama
_ = SimpleLama()
print("LaMa weights cached successfully.")
PYEOF

# =============================================================================
# Application
# =============================================================================
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

WORKDIR /app

# =============================================================================
# Default environment variables (all overridable at runtime via -e flags)
# =============================================================================
ENV DATASET_PATH=/data
ENV MODE=full
ENV DILATION_ITER=10
ENV BLUR_KERNEL=21
ENV JPEG_QUALITY=95
ENV DISPLAY_SCALE=0.5

# /data   — source images (read) + output/ subfolder (written)
VOLUME ["/data"]

ENTRYPOINT ["inpaint360"]
