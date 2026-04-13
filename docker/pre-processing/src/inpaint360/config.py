"""Configuration management for inpaint360."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_FLOOR_PROMPT = (
    "seamless wooden parquet floor, clean hardwood, "
    "natural wood grain, no objects, photorealistic, sharp"
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

# Processing modes:
#   "full"    — interactive masking + inpainting (default)
#   "crop"    — crop equirectangular frames only (128:2128), no inpainting
#   "inpaint" — interactive masking + inpainting on already-cropped images
VALID_MODES = ("full", "crop", "inpaint")


@dataclass
class Config:
    """Runtime configuration."""
    dataset_path:    Path
    mode:            str   # "full" | "crop" | "inpaint"
    dilation_iter:   int
    blur_kernel:     int
    jpeg_quality:    int
    display_scale:   float
    nadir_fov:       float
    nadir_patch_size: int
    inpaint_passes:  int
    sd_steps:        int
    sd_strength:     float
    sd_guidance:     float
    floor_prompt:    str
    inpaint_backend: str   # "sd2" | "lama"


def load_config() -> Config:
    """Load configuration from environment variables."""
    dataset_path = Path(os.environ.get("DATASET_PATH", "/data"))
    if not dataset_path.exists():
        log.error("DATASET_PATH '%s' does not exist.", dataset_path)
        sys.exit(1)

    mode = os.environ.get("MODE", "full").lower()
    if mode not in VALID_MODES:
        log.error("Invalid MODE '%s'. Choose from: %s", mode, ", ".join(VALID_MODES))
        sys.exit(1)

    blur_kernel = int(os.environ.get("BLUR_KERNEL", "31"))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    display_scale = float(os.environ.get("DISPLAY_SCALE", "0.5"))
    display_scale = max(0.05, min(1.0, display_scale))

    backend = os.environ.get("INPAINT_BACKEND", "sd2").lower()
    if backend not in ("sd2", "lama"):
        log.warning("Unknown INPAINT_BACKEND '%s', defaulting to 'sd2'.", backend)
        backend = "sd2"

    return Config(
        dataset_path=dataset_path,
        mode=mode,
        dilation_iter=int(os.environ.get("DILATION_ITER", "14")),
        blur_kernel=blur_kernel,
        jpeg_quality=int(os.environ.get("JPEG_QUALITY", "95")),
        display_scale=display_scale,
        nadir_fov=float(os.environ.get("NADIR_FOV", "110")),
        nadir_patch_size=int(os.environ.get("NADIR_PATCH_SIZE", "1024")),
        inpaint_passes=int(os.environ.get("INPAINT_PASSES", "2")),
        sd_steps=int(os.environ.get("SD_STEPS", "35")),
        sd_strength=float(os.environ.get("SD_STRENGTH", "0.80")),
        sd_guidance=float(os.environ.get("SD_GUIDANCE", "7.5")),
        floor_prompt=os.environ.get("FLOOR_PROMPT", DEFAULT_FLOOR_PROMPT),
        inpaint_backend=backend,
    )
