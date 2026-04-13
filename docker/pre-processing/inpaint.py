#!/usr/bin/env python3
"""
Interactive 360° tripod removal: paint-brush masking + nadir-reprojected SD2 inpainting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT CHANGED vs THE ORIGINAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. NADIR REPROJECTION  (biggest quality win)
     Before inpainting, the masked region is extracted as a flat gnomonic
     (perspective) patch centred on the nadir.  LaMa / SD2 were trained on
     regular photographs — inpainting in undistorted space gives coherent
     results.  The patch is then remapped back into equirectangular space
     and blended with a soft feathered mask.

  2. DIFFUSERS SD2 INPAINTING  (GPU path)
     Replaces simple_lama_inpainting with HuggingFace diffusers running
     stabilityai/stable-diffusion-2-inpainting directly (no IOPaint needed).
     A floor-specific prompt guides generation toward the correct texture.
     Automatic fallback to LaMa when diffusers / CUDA is unavailable.

     NOTE: IOPaint's ModelManager only exposes ['lama','cv2'] unless the
     diffusers extra is installed and wired in — so we call diffusers directly,
     which is both simpler and version-stable.

  3. MULTI-PASS INPAINTING
     SD2 (or LaMa) is run INPAINT_PASSES times, progressively eroding the
     mask so later passes refine edges rather than re-hallucinate the centre.

  4. FLOOR PRE-FILL  (gives the model a better starting point)
     The masked region is seeded with real floor texture sampled from just
     above the mask boundary before the model runs, removing the need to
     hallucinate from pure noise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL (additions to original requirements)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pip install diffusers accelerate transformers py360convert
  # diffusers will auto-download stabilityai/stable-diffusion-2-inpainting
  # on first run (~5 GB) — set HF_HOME to control the cache location.
  # iopaint is no longer required (only simple-lama-inpainting as fallback).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW  (unchanged from original)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PHASE 1 — Interactive masking (OpenCV window):
    Paint the tripod with left-click drag.  A = accept, R = reset, S = skip.

  PHASE 2 — Batch inpainting (headless):
    Each image is nadir-reprojected, inpainted with SD2, and blended back.
    Results written to DATASET_PATH/output/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT VARIABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DATASET_PATH      Directory containing source JPG images  (default: /data)
  DILATION_ITER     Mask dilation iterations                 (default: 14)
  BLUR_KERNEL       Feathering Gaussian blur kernel size     (default: 31)
  JPEG_QUALITY      Output JPEG quality [1-95]               (default: 95)
  DISPLAY_SCALE     Window display scale factor [0.1-1.0]    (default: 0.5)
  NADIR_FOV         Gnomonic patch FOV in degrees            (default: 110)
  NADIR_PATCH_SIZE  Nadir patch resolution in pixels         (default: 1024)
  INPAINT_PASSES    Number of inpainting refinement passes   (default: 2)
  SD_STEPS          Stable Diffusion denoising steps         (default: 35)
  SD_STRENGTH       SD inpainting strength [0.0-1.0]         (default: 0.80)
  SD_GUIDANCE       SD classifier-free guidance scale        (default: 7.5)
  FLOOR_PROMPT      Text prompt passed to SD2                (see default below)
  INPAINT_BACKEND   "sd2" | "lama"  (default: "sd2")
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

WINDOW_NAME = "Masking - 360 Tripod Removal"

DEFAULT_FLOOR_PROMPT = (
    "seamless wooden parquet floor, clean hardwood, "
    "natural wood grain, no objects, photorealistic, sharp"
)
DEFAULT_FLOOR_NEG_PROMPT = (
    "tripod, camera, object, shadow, dark spot, person, "
    "blurry, distorted, artifact, watermark"
)

# OpenCV colours (BGR)
CLR_POSITIVE = (57,  255, 20)
CLR_NEGATIVE = (0,   80,  255)
CLR_MASK     = (0,   200, 100)
CLR_WHITE    = (255, 255, 255)
CLR_YELLOW   = (0,   255, 255)
CLR_GRAY     = (160, 160, 160)
CLR_HUD_BG   = (15,  15,  15)

POINT_RADIUS = 9


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    dataset_path:    Path
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
    dataset_path = Path(os.environ.get("DATASET_PATH", "/data"))
    if not dataset_path.exists():
        log.error("DATASET_PATH '%s' does not exist.", dataset_path)
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


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def expand_and_feather(mask: np.ndarray, dilation_iter: int, blur_kernel: int) -> np.ndarray:
    """Dilate then Gaussian-blur the binary mask. Returns float32 in [0, 1]."""
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation_iter)
    blurred = cv2.GaussianBlur(dilated.astype(np.float32), (blur_kernel, blur_kernel), 0)
    return np.clip(blurred, 0.0, 1.0)


def erode_mask(mask_u8: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Erode a uint8 binary mask (used between multi-pass refinement steps)."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(mask_u8, kernel, iterations=iterations)


# ---------------------------------------------------------------------------
# Nadir reprojection helpers
# ---------------------------------------------------------------------------
def build_nadir_inverse_map(
    eq_h: int,
    eq_w: int,
    patch_size: int,
    fov_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the inverse mapping from equirectangular space → nadir gnomonic patch.

    For every pixel (ex, ey) in the equirectangular image this returns the
    corresponding floating-point coordinate (map_x, map_y) in the nadir patch
    so that cv2.remap can sample the inpainted patch back into equirectangular
    space.

    The gnomonic projection is centred at the south pole (lon=0°, lat=-90°),
    i.e. the camera looks straight down.  Pixels above the equator (lat > 0)
    are outside the projection and receive coordinate (-1, -1).

    Returns
    -------
    map_x, map_y : np.ndarray float32, shape (eq_h, eq_w)
        Coordinates into the nadir patch for cv2.remap.
    """
    ey_grid, ex_grid = np.mgrid[0:eq_h, 0:eq_w].astype(np.float32)

    # Equirectangular → spherical angles
    lon = (ex_grid / eq_w) * 2.0 * np.pi - np.pi          # [-π, π]
    lat = np.pi / 2.0 - (ey_grid / eq_h) * np.pi           # [π/2, -π/2]

    # Spherical → 3-D unit vector
    cos_lat = np.cos(lat)
    X =  cos_lat * np.cos(lon)
    Y =  cos_lat * np.sin(lon)
    Z =  np.sin(lat)                                         # negative = below equator

    # Gnomonic projection onto plane z = -1  (camera looks toward -Z)
    # px = X / (-Z),  py = Y / (-Z)   (valid only when Z < 0)
    half_tan = float(np.tan(np.radians(fov_deg / 2.0)))

    with np.errstate(divide="ignore", invalid="ignore"):
        px = X / (-Z)
        py = Y / (-Z)

    # Map gnomonic coords [-half_tan, half_tan] → patch pixel [0, patch_size]
    map_x = ((px / half_tan + 1.0) / 2.0 * patch_size).astype(np.float32)
    map_y = ((py / half_tan + 1.0) / 2.0 * patch_size).astype(np.float32)

    # Pixels above the equator or outside the patch FOV are invalid
    invalid = (Z >= -1e-6) | (px < -half_tan) | (px > half_tan) | \
              (py < -half_tan) | (py > half_tan)
    map_x[invalid] = -1.0
    map_y[invalid] = -1.0

    return map_x, map_y


def extract_nadir_patch(
    equi_rgb: np.ndarray,
    patch_size: int,
    fov_deg: float,
) -> np.ndarray:
    """
    Extract the nadir region of an equirectangular image as a flat gnomonic patch.

    Uses py360convert when available, otherwise falls back to a hand-rolled
    bilinear remap (slower but dependency-free).
    """
    try:
        import py360convert
        # e2p expects HWC float or uint8
        patch = py360convert.e2p(
            equi_rgb,
            fov_deg=(fov_deg, fov_deg),
            u_deg=0,
            v_deg=-90,
            out_hw=(patch_size, patch_size),
            mode="bilinear",
        )
        return patch.astype(np.uint8)
    except ImportError:
        pass

    # --- fallback: manual gnomonic sampling ---
    eq_h, eq_w = equi_rgb.shape[:2]
    map_x, map_y = build_nadir_inverse_map(eq_h, eq_w, patch_size, fov_deg)

    # Invert: we need forward map (patch → equirect) not inverse
    # Build forward map by computing equirect coords for each patch pixel
    pi_grid, pj_grid = np.mgrid[0:patch_size, 0:patch_size].astype(np.float32)
    half_tan = float(np.tan(np.radians(fov_deg / 2.0)))

    gx = (pj_grid / patch_size * 2.0 - 1.0) * half_tan
    gy = (pi_grid / patch_size * 2.0 - 1.0) * half_tan

    # 3-D direction (camera looks toward -Z, up = +X)
    norm = np.sqrt(gx**2 + gy**2 + 1.0)
    X =  gx / norm
    Y =  gy / norm
    Z = -1.0 / norm

    lon = np.arctan2(Y, X)
    lat = np.arcsin(np.clip(Z, -1.0, 1.0))

    # Spherical → equirect pixel
    fwd_x = ((lon + np.pi) / (2.0 * np.pi) * eq_w).astype(np.float32)
    fwd_y = ((np.pi / 2.0 - lat) / np.pi * eq_h).astype(np.float32)

    patch = cv2.remap(
        equi_rgb, fwd_x, fwd_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return patch


def blend_patch_back(
    equi_rgb:  np.ndarray,
    patch_rgb: np.ndarray,
    mask_float: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """
    Remap the inpainted nadir patch back into equirectangular space and blend
    it with the original image using the feathered mask as the blend weight.

    Parameters
    ----------
    equi_rgb   : H×W×3 uint8, original equirectangular frame
    patch_rgb  : P×P×3 uint8, inpainted gnomonic patch
    mask_float : H×W float32 [0,1], feathered blend mask (1 = replace)
    map_x, map_y : H×W float32, inverse map from build_nadir_inverse_map
    """
    # Sample inpainted patch at the precomputed equirect → patch coordinates
    sampled = cv2.remap(
        patch_rgb, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Zero out sampled pixels that mapped to invalid coords
    valid = (map_x >= 0) & (map_y >= 0)            # H×W bool
    blend_weight = (mask_float * valid).astype(np.float32)[:, :, None]

    result = (
        equi_rgb.astype(np.float32) * (1.0 - blend_weight) +
        sampled.astype(np.float32) * blend_weight
    ).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Floor pre-fill  (seed masked region with real floor texture)
# ---------------------------------------------------------------------------
def prefill_mask_with_floor(image_np: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Seed the masked region with real floor texture sampled from the nearest
    clean row just above the mask boundary.

    This gives the inpainting model a coherent starting texture to refine
    instead of hallucinating from scratch, which significantly improves
    texture continuity for repetitive patterns like parquet.
    """
    filled = image_np.copy()
    h = image_np.shape[0]

    mask_rows = np.where(mask_u8.max(axis=1) > 30)[0]
    if len(mask_rows) == 0:
        return image_np

    top_row = int(mask_rows.min())
    sample_start = max(0, top_row - 60)
    sample_end   = max(1, top_row)
    sample_stripe = image_np[sample_start:sample_end, :]  # ~60 rows of clean floor

    if sample_stripe.shape[0] == 0:
        return image_np

    for row in range(top_row, h):
        row_mask = mask_u8[row].astype(np.float32) / 255.0
        if row_mask.max() < 0.01:
            continue
        src_row = sample_stripe[(row - top_row) % sample_stripe.shape[0]]
        blend   = row_mask[:, None]
        filled[row] = np.clip(
            filled[row].astype(np.float32) * (1.0 - blend) +
            src_row.astype(np.float32) * blend,
            0, 255,
        ).astype(np.uint8)

    return filled


# ---------------------------------------------------------------------------
# Inpainting backends
# ---------------------------------------------------------------------------
class SD2InpaintBackend:
    """
    HuggingFace diffusers — stabilityai/stable-diffusion-2-inpainting.

    Uses diffusers directly (no IOPaint dependency).  The model is downloaded
    automatically on first use (~5 GB).  Set HF_HOME to control the cache.

    fp16 weights are used to halve VRAM usage with negligible quality loss.
    Use float32 if you see NaN / black-frame artefacts on older GPUs.

    Requires:  pip install diffusers accelerate transformers
    """

    MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"

    def __init__(self, cfg: Config):
        from diffusers import StableDiffusionInpaintPipeline

        device = torch.device("cuda")
        log.info(
            "Loading diffusers SD2 inpainting model on %s "
            "(first run downloads ~5 GB) ...", device,
        )
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,   # halves VRAM; swap to float32 if NaN artefacts appear
        ).to(device)
        # Reduces peak VRAM by ~30% — safe to remove if you have >= 16 GB
        self._pipe.enable_attention_slicing()
        self._cfg = cfg
        log.info("SD2 inpainting model ready.")

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        # diffusers expects PIL RGB image + PIL L mask at the same resolution
        image_rgb = image_pil.convert("RGB")
        mask_l    = mask_pil.convert("L")
        w, h      = image_rgb.size

        output = self._pipe(
            prompt=self._cfg.floor_prompt,
            negative_prompt=DEFAULT_FLOOR_NEG_PROMPT,
            image=image_rgb,
            mask_image=mask_l,
            height=h,
            width=w,
            num_inference_steps=self._cfg.sd_steps,
            guidance_scale=self._cfg.sd_guidance,
            strength=self._cfg.sd_strength,
        )
        return output.images[0].convert("RGB")


class LamaInpaintBackend:
    """
    IOPaint LaMa with CROP HD strategy — the same approach used by cleanup.pictures.

    WHY NOT simple_lama_inpainting?
    ──────────────────────────────────────────────────────────────────────────
    simple_lama_inpainting internally resizes every input to 512×512 before
    inference, then upscales the result back to the original size.  This
    destroys texture detail and produces the characteristic blurry smear on
    the output.

    WHY IOPaint CROP HD STRATEGY?
    ──────────────────────────────────────────────────────────────────────────
    IOPaint's CROP strategy:
      1. Crops a padded bounding box around the mask at FULL resolution.
      2. Runs LaMa on that crop (LaMa generalises well beyond its 256 px
         training resolution).
      3. Pastes the inpainted crop back into the original image.

    No destructive downscaling — LaMa sees real floor texture at native
    resolution and reconstructs it faithfully.  This is exactly what
    cleanup.pictures does under the hood.

    TUNABLE ENV VARS
    ──────────────────────────────────────────────────────────────────────────
    LAMA_CROP_MARGIN   Extra pixels of context around the mask (default 196).
                       Larger = more surrounding context for LaMa, but slower.
    LAMA_CROP_TRIGGER  Minimum image side length (px) that activates the CROP
                       strategy (default 512).  Below this, ORIGINAL is used.
    """

    def __init__(self):
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy  # noqa: F401

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading IOPaint LaMa model on %s ...", device)
        self._model          = ModelManager(name="lama", device=device)
        self._InpaintRequest = InpaintRequest
        self._HDStrategy     = HDStrategy
        self._crop_margin    = int(os.environ.get("LAMA_CROP_MARGIN",  "196"))
        self._crop_trigger   = int(os.environ.get("LAMA_CROP_TRIGGER", "512"))
        log.info(
            "IOPaint LaMa ready (CROP strategy, margin=%d, trigger=%d).",
            self._crop_margin, self._crop_trigger,
        )

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        image_np = np.array(image_pil.convert("RGB"))
        mask_np  = np.array(mask_pil.convert("L"))

        config = self._InpaintRequest(
            # CROP: extract a padded bounding box around the mask and run LaMa
            # at native resolution on that crop only.  No global downscaling.
            hd_strategy=self._HDStrategy.CROP,
            hd_strategy_crop_margin=self._crop_margin,
            hd_strategy_crop_trigger_size=self._crop_trigger,
        )
        result_np = self._model(image_np, mask_np, config)
        return Image.fromarray(result_np.astype(np.uint8))


class SimpleLamaFallback:
    """
    Last-resort fallback using simple_lama_inpainting.
    Only used when IOPaint itself is not importable.
    Quality is significantly lower due to internal 512-px downscaling.
    """

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        log.warning(
            "Falling back to simple_lama_inpainting.  Results will be lower quality "
            "because this library resizes inputs to 512 px internally.  "
            "Ensure IOPaint is installed:  pip install iopaint"
        )
        self._lama = SimpleLama()

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        return self._lama(image_pil, mask_pil)


def load_backend(cfg: Config):
    """
    Load the best available inpainting backend.

    Priority:
      1. SD2 via diffusers          (best quality, requires GPU + ~5 GB model)
      2. IOPaint LaMa + CROP HD     (cleanup.pictures quality, already installed)
      3. simple_lama_inpainting     (last resort — low quality due to downscaling)
    """
    if cfg.inpaint_backend == "sd2":
        try:
            return SD2InpaintBackend(cfg)
        except Exception as exc:
            log.warning(
                "SD2 backend failed to load (%s) — trying IOPaint LaMa.\n"
                "  Install diffusers:  pip install diffusers accelerate transformers",
                exc,
            )

    # IOPaint LaMa (CROP HD strategy) — the cleanup.pictures approach
    try:
        return LamaInpaintBackend()
    except Exception as exc:
        log.warning(
            "IOPaint LaMa failed to load (%s) — falling back to simple_lama_inpainting.",
            exc,
        )

    return SimpleLamaFallback()


# ---------------------------------------------------------------------------
# Multi-pass inpainting
# ---------------------------------------------------------------------------
def inpaint_multipass(
    backend,
    image_pil:  Image.Image,
    mask_float: np.ndarray,
    passes:     int,
) -> Image.Image:
    """
    Run the inpainting backend ``passes`` times.

    Pass 1  — inpaint using the full feathered mask (fill the gap)
    Pass 2+ — erode the mask so subsequent passes focus on edge blending
               rather than re-hallucinating the already-filled centre.
    """
    mask_u8      = (mask_float * 255).astype(np.uint8)
    current_img  = image_pil
    current_mask = mask_u8.copy()

    for i in range(passes):
        if current_mask.max() == 0:
            log.debug("Multi-pass: mask fully eroded after %d pass(es), stopping.", i)
            break

        mask_pil = Image.fromarray(current_mask, mode="L")
        log.debug("  Inpaint pass %d/%d ...", i + 1, passes)
        current_img = backend(current_img, mask_pil)

        # Erode for the next pass: refine edges, not the filled centre
        if i < passes - 1:
            current_mask = erode_mask(current_mask, iterations=4)

    return current_img


# ---------------------------------------------------------------------------
# OpenCV rendering helpers  (unchanged from original)
# ---------------------------------------------------------------------------
def draw_hud(canvas, img_idx, total, filename, n_queued):
    w      = canvas.shape[1]
    hud_h  = 72
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), CLR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

    lines = [
        (f"  [{img_idx + 1}/{total}] {filename}   |   Queued: {n_queued}", CLR_YELLOW),
        ("  Left-click and drag to paint the mask directly.",               CLR_WHITE),
        ("  A: accept & next   R: reset mask   S: skip image   Q: quit",    CLR_GRAY),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (0, 17 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def apply_mask_overlay(canvas, mask_float, alpha=0.45):
    colored    = np.zeros_like(canvas)
    colored[:] = CLR_MASK
    mask_bool  = mask_float > 0.1
    canvas[mask_bool] = (
        canvas[mask_bool] * (1.0 - alpha) + colored[mask_bool] * alpha
    ).astype(np.uint8)

    mask_u8 = (mask_float * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask_u8 > 30).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, CLR_WHITE, 1)


# ---------------------------------------------------------------------------
# Interactive masking phase  (unchanged from original)
# ---------------------------------------------------------------------------
@dataclass
class MaskingJob:
    img_path: Path
    mask:     np.ndarray   # float32 [0, 1], full resolution


def run_interactive_phase(image_files: list, cfg: Config) -> list:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    jobs = []

    def load_for_display(path):
        full = cv2.imread(str(path))
        if full is None:
            raise IOError(f"cv2.imread returned None for {path}")
        full = full[128:2128, :]
        h, w = full.shape[:2]
        dw, dh = int(w * cfg.display_scale), int(h * cfg.display_scale)
        disp = cv2.resize(full, (dw, dh), interpolation=cv2.INTER_AREA)
        return full, disp

    total   = len(image_files)
    img_idx = 0

    state = {
        "is_drawing": False,
        "last_pt":    None,
        "brush_size": int(25 / cfg.display_scale),
    }
    current_mask = [None]

    def mouse_callback(event, x, y, flags, param):
        if current_mask[0] is None:
            return
        fx = int(x / cfg.display_scale)
        fy = int(y / cfg.display_scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            state["is_drawing"] = True
            state["last_pt"]    = (fx, fy)
            cv2.circle(current_mask[0], (fx, fy), state["brush_size"], 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and state["is_drawing"]:
            if state["last_pt"]:
                cv2.line(current_mask[0], state["last_pt"], (fx, fy),
                         255, thickness=state["brush_size"] * 2)
            cv2.circle(current_mask[0], (fx, fy), state["brush_size"], 255, -1)
            state["last_pt"] = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP:
            state["is_drawing"] = False
            state["last_pt"]    = None

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    try:
        while img_idx < total:
            img_path = image_files[img_idx]
            log.info("Image %d/%d: %s", img_idx + 1, total, img_path.name)

            try:
                full_bgr, display_bgr = load_for_display(img_path)
            except Exception as exc:
                log.error("Cannot load '%s': %s — skipping.", img_path.name, exc)
                img_idx += 1
                continue

            h, w = full_bgr.shape[:2]
            current_mask[0]      = np.zeros((h, w), dtype=np.uint8)
            state["is_drawing"]  = False
            advance = False

            while not advance:
                frame = display_bgr.copy()

                if np.any(current_mask[0]):
                    dh2, dw2 = frame.shape[:2]
                    mask_disp = cv2.resize(current_mask[0], (dw2, dh2),
                                           interpolation=cv2.INTER_NEAREST)
                    apply_mask_overlay(frame, (mask_disp / 255.0).astype(np.float32))

                draw_hud(frame, img_idx, total, img_path.name, len(jobs))
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(20) & 0xFF

                if key == ord("q"):
                    log.info("Q pressed — stopping interactive phase.")
                    return jobs
                elif key == ord("s"):
                    log.info("Skipped: %s", img_path.name)
                    img_idx += 1
                    advance  = True
                elif key == ord("a"):
                    mask_float = current_mask[0].astype(np.float32) / 255.0
                    final_mask = expand_and_feather(
                        mask_float, cfg.dilation_iter, cfg.blur_kernel
                    )
                    jobs.append(MaskingJob(img_path=img_path, mask=final_mask))
                    log.info("Queued %s (%d/%d).", img_path.name, len(jobs), total)
                    img_idx += 1
                    advance  = True
                elif key == ord("r"):
                    current_mask[0] = np.zeros((h, w), dtype=np.uint8)
                    log.info("Reset mask for %s.", img_path.name)

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("Window closed — stopping interactive phase.")
                    return jobs
    finally:
        cv2.destroyAllWindows()

    return jobs


# ---------------------------------------------------------------------------
# Batch inpainting phase  (improved)
# ---------------------------------------------------------------------------
def inpaint_one(
    job:     MaskingJob,
    backend,
    cfg:     Config,
) -> Optional[Image.Image]:
    """
    Full inpainting pipeline for a single image:

      1. Crop equirectangular frame (same as original)
      2. Pre-fill masked region with sampled floor texture
      3. Extract nadir gnomonic patch  (undistorts the floor)
      4. Project mask into patch space
      5. Multi-pass inpainting on the flat patch
      6. Remap inpainted patch back into equirectangular space
      7. Feather-blend result over original
    """
    # 1 — load and crop
    image_pil = Image.open(job.img_path).convert("RGB")
    w_orig, _ = image_pil.size
    image_pil = image_pil.crop((0, 128, w_orig, 2128))
    image_np  = np.array(image_pil)   # H×W×3 uint8 RGB

    mask_float = job.mask              # H×W float32 [0,1]
    mask_u8    = (mask_float * 255).astype(np.uint8)

    eq_h, eq_w = image_np.shape[:2]
    patch_size = cfg.nadir_patch_size
    fov_deg    = cfg.nadir_fov

    # 2 — pre-fill masked region with real floor texture
    log.debug("  Pre-filling mask with floor texture ...")
    image_np = prefill_mask_with_floor(image_np, mask_u8)

    # 3 — extract flat nadir patch from (pre-filled) equirectangular image
    log.debug("  Extracting nadir patch (FOV=%.0f°, size=%dpx) ...", fov_deg, patch_size)
    nadir_patch = extract_nadir_patch(image_np, patch_size, fov_deg)  # P×P×3 uint8

    # 4 — build inverse map (equirect → patch coords) for blending back later,
    #     and project the equirect mask into patch space using the SAME forward
    #     gnomonic sampling used for the image.
    #
    #     BUG NOTE: cv2.remap(mask_float, map_x, map_y) would be WRONG here.
    #     map_x/map_y are shaped (eq_h, eq_w) — they map equirect→patch coords.
    #     Using them as a remap source-lookup produces an equirect-shaped output,
    #     not a patch-shaped one, so the mask always comes out empty.
    #     The correct approach is to run the same forward gnomonic projection on
    #     the mask that we already use for the image (extract_nadir_patch).
    log.debug("  Building nadir inverse map ...")
    map_x, map_y = build_nadir_inverse_map(eq_h, eq_w, patch_size, fov_deg)

    # Project mask into patch space: treat the float mask as a single-channel
    # image and run the same gnomonic extraction used for the RGB image.
    mask_as_rgb      = np.stack([mask_u8, mask_u8, mask_u8], axis=2)  # H×W×3 uint8
    nadir_mask_rgb   = extract_nadir_patch(mask_as_rgb, patch_size, fov_deg)
    nadir_mask_float = nadir_mask_rgb[:, :, 0].astype(np.float32) / 255.0
    nadir_mask_u8    = nadir_mask_rgb[:, :, 0]

    if nadir_mask_u8.max() < 10:
        log.warning("  Mask does not overlap nadir patch — running full-equirect inpaint.")
        result_pil = inpaint_multipass(
            backend,
            Image.fromarray(image_np),
            mask_float,
            cfg.inpaint_passes,
        )
        return result_pil

    # 5 — multi-pass inpainting on the flat undistorted patch
    log.debug("  Inpainting nadir patch (%d pass[es]) ...", cfg.inpaint_passes)
    inpainted_patch = inpaint_multipass(
        backend,
        Image.fromarray(nadir_patch),
        nadir_mask_float,
        cfg.inpaint_passes,
    )
    inpainted_patch_np = np.array(inpainted_patch)

    # 6 & 7 — remap inpainted patch back and feather-blend into equirect frame
    log.debug("  Blending patch back into equirectangular frame ...")
    result_np = blend_patch_back(
        image_np,
        inpainted_patch_np,
        mask_float,
        map_x,
        map_y,
    )

    return Image.fromarray(result_np)


def run_batch_inpainting(jobs: list, cfg: Config) -> None:
    if not jobs:
        log.info("No images queued — nothing to inpaint.")
        return

    output_dir = cfg.dataset_path / "output"
    output_dir.mkdir(exist_ok=True)

    log.info("Loading inpainting backend: %s ...", cfg.inpaint_backend.upper())
    backend = load_backend(cfg)

    log.info("Processing %d image(s) ...", len(jobs))
    errors = []

    for job in tqdm(jobs, desc="Inpainting", unit="img"):
        out_path = output_dir / job.img_path.name
        try:
            result = inpaint_one(job, backend, cfg)
            if result is not None:
                result.save(
                    out_path,
                    format="JPEG",
                    quality=cfg.jpeg_quality,
                    subsampling=0,
                )
        except Exception as exc:
            log.error("Failed to inpaint '%s': %s", job.img_path.name, exc)
            errors.append(job.img_path.name)

    success = len(jobs) - len(errors)
    log.info("Inpainting complete — %d/%d succeeded.", success, len(jobs))
    if errors:
        log.warning("Failed images (%d): %s", len(errors), ", ".join(errors))
    log.info("Results saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()

    images = sorted(
        p for p in cfg.dataset_path.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    )
    if not images:
        log.error("No supported images found in '%s'.", cfg.dataset_path)
        sys.exit(1)
    log.info("Found %d image(s) in %s.", len(images), cfg.dataset_path)

    log.info("=" * 60)
    log.info("PHASE 1 — Interactive masking  (%d images)", len(images))
    log.info("=" * 60)
    jobs = run_interactive_phase(images, cfg)

    if not jobs:
        log.info("No images accepted — exiting without inpainting.")
        return

    log.info("=" * 60)
    log.info("PHASE 2 — Batch inpainting  (%d images queued)", len(jobs))
    log.info("=" * 60)
    run_batch_inpainting(jobs, cfg)


if __name__ == "__main__":
    main()