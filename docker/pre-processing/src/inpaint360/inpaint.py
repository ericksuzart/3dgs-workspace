"""Floor pre-fill and multi-pass inpainting logic."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .backends import InpaintBackend

log = logging.getLogger(__name__)


def prefill_mask_with_floor(image_np: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Seed the masked region with real floor texture sampled from the nearest
    clean row just above the mask boundary.
    """
    filled = image_np.copy()
    h = image_np.shape[0]

    mask_rows = np.where(mask_u8.max(axis=1) > 30)[0]
    if len(mask_rows) == 0:
        return image_np

    top_row = int(mask_rows.min())
    sample_start = max(0, top_row - 60)
    sample_end = max(1, top_row)
    sample_stripe = image_np[sample_start:sample_end, :]

    if sample_stripe.shape[0] == 0:
        return image_np

    for row in range(top_row, h):
        row_mask = mask_u8[row].astype(np.float32) / 255.0
        if row_mask.max() < 0.01:
            continue
        src_row = sample_stripe[(row - top_row) % sample_stripe.shape[0]]
        blend = row_mask[:, None]
        filled[row] = np.clip(
            filled[row].astype(np.float32) * (1.0 - blend) +
            src_row.astype(np.float32) * blend,
            0, 255,
        ).astype(np.uint8)

    return filled


def inpaint_multipass(
    backend: InpaintBackend,
    image_pil: Image.Image,
    mask_float: np.ndarray,
    passes: int,
) -> Image.Image:
    """
    Run the inpainting backend ``passes`` times.

    Pass 1  -- inpaint using the full feathered mask
    Pass 2+ -- erode the mask so subsequent passes focus on edge blending
    """
    mask_u8 = (mask_float * 255).astype(np.uint8)
    current_img = image_pil
    current_mask = mask_u8.copy()

    for i in range(passes):
        if current_mask.max() == 0:
            log.debug("Multi-pass: mask fully eroded after %d pass(es), stopping.", i)
            break

        mask_pil = Image.fromarray(current_mask, mode="L")
        log.debug("  Inpaint pass %d/%d ...", i + 1, passes)
        current_img = backend(current_img, mask_pil)

        if i < passes - 1:
            from .masks import erode_mask
            current_mask = erode_mask(current_mask, iterations=4)

    return current_img


@dataclass
class MaskingJob:
    """Represents a manually masked image ready for inpainting."""
    img_path: Path
    mask: np.ndarray   # float32 [0, 1], full resolution


def inpaint_one(
    job: MaskingJob,
    backend: InpaintBackend,
    config,
) -> Optional[Image.Image]:
    """
    Full inpainting pipeline for a single image:
      1. Load image (crop if in 'full' mode)
      2. Pre-fill masked region with sampled floor texture
      3. Extract nadir gnomonic patch
      4. Project mask into patch space
      5. Multi-pass inpainting on the flat patch
      6. Remap inpainted patch back into equirectangular space
      7. Feather-blend result over original
    """
    from .reprojection import build_nadir_inverse_map, extract_nadir_patch, blend_patch_back

    # 1 -- load (and crop if in 'full' mode)
    image_pil = Image.open(job.img_path).convert("RGB")
    w_orig, h_orig = image_pil.size

    is_cropped_mode = getattr(config, 'mode', 'full') == 'inpaint'
    if not is_cropped_mode:
        image_pil = image_pil.crop((0, 128, w_orig, 2128))
    image_np = np.array(image_pil)

    mask_float = job.mask
    mask_u8 = (mask_float * 255).astype(np.uint8)

    eq_h, eq_w = image_np.shape[:2]
    patch_size = config.nadir_patch_size
    fov_deg = config.nadir_fov

    # 2 -- pre-fill masked region with real floor texture
    log.debug("  Pre-filling mask with floor texture ...")
    image_np = prefill_mask_with_floor(image_np, mask_u8)

    # 3 -- extract flat nadir patch
    log.debug("  Extracting nadir patch (FOV=%.0f, size=%dpx) ...", fov_deg, patch_size)
    nadir_patch = extract_nadir_patch(image_np, patch_size, fov_deg)

    # 4 -- build inverse map and project mask into patch space
    log.debug("  Building nadir inverse map ...")
    map_x, map_y = build_nadir_inverse_map(eq_h, eq_w, patch_size, fov_deg)

    mask_as_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=2)
    nadir_mask_rgb = extract_nadir_patch(mask_as_rgb, patch_size, fov_deg)
    nadir_mask_float = nadir_mask_rgb[:, :, 0].astype(np.float32) / 255.0
    nadir_mask_u8 = nadir_mask_rgb[:, :, 0]

    if nadir_mask_u8.max() < 10:
        log.warning("  Mask does not overlap nadir patch -- running full-equirect inpaint.")
        from .masks import expand_and_feather
        result_pil = inpaint_multipass(
            backend,
            Image.fromarray(image_np),
            expand_and_feather(mask_float, config.dilation_iter, config.blur_kernel),
            config.inpaint_passes,
        )
        return result_pil

    # 5 -- multi-pass inpainting on the flat undistorted patch
    log.debug("  Inpainting nadir patch (%d pass[es]) ...", config.inpaint_passes)
    inpainted_patch = inpaint_multipass(
        backend,
        Image.fromarray(nadir_patch),
        nadir_mask_float,
        config.inpaint_passes,
    )
    inpainted_patch_np = np.array(inpainted_patch)

    # 6 & 7 -- remap and blend
    log.debug("  Blending patch back into equirectangular frame ...")
    result_np = blend_patch_back(
        image_np,
        inpainted_patch_np,
        mask_float,
        map_x,
        map_y,
    )

    return Image.fromarray(result_np)
