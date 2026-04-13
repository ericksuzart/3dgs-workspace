"""Nadir reprojection helpers for 360° equirectangular images."""

from typing import Tuple

import cv2
import numpy as np


def build_nadir_inverse_map(
    eq_h: int,
    eq_w: int,
    patch_size: int,
    fov_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the inverse mapping from equirectangular space to nadir gnomonic patch.

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

    # Equirectangular -> spherical angles
    lon = (ex_grid / eq_w) * 2.0 * np.pi - np.pi          # [-pi, pi]
    lat = np.pi / 2.0 - (ey_grid / eq_h) * np.pi           # [pi/2, -pi/2]

    # Spherical -> 3-D unit vector
    cos_lat = np.cos(lat)
    X =  cos_lat * np.cos(lon)
    Y =  cos_lat * np.sin(lon)
    Z =  np.sin(lat)                                         # negative = below equator

    # Gnomonic projection onto plane z = -1  (camera looks toward -Z)
    half_tan = float(np.tan(np.radians(fov_deg / 2.0)))

    with np.errstate(divide="ignore", invalid="ignore"):
        px = X / (-Z)
        py = Y / (-Z)

    # Map gnomonic coords [-half_tan, half_tan] -> patch pixel [0, patch_size]
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

    # Invert: we need forward map (patch -> equirect) not inverse
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

    # Spherical -> equirect pixel
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
    """
    sampled = cv2.remap(
        patch_rgb, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Zero out sampled pixels that mapped to invalid coords
    valid = (map_x >= 0) & (map_y >= 0)            # HxW bool
    blend_weight = (mask_float * valid).astype(np.float32)[:, :, None]

    result = (
        equi_rgb.astype(np.float32) * (1.0 - blend_weight) +
        sampled.astype(np.float32) * blend_weight
    ).astype(np.uint8)

    return result
