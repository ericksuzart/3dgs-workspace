"""Mask manipulation utilities."""

import cv2
import numpy as np


def expand_and_feather(mask: np.ndarray, dilation_iter: int, blur_kernel: int) -> np.ndarray:
    """Dilate then Gaussian-blur the binary mask. Returns float32 in [0, 1]."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation_iter)
    blurred = cv2.GaussianBlur(dilated.astype(np.float32), (blur_kernel, blur_kernel), 0)
    return np.clip(blurred, 0.0, 1.0)


def erode_mask(mask_u8: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Erode a uint8 binary mask (used between multi-pass refinement steps)."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(mask_u8, kernel, iterations=iterations)
