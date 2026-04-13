"""Interactive masking phase using OpenCV."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .masks import expand_and_feather
from .inpaint import MaskingJob

log = logging.getLogger(__name__)

WINDOW_NAME = "Masking - 360 Tripod Removal"

# OpenCV colours (BGR)
CLR_POSITIVE = (57, 255, 20)
CLR_NEGATIVE = (0, 80, 255)
CLR_MASK = (0, 200, 100)
CLR_WHITE = (255, 255, 255)
CLR_YELLOW = (0, 255, 255)
CLR_GRAY = (160, 160, 160)
CLR_HUD_BG = (15, 15, 15)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def draw_hud(canvas, img_idx, total, filename, n_queued):
    w = canvas.shape[1]
    hud_h = 72
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), CLR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

    lines = [
        (f"  [{img_idx + 1}/{total}] {filename}   |   Queued: {n_queued}", CLR_YELLOW),
        ("  Left-click and drag to paint the mask directly.", CLR_WHITE),
        ("  A: accept & next   R: reset mask   S: skip image   Q: quit", CLR_GRAY),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (0, 17 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def apply_mask_overlay(canvas, mask_float, alpha=0.45):
    colored = np.zeros_like(canvas)
    colored[:] = CLR_MASK
    mask_bool = mask_float > 0.1
    canvas[mask_bool] = (
        canvas[mask_bool] * (1.0 - alpha) + colored[mask_bool] * alpha
    ).astype(np.uint8)

    mask_u8 = (mask_float * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask_u8 > 30).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, CLR_WHITE, 1)


def run_interactive_phase(image_files: list, cfg) -> list:
    """Run the interactive masking phase. Returns a list of MaskingJob objects."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    jobs = []

    is_cropped_mode = cfg.mode == "inpaint"

    def load_for_display(path):
        full = cv2.imread(str(path))
        if full is None:
            raise IOError(f"cv2.imread returned None for {path}")
        # In 'inpaint' mode, images are already cropped — no cropping needed
        if not is_cropped_mode:
            full = full[128:2128, :]
        h, w = full.shape[:2]
        dw, dh = int(w * cfg.display_scale), int(h * cfg.display_scale)
        disp = cv2.resize(full, (dw, dh), interpolation=cv2.INTER_AREA)
        return full, disp

    total = len(image_files)
    img_idx = 0

    state = {
        "is_drawing": False,
        "last_pt": None,
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
            state["last_pt"] = (fx, fy)
            cv2.circle(current_mask[0], (fx, fy), state["brush_size"], 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and state["is_drawing"]:
            if state["last_pt"]:
                cv2.line(current_mask[0], state["last_pt"], (fx, fy),
                         255, thickness=state["brush_size"] * 2)
            cv2.circle(current_mask[0], (fx, fy), state["brush_size"], 255, -1)
            state["last_pt"] = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP:
            state["is_drawing"] = False
            state["last_pt"] = None

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    try:
        while img_idx < total:
            img_path = image_files[img_idx]
            log.info("Image %d/%d: %s", img_idx + 1, total, img_path.name)

            try:
                full_bgr, display_bgr = load_for_display(img_path)
            except Exception as exc:
                log.error("Cannot load '%s': %s -- skipping.", img_path.name, exc)
                img_idx += 1
                continue

            h, w = full_bgr.shape[:2]
            current_mask[0] = np.zeros((h, w), dtype=np.uint8)
            state["is_drawing"] = False
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
                    log.info("Q pressed -- stopping interactive phase.")
                    return jobs
                elif key == ord("s"):
                    log.info("Skipped: %s", img_path.name)
                    img_idx += 1
                    advance = True
                elif key == ord("a"):
                    mask_float = current_mask[0].astype(np.float32) / 255.0
                    final_mask = expand_and_feather(
                        mask_float, cfg.dilation_iter, cfg.blur_kernel
                    )
                    jobs.append(MaskingJob(img_path=img_path, mask=final_mask))
                    log.info("Queued %s (%d/%d).", img_path.name, len(jobs), total)
                    img_idx += 1
                    advance = True
                elif key == ord("r"):
                    current_mask[0] = np.zeros((h, w), dtype=np.uint8)
                    log.info("Reset mask for %s.", img_path.name)

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("Window closed -- stopping interactive phase.")
                    return jobs
    finally:
        cv2.destroyAllWindows()

    return jobs
