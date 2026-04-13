"""CLI entry point for inpaint360."""

import logging
import sys
from pathlib import Path

from .config import load_config
from .config import SUPPORTED_EXTENSIONS
from .batch import run_batch_inpainting
from .batch import run_crop_only

log = logging.getLogger(__name__)


def main() -> None:
    """Main entry point: crop / interactive masking -> batch inpainting."""
    cfg = load_config()

    images = sorted(
        p for p in cfg.dataset_path.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    )
    if not images:
        log.error("No supported images found in '%s'.", cfg.dataset_path)
        sys.exit(1)
    log.info("Found %d image(s) in %s.", len(images), cfg.dataset_path)

    # ── Mode: crop-only (no masking, no inpainting) ──────────────────────
    if cfg.mode == "crop":
        log.info("=" * 60)
        log.info("CROP-ONLY MODE — cropping equirectangular frames")
        log.info("=" * 60)
        run_crop_only(images, cfg)
        return

    # ── Mode: full or inpaint (interactive masking + inpainting) ─────────
    # Lazy import: cv2 in interactive.py triggers Qt/X11 at import time
    from .interactive import run_interactive_phase

    log.info("=" * 60)
    if cfg.mode == "full":
        log.info("PHASE 1 -- Interactive masking  (%d images)", len(images))
    else:
        log.info("PHASE 1 -- Interactive masking on CROPPED images  (%d images)", len(images))
    log.info("=" * 60)
    jobs = run_interactive_phase(images, cfg)

    if not jobs:
        log.info("No images accepted -- exiting without inpainting.")
        return

    log.info("=" * 60)
    log.info("PHASE 2 -- Batch inpainting  (%d images queued)", len(jobs))
    log.info("=" * 60)
    run_batch_inpainting(jobs, cfg)


if __name__ == "__main__":
    main()
