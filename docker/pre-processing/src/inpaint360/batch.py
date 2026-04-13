"""Batch inpainting pipeline."""

import logging

from PIL import Image
from tqdm import tqdm

from .backends import load_backend
from .inpaint import inpaint_one, MaskingJob

log = logging.getLogger(__name__)


def run_crop_only(image_files: list, cfg) -> None:
    """Crop equirectangular images (rows 128:2128) without any inpainting."""
    output_dir = cfg.dataset_path / "output"
    output_dir.mkdir(exist_ok=True)

    log.info("Cropping %d image(s) to output/ ...", len(image_files))
    errors = []

    for img_path in tqdm(image_files, desc="Cropping", unit="img"):
        out_path = output_dir / img_path.name
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            # Crop rows 128 to 2128 (same as the interactive phase does)
            cropped = img.crop((0, 128, w, 2128))
            cropped.save(
                out_path,
                format="JPEG",
                quality=cfg.jpeg_quality,
                subsampling=0,
            )
        except Exception as exc:
            log.error("Failed to crop '%s': %s", img_path.name, exc)
            errors.append(img_path.name)

    success = len(image_files) - len(errors)
    log.info("Cropping complete -- %d/%d succeeded.", success, len(image_files))
    if errors:
        log.warning("Failed images (%d): %s", len(errors), ", ".join(errors))
    log.info("Cropped images saved to: %s", output_dir)


def run_batch_inpainting(jobs: list, cfg) -> None:
    """Run inpainting on all queued jobs and save results."""
    if not jobs:
        log.info("No images queued -- nothing to inpaint.")
        return

    output_dir = cfg.dataset_path / "output"
    output_dir.mkdir(exist_ok=True)

    log.info("Loading inpainting backend: %s ...", cfg.inpaint_backend.upper())
    backend = load_backend(
        cfg.inpaint_backend,
        floor_prompt=cfg.floor_prompt,
        sd_steps=cfg.sd_steps,
        sd_guidance=cfg.sd_guidance,
        sd_strength=cfg.sd_strength,
    )

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
    log.info("Inpainting complete -- %d/%d succeeded.", success, len(jobs))
    if errors:
        log.warning("Failed images (%d): %s", len(errors), ", ".join(errors))
    log.info("Results saved to: %s", output_dir)
