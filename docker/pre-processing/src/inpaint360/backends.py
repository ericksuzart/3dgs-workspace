"""Inpainting backend abstractions (SD2, LaMa, fallback)."""

import logging
import os
from typing import Protocol

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_NEG_PROMPT = (
    "tripod, camera, object, shadow, dark spot, person, "
    "blurry, distorted, artifact, watermark"
)


class InpaintBackend(Protocol):
    """Protocol for inpainting backends."""

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image: ...


class SD2InpaintBackend:
    """
    HuggingFace diffusers -- stabilityai/stable-diffusion-2-inpainting.

    Requires:  pip install diffusers accelerate transformers
    """

    MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"

    def __init__(self, floor_prompt: str, sd_steps: int, sd_guidance: float, sd_strength: float):
        from diffusers import StableDiffusionInpaintPipeline

        device = torch.device("cuda")
        log.info(
            "Loading diffusers SD2 inpainting model on %s "
            "(first run downloads ~5 GB) ...", device,
        )
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
        ).to(device)
        self._pipe.enable_attention_slicing()
        self._prompt = floor_prompt
        self._steps = sd_steps
        self._guidance = sd_guidance
        self._strength = sd_strength
        log.info("SD2 inpainting model ready.")

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        image_rgb = image_pil.convert("RGB")
        mask_l = mask_pil.convert("L")
        w, h = image_rgb.size

        output = self._pipe(
            prompt=self._prompt,
            negative_prompt=DEFAULT_NEG_PROMPT,
            image=image_rgb,
            mask_image=mask_l,
            height=h,
            width=w,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance,
            strength=self._strength,
        )
        return output.images[0].convert("RGB")


class LamaInpaintBackend:
    """
    IOPaint LaMa with CROP HD strategy (cleanup.pictures quality).
    """

    def __init__(self):
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading IOPaint LaMa model on %s ...", device)
        self._model = ModelManager(name="lama", device=device)
        self._InpaintRequest = InpaintRequest
        self._HDStrategy = HDStrategy
        self._crop_margin = int(os.environ.get("LAMA_CROP_MARGIN", "196"))
        self._crop_trigger = int(os.environ.get("LAMA_CROP_TRIGGER", "512"))
        log.info(
            "IOPaint LaMa ready (CROP strategy, margin=%d, trigger=%d).",
            self._crop_margin, self._crop_trigger,
        )

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        image_np = np.array(image_pil.convert("RGB"))
        mask_np = np.array(mask_pil.convert("L"))

        config = self._InpaintRequest(
            hd_strategy=self._HDStrategy.CROP,
            hd_strategy_crop_margin=self._crop_margin,
            hd_strategy_crop_trigger_size=self._crop_trigger,
        )
        result_np = self._model(image_np, mask_np, config)
        return Image.fromarray(result_np.astype(np.uint8))


class SimpleLamaFallback:
    """Last-resort fallback using simple_lama_inpainting."""

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        log.warning(
            "Falling back to simple_lama_inpainting.  Results will be lower quality. "
            "Ensure IOPaint is installed:  pip install iopaint"
        )
        self._lama = SimpleLama()

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        return self._lama(image_pil, mask_pil)


def load_backend(backend_name: str, **kwargs) -> InpaintBackend:
    """
    Load the best available inpainting backend.

    Priority:
      1. SD2 via diffusers          (best quality, requires GPU + ~5 GB model)
      2. IOPaint LaMa + CROP HD     (cleanup.pictures quality)
      3. simple_lama_inpainting     (last resort)
    """
    if backend_name == "sd2":
        try:
            return SD2InpaintBackend(**kwargs)
        except Exception as exc:
            log.warning(
                "SD2 backend failed to load (%s) -- trying IOPaint LaMa.\n"
                "  Install diffusers:  pip install diffusers accelerate transformers",
                exc,
            )

    try:
        return LamaInpaintBackend()
    except Exception as exc:
        log.warning(
            "IOPaint LaMa failed to load (%s) -- falling back to simple_lama_inpainting.",
            exc,
        )

    return SimpleLamaFallback()
