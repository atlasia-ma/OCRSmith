# src/ocrsmith/core/backgrounds/strategies/TextureBackground.py

import numpy as np
from PIL import Image

from .BaseBackground import BaseBackground


class TextureBackground(BaseBackground):
    """Strategy for rendering textured backgrounds."""

    def __init__(self, base_color: tuple[int, int, int] = (240, 240, 240), noise_level: int = 20):
        super().__init__()
        self.default_base_color = base_color
        self.default_noise_level = noise_level

    def render(self, width: int, height: int, **kwargs) -> Image.Image:
        base_color = kwargs.get("base_color", self.default_base_color)
        noise_level = kwargs.get("noise_level", self.default_noise_level)

        # Vectorized noise generation for performance
        base = np.full((height, width, 3), base_color, dtype=np.int16)
        noise = np.random.randint(-noise_level, noise_level + 1, size=(height, width, 1), dtype=np.int16)
        noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")
