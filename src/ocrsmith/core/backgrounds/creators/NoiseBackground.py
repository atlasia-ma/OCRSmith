# src/ocrsmith/core/backgrounds/creators/NoiseBackground.py

from .BaseBackground import BaseBackground
from typing import Tuple
from PIL import Image
import numpy as np

class NoiseBackground(BaseBackground):
    def __init__(self, noise_type: str = 'random', intensity: float = 0.5, 
                 base_color: Tuple[int, int, int] = (128, 128, 128), **_ignored):
        self.noise_type = noise_type
        self.intensity = intensity
        self.base_color = base_color
    
    def render(self, width: int, height: int) -> Image.Image:
        if self.noise_type == 'random':
            noise = np.random.random((height, width, 3))
            return self._apply_color(noise)

        elif self.noise_type == 'gaussian':
            noise = np.random.normal(loc=235, scale=10, size=(height, width)).clip(0, 255)
            return Image.fromarray(noise.astype(np.uint8), mode='L').convert("RGBA")

        else:
            raise ValueError(f"Unknown noise type '{self.noise_type}' for NoiseBackground")

    def _apply_color(self, noise: np.ndarray) -> Image.Image:
        base = np.array(self.base_color) / 255.0
        result = base + (noise - 0.5) * self.intensity
        result = np.clip(result, 0, 1) * 255
        return Image.fromarray(result.astype(np.uint8))