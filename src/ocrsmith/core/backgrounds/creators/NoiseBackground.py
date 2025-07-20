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
        else:
            x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
            noise = np.stack([
                np.sin(x) * np.cos(y),
                np.sin(x + 2) * np.cos(y + 2),
                np.sin(x + 4) * np.cos(y + 4)
            ], axis=2)
            noise = (noise + 1) / 2  
        
        # Apply noise to base color
        base = np.array(self.base_color) / 255.0
        result = base + (noise - 0.5) * self.intensity
        result = np.clip(result, 0, 1) * 255
        
        return Image.fromarray(result.astype(np.uint8))
