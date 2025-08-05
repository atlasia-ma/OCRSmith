# src/ocrsmith/core/backgrounds/strategies/TextureBackground.py
from .BaseBackground import BaseBackground
from PIL import Image
import random
from typing import Tuple

class TextureBackground(BaseBackground):
    """Strategy for rendering textured backgrounds."""
    
    def __init__(self, base_color: Tuple[int, int, int] = (240, 240, 240), 
                 noise_level: int = 20):
        super().__init__()
        self.default_base_color = base_color
        self.default_noise_level = noise_level
    
    def render(self, width: int, height: int, **kwargs) -> Image.Image:
        base_color = kwargs.get('base_color', self.default_base_color)
        noise_level = kwargs.get('noise_level', self.default_noise_level)
        
        img = Image.new('RGB', (width, height), base_color)
        pixels = img.load()
        
        for y in range(height):
            for x in range(width):
                # Add random noise to base color
                noise = random.randint(-noise_level, noise_level)
                r = max(0, min(255, base_color[0] + noise))
                g = max(0, min(255, base_color[1] + noise))
                b = max(0, min(255, base_color[2] + noise))
                pixels[x, y] = (r, g, b)
        
        return img