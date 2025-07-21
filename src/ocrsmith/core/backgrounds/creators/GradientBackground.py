# src/ocrsmith/core/backgrounds/creators/GradientBackground.py

from .BaseBackground import BaseBackground
from PIL import Image, ImageDraw
from typing import Tuple

class GradientBackground(BaseBackground):
    
    def __init__(self, start_color: Tuple[int, int, int] = (255, 255, 255), end_color: Tuple[int, int, int] = (200, 200, 200), 
                 direction: str = 'horizontal', **_ignored):
        self.start_color = start_color
        self.end_color = end_color
        self.direction = direction
    
    def render(self, width: int, height: int) -> Image.Image:
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        if self.direction == 'horizontal':
            for x in range(width):
                ratio = x / width
                color = self._interpolate_color(self.start_color, self.end_color, ratio)
                draw.line([(x, 0), (x, height)], fill=color)
        
        elif self.direction == 'vertical':
            for y in range(height):
                ratio = y / height
                color = self._interpolate_color(self.start_color, self.end_color, ratio)
                draw.line([(0, y), (width, y)], fill=color)
        
        elif self.direction == 'diagonal':
            for y in range(height):
                for x in range(width):
                    ratio = (x + y) / (width + height)
                    color = self._interpolate_color(self.start_color, self.end_color, ratio)
                    img.putpixel((x, y), color)
        
        return img
    
    def _interpolate_color(self, color1: Tuple[int, int, int], 
                          color2: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        return tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))
