# src/ocrsmith/core/backgrounds/strategies/GradientBackground.py
from .BaseBackground import BaseBackground
from PIL import Image, ImageDraw
from typing import Tuple

class GradientBackground(BaseBackground):
    """Strategy for rendering gradient backgrounds."""
    
    def __init__(self, start_color: Tuple[int, int, int] = (255, 255, 255), 
                 end_color: Tuple[int, int, int] = (200, 200, 200),
                 direction: str = 'horizontal'):
        super().__init__()
        self.default_start_color = start_color
        self.default_end_color = end_color
        self.default_direction = direction
    
    def render(self, width: int, height: int, **kwargs) -> Image.Image:
        start_color = kwargs.get('start_color', self.default_start_color)
        end_color = kwargs.get('end_color', self.default_end_color)
        direction = kwargs.get('direction', self.default_direction)
        
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        if direction == 'horizontal':
            for x in range(width):
                ratio = x / width
                color = self._interpolate_color(start_color, end_color, ratio)
                draw.line([(x, 0), (x, height)], fill=color)
        
        elif direction == 'vertical':
            for y in range(height):
                ratio = y / height
                color = self._interpolate_color(start_color, end_color, ratio)
                draw.line([(0, y), (width, y)], fill=color)
        
        elif direction == 'diagonal':
            for y in range(height):
                for x in range(width):
                    ratio = (x + y) / (width + height)
                    color = self._interpolate_color(start_color, end_color, ratio)
                    img.putpixel((x, y), color)
        
        return img
    
    def _interpolate_color(self, color1: Tuple[int, int, int],
                          color2: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        return tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))
