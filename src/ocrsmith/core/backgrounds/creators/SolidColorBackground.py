# src/orsmith/core/backgrounds/creators/SolidColorBackground.py

from .BaseBackground import BaseBackground
from PIL import Image
from typing import Tuple, Union

class SolidColorBackground(BaseBackground):
    def __init__(self, color: Union[str, Tuple[int, int, int]] = (255, 255, 255), **_ignored):
        if isinstance(color, str):
            color = color.lstrip('#')
            self.color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        else:
            self.color = color
    
    def render(self, width: int, height: int) -> Image.Image:
        return Image.new('RGB', (width, height), self.color)

