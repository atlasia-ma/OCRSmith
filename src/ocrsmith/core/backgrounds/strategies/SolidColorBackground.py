# src/ocrsmith/core/backgrounds/strategies/SolidColorBackground.py

from PIL import Image

from .BaseBackground import BaseBackground


class SolidColorBackground(BaseBackground):
    """Strategy for rendering solid color backgrounds."""

    def __init__(self, color: tuple[int, int, int] = (255, 255, 255)):
        super().__init__()
        self.default_color = color

    def render(self, width: int, height: int, **kwargs) -> Image.Image:
        color = kwargs.get("color", self.default_color)
        return Image.new("RGB", (width, height), color)
