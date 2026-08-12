# src/ocrsmith/core/backgrounds/BackgroundStrategy.py

from abc import ABC, abstractmethod

from PIL import Image


class BackgroundStrategy(ABC):
    """Strategy interface for background rendering."""

    @abstractmethod
    def render(self, width: int, height: int, **kwargs) -> Image.Image:
        """Render a background image with given dimensions."""
        pass
