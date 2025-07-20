# src/ocrsmith/core/backgrounds/BackgroundInterface.py

from abc import ABC, abstractmethod
from PIL import Image

class BackgroundInterface(ABC):
    @abstractmethod
    def render(self, width: int, height: int) -> Image.Image:
        pass
