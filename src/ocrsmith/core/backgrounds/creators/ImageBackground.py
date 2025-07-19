# src/ocrsmith/core/backgrounds/creators/ImageBackground.py
from .BaseBackground import BaseBackground

class ImageBackground(BaseBackground):
    def __init__(self, image_path, **_ignored):
        super().__init__()
        self.image_path = image_path

    def render(self):
        print(f"Rendering background with image: {self.image_path}")
    