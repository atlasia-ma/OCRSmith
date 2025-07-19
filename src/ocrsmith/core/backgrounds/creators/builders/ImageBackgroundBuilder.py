# src/ocrsmith/core/backgrounds/creators/builders/ImageBackgroundBuilder.py

from ..ImageBackground import ImageBackground

class ImageBackgroundBuilder:
    def __init__(self):
        self._instance = None
        
    def __call__(self, image_path, **_ignored):
        if not self._instance:
            self._instance = ImageBackground(image_path)
        return self._instance