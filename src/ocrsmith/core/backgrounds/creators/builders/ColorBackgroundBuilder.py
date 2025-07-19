# src/ocrsmith/core/backgrounds/creators/builders/ColorBackgroundBuilder.py

from ..ColorBackground import ColorBackground

class ColorBackgroundBuilder:
    def __init__(self):
        self._instance = None
        
    def __call__(self, color, **_ignored):
        if not self._instance:
            self._instance = ColorBackground(color)
        return self._instance