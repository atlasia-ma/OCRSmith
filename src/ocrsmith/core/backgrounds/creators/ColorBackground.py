# src/orsmith/core/backgrounds/creators/ColorBackground.py

from .BaseBackground import BaseBackground

class ColorBackground(BaseBackground):
    def __init__(self, color):
        super().__init__()
        self.color = color

    def render(self):
        print(f"Rendering solid background with color: {self.color}") 

    