# src/ocrsmith/core/text_placement/placers/RandomPlacementStrategy.py

from ..TextPlacementStrategy import TextPlacementStrategy
from ..PlacementResult import PlacementResult

import random

class RandomPlacementStrategy(TextPlacementStrategy):
    """Places text at random positions within margins"""
    def __init__(self, margin_x=20, margin_y=20):
        self.margin_x = margin_x
        self.margin_y = margin_y
    
    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size
        
        max_x = max(0, bg_w - text_w - self.margin_x)
        max_y = max(0, bg_h - text_h - self.margin_y)
        
        x = random.randint(self.margin_x, max_x) if max_x > self.margin_x else self.margin_x
        y = random.randint(self.margin_y, max_y) if max_y > self.margin_y else self.margin_y
        
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)
        
        bbox = (x, y, x + text_w, y + text_h)
        
        metadata = {
            'placement_type': 'random',
            'position': (x, y),
            'margins': (self.margin_x, self.margin_y)
        }
        
        return PlacementResult(composed_image, bbox, metadata)
    