# src/ocrsmith/core/text_placement/placers/PageNumberPlacementStrategy.py

from ..TextPlacementStrategy import TextPlacementStrategy
from ..PlacementResult import PlacementResult

class PageNumberPlacementStrategy(TextPlacementStrategy):
    """Places text as page number (bottom right corner)"""
    def __init__(self, bottom_margin=30, right_margin=30):
        self.bottom_margin = bottom_margin
        self.right_margin = right_margin
    
    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size
        
        # Calculate page number position (bottom right)
        x = bg_w - text_w - self.right_margin
        y = bg_h - text_h - self.bottom_margin
        
        # Compose image
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)
        
        bbox = (x, y, x + text_w, y + text_h)
        
        metadata = {
            'placement_type': 'page_number',
            'position': (x, y),
            'margins': (self.bottom_margin, self.right_margin),
            'content_type': 'page_number'
        }
        
        return PlacementResult(composed_image, bbox, metadata)