# src/ocrsmith/core/text_placement/placers/PageTitlePlacementStrategy.py

from ..TextPlacementStrategy import TextPlacementStrategy
from ..PlacementResult import PlacementResult

class PageTitlePlacementStrategy(TextPlacementStrategy):
    """Places text as page title (top center with margins)"""
    def __init__(self, top_margin=50, side_margin=20):
        self.top_margin = top_margin
        self.side_margin = side_margin
    
    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size
        
        # Calculate title position (center horizontally, top with margin)
        x = (bg_w - text_w) // 2
        y = self.top_margin
        
        # Compose image
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)
        
        bbox = (x, y, x + text_w, y + text_h)
        
        metadata = {
            'placement_type': 'page_title',
            'position': (x, y),
            'margins': (self.top_margin, self.side_margin),
            'content_type': 'title'
        }
        
        return PlacementResult(composed_image, bbox, metadata)