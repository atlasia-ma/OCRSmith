# src/ocrsmith/core/text_placement/placers/CenterPlacementStrategy.py

from ..TextPlacementStrategy import TextPlacementStrategy
from ..PlacementResult import PlacementResult

class CenterPlacementStrategy(TextPlacementStrategy):
    """Centers text on background"""
    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size
        
        x = (bg_w - text_w) // 2
        y = (bg_h - text_h) // 2
        
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)
        
        bbox = (x, y, x + text_w, y + text_h)
        
        metadata = {
            'placement_type': 'center',
            'position': (x, y),
            'center_offset': ((bg_w - text_w) / 2, (bg_h - text_h) / 2)
        }
        
        return PlacementResult(composed_image, bbox, metadata)