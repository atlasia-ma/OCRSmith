# src/ocrsmith/core/text_placement/strategies/CenterPlacementStrategy.py

from ..PlacementResult import PlacementResult
from ..TextPlacementStrategy import TextPlacementStrategy


class CenterPlacementStrategy(TextPlacementStrategy):
    """Centers text on background"""

    def __init__(self, padding: int = 0):
        self.padding = padding

    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size

        x = max(self.padding, (bg_w - text_w) // 2)
        y = max(self.padding, (bg_h - text_h) // 2)

        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)

        bbox = (x, y, x + text_w, y + text_h)

        metadata = {
            "placement_type": "center",
            "position": (x, y),
            "center_offset": ((bg_w - text_w) / 2, (bg_h - text_h) / 2),
            "padding": self.padding,
        }

        return PlacementResult(composed_image, bbox, metadata)
