# src/ocrsmith/core/text_placement/strategies/PageNumberPlacementStrategy.py

from ..PlacementResult import PlacementResult
from ..TextPlacementStrategy import TextPlacementStrategy


class PageNumberPlacementStrategy(TextPlacementStrategy):
    """Places text as page number (bottom right corner)"""

    def __init__(
        self,
        position: str = "bottom_right",
        margin: int = 20,
        bottom_margin: int = None,
        right_margin: int = None,
    ):
        # Backward-compat and schema support
        self.position = position
        if bottom_margin is not None and right_margin is not None:
            self.bottom_margin = bottom_margin
            self.right_margin = right_margin
        else:
            self.bottom_margin = margin
            self.right_margin = margin

    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size

        # Calculate page number position
        if self.position == "bottom_left":
            x = self.right_margin
            y = bg_h - text_h - self.bottom_margin
        elif self.position == "bottom_center":
            x = (bg_w - text_w) // 2
            y = bg_h - text_h - self.bottom_margin
        else:  # bottom_right
            x = bg_w - text_w - self.right_margin
            y = bg_h - text_h - self.bottom_margin

        # Compose image
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)

        bbox = (x, y, x + text_w, y + text_h)

        metadata = {
            "placement_type": "page_number",
            "position": (x, y),
            "margins": (self.bottom_margin, self.right_margin),
            "content_type": "page_number",
        }

        return PlacementResult(composed_image, bbox, metadata)
