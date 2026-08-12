# src/ocrsmith/core/text_placement/strategies/PageTitlePlacementStrategy.py

from ..PlacementResult import PlacementResult
from ..TextPlacementStrategy import TextPlacementStrategy


class PageTitlePlacementStrategy(TextPlacementStrategy):
    """Places text as page title (top center with margins)"""

    def __init__(
        self, position: str = "top_center", margin: int = 30, top_margin: int = None, side_margin: int = None
    ):
        self.position = position
        if top_margin is not None and side_margin is not None:
            self.top_margin = top_margin
            self.side_margin = side_margin
        else:
            self.top_margin = margin
            self.side_margin = margin

    def place_text(self, text_image, background_image, **kwargs) -> PlacementResult:
        bg_w, bg_h = background_image.size
        text_w, text_h = text_image.size

        # Calculate title position
        if self.position == "top_left":
            x = self.side_margin
            y = self.top_margin
        elif self.position == "top_right":
            x = bg_w - text_w - self.side_margin
            y = self.top_margin
        else:  # top_center
            x = (bg_w - text_w) // 2
            y = self.top_margin

        # Compose image
        composed_image = background_image.copy()
        composed_image.paste(text_image, (x, y), text_image)

        bbox = (x, y, x + text_w, y + text_h)

        metadata = {
            "placement_type": "page_title",
            "position": (x, y),
            "margins": (self.top_margin, self.side_margin),
            "content_type": "title",
        }

        return PlacementResult(composed_image, bbox, metadata)
