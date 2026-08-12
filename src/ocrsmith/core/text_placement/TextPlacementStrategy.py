# src/ocrsmith/core/text_placement/TextPlacementStrategy.py

from abc import ABC, abstractmethod


class TextPlacementStrategy(ABC):
    @abstractmethod
    def place_text(self, text_image, background_image, **kwargs):
        """Place text image onto background and return PlacementResult."""
        raise NotImplementedError
