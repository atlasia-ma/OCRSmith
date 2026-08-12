# src/ocrsmith/core/text_placement/strategies/BasePlacementStrategy.py

from abc import abstractmethod

from ..TextPlacementStrategy import TextPlacementStrategy


class BasePlacementStrategy(TextPlacementStrategy):
    @abstractmethod
    def place_text(self, text_image, background_image, **kwargs):
        raise NotImplementedError
