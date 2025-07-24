# src/ocrsmith/core/text_placement/placers/BasePlacementStrategy.py

from abc import abstractmethod

from ..TextPlacementStrategy import TextPlacementStrategy

class BasePlacementStrategy(TextPlacementStrategy):
    @abstractmethod
    def render_text():
        pass
    