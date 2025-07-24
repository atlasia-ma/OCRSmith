# src/ocrsmith/core/text_placement/TextPlacementStrategy.py

from abc import ABC, abstractmethod

class TextPlacementStrategy(ABC):
    @abstractmethod
    def place_text():
        pass
    