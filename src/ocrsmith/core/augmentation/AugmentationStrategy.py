# src/ocrsmith/core/augmentation/AugmentationStrategy.py

from abc import ABC, abstractmethod

class AugmentationStrategy(ABC):
    @abstractmethod
    def apply():
        pass
    