# src/ocrsmith/core/augmentation/strategies/BaseAugmentation.py

from abc import abstractmethod

from ..AugmentationStrategy import AugmentationStrategy


class BaseAugmentation(AugmentationStrategy):
    @abstractmethod
    def apply():
        pass
