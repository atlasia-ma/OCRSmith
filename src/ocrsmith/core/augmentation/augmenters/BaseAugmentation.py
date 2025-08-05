# src/ocrsmith/core/augmentation/augmenters/BaseAugmentation.py

from abc import abstractmethod

from ..AugmentationStrategy import AugmentationStrategy

class BaseAugmentation(AugmentationStrategy):
    @abstractmethod
    def apply():
        pass
    