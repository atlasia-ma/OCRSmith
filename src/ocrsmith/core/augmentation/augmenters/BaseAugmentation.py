# src/ocrsmith/core/text_placement/BaseAugmentation

from abc import abstractmethod

from ..AugmentationStrategy import AugmentationStrategy

class BaseAugmentation(AugmentationStrategy):
    @abstractmethod
    def apply():
        pass
    