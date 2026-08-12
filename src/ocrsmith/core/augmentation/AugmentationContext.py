# src/ocrsmith/core/augmentation/AugmentationContext.py


class AugmentationContext:
    def __init__(self, strategy=None):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def apply(self, image, **kwargs):
        if not self._strategy:
            raise ValueError("No augmentation strategy set")
        return self._strategy.apply(image, **kwargs)
