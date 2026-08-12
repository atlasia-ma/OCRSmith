# src/ocrsmith/core/augmentation/strategies/BrightnessAugmentation.py

from PIL import ImageEnhance

from ..AugmentationStrategy import AugmentationStrategy


class BrightnessAugmentation(AugmentationStrategy):
    """Adjusts image brightness"""

    def __init__(self, brightness_factor=1.2):
        self.brightness_factor = brightness_factor

    def apply(self, image, **kwargs):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.brightness_factor)
