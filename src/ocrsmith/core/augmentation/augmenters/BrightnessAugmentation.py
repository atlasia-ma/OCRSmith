# src/ocrsmith/core/augmentation/augmenters/BrightnessAugmentation.py

from ..AugmentationStrategy import AugmentationStrategy

from PIL import ImageEnhance
        
class BrightnessAugmentation(AugmentationStrategy):
    """Adjusts image brightness"""
    def __init__(self, brightness_factor=1.2):
        self.brightness_factor = brightness_factor
    
    def apply(self, image, **kwargs):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.brightness_factor)