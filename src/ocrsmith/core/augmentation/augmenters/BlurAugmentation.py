# src/ocrsmith/core/augmentation/augmenters/BlurAugmentation.py

from ..AugmentationStrategy import AugmentationStrategy

import numpy as np
from PIL import ImageFilter

class BlurAugmentation(AugmentationStrategy):
    """Applies blur to the image"""
    def __init__(self, blur_radius=1.0):
        self.blur_radius = blur_radius
    
    def apply(self, image, **kwargs):
        return image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))