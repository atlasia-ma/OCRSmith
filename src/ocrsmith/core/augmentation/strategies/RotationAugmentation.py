# src/ocrsmith/core/augmentation/strategies/RotationAugmentation.py

import random

from ..AugmentationStrategy import AugmentationStrategy


class RotationAugmentation(AugmentationStrategy):
    """Rotates the image slightly"""

    def __init__(self, max_angle=5):
        self.max_angle = max_angle

    def apply(self, image, **kwargs):
        angle = random.uniform(-self.max_angle, self.max_angle)
        return image.rotate(angle, expand=True, fillcolor="white")
