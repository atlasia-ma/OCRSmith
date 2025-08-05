# src/ocrsmith/core/augmentation/augmenters/NoiseAugmentation.py

from ..AugmentationStrategy import AugmentationStrategy

import numpy as np
from PIL import Image

class NoiseAugmentation(AugmentationStrategy):
    """Adds noise to the image"""
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
    
    def apply(self, image, **kwargs):    
        img_array = np.array(image)
        noise = np.random.normal(0, self.noise_factor * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    