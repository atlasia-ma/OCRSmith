# src/ocrsmith/core/augmentation/AugmentationPipeline.py

from .augmentation import AugmentationStrategy

class AugmentationPipeline:
    """Manages and applies multiple augmentations"""
    def __init__(self):
        self.augmentations = []
    
    def add_augmentation(self, augmentation: AugmentationStrategy, probability=1.0):
        """Add augmentation with probability of being applied"""
        self.augmentations.append((augmentation, probability))
    
    def apply_all(self, image):
        """Apply all augmentations based on their probabilities"""
        import random
        result_image = image.copy()
        
        for augmentation, probability in self.augmentations:
            if random.random() < probability:
                result_image = augmentation.apply(result_image)
        
        return result_image
    