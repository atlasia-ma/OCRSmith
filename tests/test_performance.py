import pytest
import time
from unittest.mock import Mock

class TestPerformance:
    """Performance tests for OCRSmith"""
    
    def test_placement_performance(self, sample_text_image, sample_background_image):
        """Test that placement strategies are reasonably fast"""
        from ocrsmith.core.text_placement import RandomPlacementStrategy
        
        strategy = RandomPlacementStrategy()
        
        start_time = time.time()
        for _ in range(100):
            result = strategy.place_text(sample_text_image, sample_background_image)
        end_time = time.time()
        
        # Should complete 100 placements in under 1 second
        assert (end_time - start_time) < 1.0
    
    def test_augmentation_pipeline_performance(self, sample_text_image):
        """Test augmentation pipeline performance"""
        from ocrsmith.core import AugmentationPipeline, NoiseAugmentation
        
        pipeline = AugmentationPipeline()
        pipeline.add_augmentation(NoiseAugmentation(noise_factor=0.1), probability=1.0)
        
        start_time = time.time()
        for _ in range(10):  # Fewer iterations for image processing
            result = pipeline.apply_all(sample_text_image)
        end_time = time.time()
        
        # Should complete 10 augmentations in under 2 seconds
        assert (end_time - start_time) < 2.0
        