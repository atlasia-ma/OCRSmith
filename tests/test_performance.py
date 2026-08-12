import time


class TestPerformance:
    """Performance tests for OCRSmith"""

    def test_placement_performance(self, sample_text_image, sample_background_image):
        """Test that placement strategies are reasonably fast"""
        from ocrsmith.core.text_placement import RandomPlacementStrategy

        strategy = RandomPlacementStrategy()

        start_time = time.time()
        for _ in range(100):
            strategy.place_text(sample_text_image, sample_background_image)
        end_time = time.time()

        # Should complete 100 placements in under 1 second
        assert (end_time - start_time) < 1.0

    def test_noise_augmentation_performance(self, sample_text_image):
        """Basic performance smoke test for a single augmentation strategy"""
        from ocrsmith.core.augmentation.strategies import NoiseAugmentation

        aug = NoiseAugmentation(noise_factor=0.1)

        start_time = time.time()
        for _ in range(10):
            _ = aug.apply(sample_text_image)
        end_time = time.time()

        assert (end_time - start_time) < 2.0
