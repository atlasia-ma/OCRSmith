from PIL import Image

from ocrsmith.core.text_placement import RandomPlacementStrategy


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_tiny_background_image(self):
        """Test placement with very small background"""
        text_img = Image.new("RGBA", (100, 50), (0, 0, 0, 255))
        tiny_bg = Image.new("RGBA", (50, 25), (255, 255, 255, 255))

        strategy = RandomPlacementStrategy(margin_x=5, margin_y=5)
        result = strategy.place_text(text_img, tiny_bg)

        # Should handle gracefully without crashing
        assert result.bbox is not None
        assert len(result.bbox) == 4

    def test_zero_size_images(self):
        """Test with zero-size images"""
        zero_img = Image.new("RGBA", (0, 0), (0, 0, 0, 255))
        normal_bg = Image.new("RGBA", (100, 100), (255, 255, 255, 255))

        strategy = RandomPlacementStrategy()
        result = strategy.place_text(zero_img, normal_bg)

        # Should handle gracefully
        assert result.bbox is not None

    def test_very_large_margins(self):
        """Test with margins larger than image"""
        text_img = Image.new("RGBA", (50, 25), (0, 0, 0, 255))
        bg_img = Image.new("RGBA", (100, 50), (255, 255, 255, 255))

        strategy = RandomPlacementStrategy(margin_x=60, margin_y=30)
        result = strategy.place_text(text_img, bg_img)

        # Should clamp to reasonable values
        assert result.bbox is not None
        assert result.bbox[0] >= 0
        assert result.bbox[1] >= 0
