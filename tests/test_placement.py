from unittest.mock import patch

from PIL import Image

from ocrsmith.core.text_placement import (
    CenterPlacementStrategy,
    GridPlacementStrategy,
    PageNumberPlacementStrategy,
    PageTitlePlacementStrategy,
    PlacementResult,
    RandomPlacementStrategy,
)
from ocrsmith.core.TextPlacementManager import TextPlacementManager


class TestPlacementResult:
    """Test PlacementResult class"""

    def test_placement_result_creation(self, sample_text_image):
        bbox = (10, 20, 110, 70)
        metadata = {"test": "data"}

        result = PlacementResult(sample_text_image, bbox, metadata)

        assert result.composed_image == sample_text_image
        assert result.bbox == bbox
        assert result.metadata == metadata

    def test_placement_result_default_metadata(self, sample_text_image):
        bbox = (10, 20, 110, 70)

        result = PlacementResult(sample_text_image, bbox)

        assert result.metadata == {}


class TestRandomPlacementStrategy:
    """Test RandomPlacementStrategy"""

    def test_random_placement_within_margins(self, sample_text_image, sample_background_image):
        strategy = RandomPlacementStrategy(margin_x=10, margin_y=10)

        with patch("random.randint") as mock_randint:
            mock_randint.return_value = 50

            result = strategy.place_text(sample_text_image, sample_background_image)

            assert isinstance(result, PlacementResult)
            assert result.bbox == (50, 50, 150, 100)
            assert result.metadata["placement_type"] == "random"
            assert result.metadata["position"] == (50, 50)
            assert result.metadata["margins"] == (10, 10)

    def test_random_placement_edge_case_small_background(self):
        text_img = Image.new("RGBA", (100, 50), (0, 0, 0, 255))
        bg_img = Image.new("RGBA", (120, 60), (255, 255, 255, 255))
        strategy = RandomPlacementStrategy(margin_x=10, margin_y=10)

        result = strategy.place_text(text_img, bg_img)

        assert result.bbox[0] == 10  # Should be at margin when no space for random
        assert result.bbox[1] == 10


class TestCenterPlacementStrategy:
    """Test CenterPlacementStrategy"""

    def test_center_placement(self, sample_text_image, sample_background_image):
        strategy = CenterPlacementStrategy()

        result = strategy.place_text(sample_text_image, sample_background_image)

        # Background: 200x100, Text: 100x50
        # Expected center: (50, 25)
        assert result.bbox == (50, 25, 150, 75)
        assert result.metadata["placement_type"] == "center"
        assert result.metadata["position"] == (50, 25)


class TestGridPlacementStrategy:
    """Test GridPlacementStrategy"""

    def test_grid_placement(self, sample_text_image, sample_background_image):
        strategy = GridPlacementStrategy(rows=2, cols=2)

        # First placement (0,0)
        result1 = strategy.place_text(sample_text_image, sample_background_image)
        assert result1.metadata["grid_cell"] == (0, 0)
        assert result1.metadata["cell_position"] == 0

        # Second placement (0,1)
        result2 = strategy.place_text(sample_text_image, sample_background_image)
        assert result2.metadata["grid_cell"] == (0, 1)
        assert result2.metadata["cell_position"] == 1


class TestPageTitlePlacementStrategy:
    """Test PageTitlePlacementStrategy"""

    def test_page_title_placement(self, sample_text_image, sample_background_image):
        strategy = PageTitlePlacementStrategy(top_margin=30, side_margin=10)

        result = strategy.place_text(sample_text_image, sample_background_image)

        # Should be centered horizontally, with top margin
        expected_x = (200 - 100) // 2  # 50
        expected_y = 30

        assert result.bbox == (expected_x, expected_y, expected_x + 100, expected_y + 50)
        assert result.metadata["placement_type"] == "page_title"
        assert result.metadata["content_type"] == "title"


class TestPageNumberPlacementStrategy:
    """Test PageNumberPlacementStrategy"""

    def test_page_number_placement(self, sample_text_image, sample_background_image):
        strategy = PageNumberPlacementStrategy(bottom_margin=20, right_margin=15)

        result = strategy.place_text(sample_text_image, sample_background_image)

        # Should be bottom-right
        expected_x = 200 - 100 - 15  # 85
        expected_y = 100 - 50 - 20  # 30

        assert result.bbox == (expected_x, expected_y, expected_x + 100, expected_y + 50)
        assert result.metadata["placement_type"] == "page_number"
        assert result.metadata["content_type"] == "page_number"


class TestTextPlacementManager:
    """Tests for TextPlacementManager registry-driven selection"""

    def test_get_random_uses_config_params(self, monkeypatch):
        # Build a minimal config dict matching schema
        from ocrsmith.config.schema import AppConfig

        config = AppConfig.model_validate(
            {
                "fonts": [{"path": "assets/fonts/dummy.ttf"}],
                "backgrounds": [{"type": "solid", "color": [255, 255, 255]}],
                "text_renderers": [{"type": "horizontal"}],
                "text_placements": [{"type": "random", "margin": 13}],
                "augmentations": [],
                "layout": {"type": "simple"},
                "output": {"images_dir": "out", "metadata_file": "out.jsonl"},
            }
        )

        manager = TextPlacementManager(config)
        ctx = manager.get_random_placement()
        strategy = ctx._strategy
        assert isinstance(strategy, RandomPlacementStrategy)
        assert strategy.margin_x == 13 and strategy.margin_y == 13
