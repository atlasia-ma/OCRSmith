import pytest
from unittest.mock import Mock, patch
from PIL import Image

from ocrsmith.core.text_placement import (
    PlacementResult,
    RandomPlacementStrategy,
    CenterPlacementStrategy,
)
from ocrsmith.core import PlacementManager

class TestPlacementResult:
    """Test PlacementResult class"""
    
    def test_placement_result_creation(self, sample_text_image):
        bbox = (10, 20, 110, 70)
        metadata = {'test': 'data'}
        
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
        
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 50
            
            result = strategy.place_text(sample_text_image, sample_background_image)
            
            assert isinstance(result, PlacementResult)
            assert result.bbox == (50, 50, 150, 100)
            assert result.metadata['placement_type'] == 'random'
            assert result.metadata['position'] == (50, 50)
            assert result.metadata['margins'] == (10, 10)
    
    def test_random_placement_edge_case_small_background(self):
        text_img = Image.new('RGBA', (100, 50), (0, 0, 0, 255))
        bg_img = Image.new('RGBA', (120, 60), (255, 255, 255, 255))
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
        assert result.metadata['placement_type'] == 'center'
        assert result.metadata['position'] == (50, 25)

class TestPlacementManager:
    """Test PlacementManager"""
    
    def test_register_and_get_strategy(self):
        manager = PlacementManager()
        # First register the default strategy to avoid KeyError
        default_strategy = RandomPlacementStrategy()
        manager.register_strategy('random', default_strategy)
        
        # Then register and test the specific strategy
        test_strategy = RandomPlacementStrategy()
        manager.register_strategy('test', test_strategy)
        
        assert manager.get_strategy('test') == test_strategy
    
    def test_default_strategy(self):
        manager = PlacementManager()
        default_strategy = RandomPlacementStrategy()
        manager.register_strategy('random', default_strategy)
        
        assert manager.get_strategy() == default_strategy
        assert manager.get_strategy(None) == default_strategy
    
    def test_get_strategy_fallback_to_default(self):
        manager = PlacementManager()
        default_strategy = RandomPlacementStrategy()
        manager.register_strategy('random', default_strategy)
        
        # Request non-existent strategy, should return default
        result = manager.get_strategy('nonexistent')
        assert result == default_strategy
    
    def test_place_text_integration(self, sample_text_image, sample_background_image):
        manager = PlacementManager()
        
        # Register default strategy first
        default_strategy = RandomPlacementStrategy()
        manager.register_strategy('random', default_strategy)
        
        # Register test strategy
        strategy = Mock()
        expected_result = PlacementResult(sample_text_image, (0, 0, 100, 50))
        strategy.place_text.return_value = expected_result
        manager.register_strategy('test', strategy)
        
        result = manager.place_text(sample_text_image, sample_background_image, 'test')
        
        assert result == expected_result
        strategy.place_text.assert_called_once_with(sample_text_image, sample_background_image)
    
    def test_place_text_with_default_strategy(self, sample_text_image, sample_background_image):
        manager = PlacementManager()
        
        # Mock the default strategy
        default_strategy = Mock()
        expected_result = PlacementResult(sample_text_image, (10, 10, 110, 60))
        default_strategy.place_text.return_value = expected_result
        manager.register_strategy('random', default_strategy)
        
        # Call without specifying strategy (should use default)
        result = manager.place_text(sample_text_image, sample_background_image)
        
        assert result == expected_result
        default_strategy.place_text.assert_called_once_with(sample_text_image, sample_background_image)
    
    def test_get_strategy_raises_error_when_not_found(self):
        manager = PlacementManager()
        
        with pytest.raises(ValueError, match="Strategy 'nonexistent' not found"):
            manager.get_strategy('nonexistent')
    
    def test_get_strategy_raises_error_when_default_not_found(self):
        manager = PlacementManager()
        
        with pytest.raises(ValueError, match="Strategy 'random' not found"):
            manager.get_strategy()
            