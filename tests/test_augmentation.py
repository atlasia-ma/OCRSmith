import pytest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from ocrsmith.core.augmentation import (
    NoiseAugmentation,
    BlurAugmentation,
    RotationAugmentation
)

from ocrsmith.core.AugmentationPipeline import AugmentationPipeline

class TestNoiseAugmentation:
    """Test NoiseAugmentation"""
    
    @patch('numpy.random.normal')
    @patch('numpy.clip')
    def test_noise_augmentation(self, mock_clip, mock_normal, sample_text_image):
        mock_normal.return_value = np.zeros((50, 100, 4))
        mock_clip.return_value = np.ones((50, 100, 4), dtype=np.uint8) * 128
        
        augmentation = NoiseAugmentation(noise_factor=0.1)
        result = augmentation.apply(sample_text_image)
        
        assert isinstance(result, Image.Image)
        mock_normal.assert_called_once()
        mock_clip.assert_called_once()

class TestBlurAugmentation:
    """Test BlurAugmentation"""
    
    def test_blur_augmentation(self, sample_text_image):
        augmentation = BlurAugmentation(blur_radius=2.0)
        
        with patch.object(sample_text_image, 'filter') as mock_filter:
            mock_filter.return_value = sample_text_image
            
            result = augmentation.apply(sample_text_image)
            
            assert result == sample_text_image
            mock_filter.assert_called_once()

class TestRotationAugmentation:
    """Test RotationAugmentation"""
    
    def test_rotation_augmentation(self, sample_text_image):
        augmentation = RotationAugmentation(max_angle=10)
        
        with patch('random.uniform', return_value=5.0):
            with patch.object(sample_text_image, 'rotate') as mock_rotate:
                mock_rotate.return_value = sample_text_image
                
                result = augmentation.apply(sample_text_image)
                
                assert result == sample_text_image
                mock_rotate.assert_called_once_with(5.0, expand=True, fillcolor='white')

class TestAugmentationPipeline:
    """Test AugmentationPipeline"""
    
    def test_add_augmentation(self):
        pipeline = AugmentationPipeline()
        augmentation = Mock()
        
        pipeline.add_augmentation(augmentation, probability=0.5)
        
        assert len(pipeline.augmentations) == 1
        assert pipeline.augmentations[0] == (augmentation, 0.5)
    
    @patch('random.random')
    def test_apply_all_with_probability(self, mock_random, sample_text_image):
        pipeline = AugmentationPipeline()
        
        aug1 = Mock()
        aug1.apply.return_value = sample_text_image
        aug2 = Mock()
        aug2.apply.return_value = sample_text_image
        
        pipeline.add_augmentation(aug1, probability=0.8)
        pipeline.add_augmentation(aug2, probability=0.3)
        
        # First aug applied (0.5 < 0.8), second not applied (0.9 > 0.3)
        mock_random.side_effect = [0.5, 0.9]
        
        result = pipeline.apply_all(sample_text_image)
        
        aug1.apply.assert_called_once()
        aug2.apply.assert_not_called()
        assert result == sample_text_image
        