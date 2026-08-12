# tests/core/test_augmentation_manager.py
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

# --- Import all necessary classes ---
from ocrsmith.config.schema import AppConfig, BlurAugmentationConfig, RotationAugmentationConfig
from ocrsmith.core.augmentation import (
    AugmentationContext,
    BlurAugmentation,
    BrightnessAugmentation,
    NoiseAugmentation,
    RotationAugmentation,
)
from ocrsmith.core.AugmentationManager import AugmentationManager

# --- Fixtures used by multiple test classes ---


@pytest.fixture
def sample_image():
    """Provides a simple PIL image for testing."""
    return Image.new("RGB", (100, 50), color="red")


# --- Tests for Individual Augmentation Strategies ---


class TestNoiseAugmentation:
    """Test NoiseAugmentation"""

    @patch("numpy.random.normal")
    @patch("numpy.clip")
    def test_noise_augmentation(self, mock_clip, mock_normal, sample_image):
        mock_normal.return_value = np.zeros((50, 100, 3))
        mock_clip.return_value = np.ones((50, 100, 3), dtype=np.uint8) * 128

        augmentation = NoiseAugmentation(noise_factor=0.1)
        result = augmentation.apply(sample_image)

        assert isinstance(result, Image.Image)
        mock_normal.assert_called_once()
        mock_clip.assert_called_once()


class TestBlurAugmentation:
    """Test BlurAugmentation"""

    def test_blur_augmentation(self, sample_image):
        augmentation = BlurAugmentation(blur_radius=2.0)

        with patch.object(sample_image, "filter") as mock_filter:
            mock_filter.return_value = sample_image

            result = augmentation.apply(sample_image)

            assert result == sample_image
            mock_filter.assert_called_once()


class TestBrightnessAugmentation:
    """Test BrightnessAugmentation"""

    @patch("PIL.ImageEnhance.Brightness")
    def test_brightness_augmentation(self, mock_brightness_class, sample_image):
        mock_enhancer = Mock()
        mock_enhancer.enhance.return_value = sample_image
        mock_brightness_class.return_value = mock_enhancer

        augmentation = BrightnessAugmentation(brightness_factor=1.5)
        result = augmentation.apply(sample_image)

        assert result == sample_image
        mock_brightness_class.assert_called_once_with(sample_image)
        mock_enhancer.enhance.assert_called_once_with(1.5)


class TestRotationAugmentation:
    """Test RotationAugmentation"""

    def test_rotation_augmentation(self, sample_image):
        augmentation = RotationAugmentation(max_angle=10)

        with (
            patch("random.uniform", return_value=5.0),
            patch.object(sample_image, "rotate") as mock_rotate,
        ):
            mock_rotate.return_value = sample_image

            result = augmentation.apply(sample_image)

            assert result == sample_image
            mock_rotate.assert_called_once_with(5.0, expand=True, fillcolor="white")


# --- Fixtures for AugmentationManager Tests ---


@pytest.fixture
def sample_app_config():
    """Provides a mock AppConfig with two valid augmentation configurations."""
    blur_config = BlurAugmentationConfig(type="blur", blur_radius=2.0)
    rot_config = RotationAugmentationConfig(type="rotation", max_angle=10)

    config = Mock(spec=AppConfig)
    config.augmentations = [blur_config, rot_config]
    return config


@pytest.fixture
def empty_app_config():
    """Provides a mock AppConfig with no augmentations."""
    config = Mock(spec=AppConfig)
    config.augmentations = []
    return config


# --- Tests for the AugmentationManager ---


class TestAugmentationManager:
    """Tests for the AugmentationManager class."""

    def test_initialization(self, sample_app_config):
        """Test that the manager initializes correctly with a valid config."""
        manager = AugmentationManager(sample_app_config)
        assert len(manager.augmentation_configs) == 2
        assert manager.augmentation_configs[0]["type"] == "blur"
        assert manager.augmentation_configs[1]["type"] == "rotation"

    def test_initialization_with_empty_config(self, empty_app_config):
        """Test that initialization succeeds with an empty augmentation list."""
        manager = AugmentationManager(empty_app_config)
        assert len(manager.augmentation_configs) == 0

    def test_initialization_with_unknown_type(self):
        """Test that an unknown augmentation type raises a ValueError during parsing."""
        unknown_config_model = Mock()
        unknown_config_model.model_dump.return_value = {"type": "warp", "factor": 2}

        config = Mock(spec=AppConfig)
        config.augmentations = [unknown_config_model]

        with pytest.raises(ValueError, match="Unknown augmentation type 'warp'"):
            AugmentationManager(config)

    @patch("random.choice")
    def test_get_random_augmentation(self, mock_random_choice, sample_app_config):
        """Test retrieving a random augmentation strategy."""
        manager = AugmentationManager(sample_app_config)

        mock_random_choice.return_value = manager.augmentation_configs[0]

        context = manager.get_random_augmentation()

        assert isinstance(context, AugmentationContext)
        assert isinstance(context._strategy, BlurAugmentation)
        assert context._strategy.blur_radius == 2.0

    def test_get_random_augmentation_on_empty_config(self, empty_app_config):
        """Test that getting a random augmentation from an empty config raises an error."""
        manager = AugmentationManager(empty_app_config)
        with pytest.raises(ValueError, match="No augmentation configurations available"):
            manager.get_random_augmentation()

    def test_get_augmentation_by_type(self, sample_app_config):
        """Test retrieving a specific augmentation by its type."""
        manager = AugmentationManager(sample_app_config)
        context = manager.get_augmentation_by_type("rotation")

        assert isinstance(context, AugmentationContext)
        assert isinstance(context._strategy, RotationAugmentation)
        assert context._strategy.max_angle == 10

    def test_get_augmentation_by_unknown_type(self, sample_app_config):
        """Test that getting an unknown type by name raises a ValueError."""
        manager = AugmentationManager(sample_app_config)
        with pytest.raises(ValueError, match="No augmentation of type 'warp' found"):
            manager.get_augmentation_by_type("warp")

    def test_apply_pipeline_probabilities_and_ranges(self, sample_image):
        """Pipeline applies with per-augmentation probability and samples ranges."""
        blur_cfg = BlurAugmentationConfig(type="blur", blur_radius=(1.0, 1.0), probability=1.0)
        rot_cfg = RotationAugmentationConfig(type="rotation", max_angle=(0.0, 0.0), probability=0.0)
        cfg = Mock(spec=AppConfig)
        cfg.augmentations = [blur_cfg, rot_cfg]
        cfg.augmentation_order = "random"
        manager = AugmentationManager(cfg)
        out = manager.apply_pipeline(sample_image)
        assert out is not None
