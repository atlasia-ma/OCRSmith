# tests/core/test_ocrsmith_engine.py
import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml
from PIL import Image

# --- Import all necessary classes ---
from ocrsmith.config.schema import AppConfig
from ocrsmith.core.engine import OCRSmithEngine
from ocrsmith.core.text_placement import PlacementResult

# --- Fixtures ---


@pytest.fixture
def app_config():
    """
    Provides a real, valid AppConfig object loaded from a YAML string.
    Using the default 'function' scope ensures each test gets a fresh
    instance, preventing state leakage between tests.
    """
    # This YAML is based on the detailed config you provided.
    # Using a real config structure avoids mocking issues with Pydantic models.
    yaml_string = """
    fonts:
      - path: 'assets/fonts/dummy.ttf'
        size: 24
    text_data:
      source_type: 'csv'
      source_path: 'assets/text_data/dummy.csv'
      text_column: 'text'
    backgrounds:
      - type: solid
        color: [255, 255, 255]
      - type: gradient
        start_color: [220, 220, 255]
        end_color: [255, 220, 220]
    text_renderers:
      - type: horizontal
    text_placements:
      - type: random
        margin: 50
    augmentations: []
    layout:
      type: simple
    output:
      images_dir: 'outputs/images'
      metadata_file: 'outputs/metadata.jsonl'
    """
    config_dict = yaml.safe_load(yaml_string)
    return AppConfig.model_validate(config_dict)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# --- Test Class for the Engine ---


# Patch all managers at the class level to ensure they are mocked for every test.
# This prevents real managers from being initialized with mock configs.
@patch("ocrsmith.core.engine.BackgroundManager")
@patch("ocrsmith.core.engine.TextRenderingManager")
@patch("ocrsmith.core.engine.TextPlacementManager")
@patch("ocrsmith.core.engine.AugmentationManager")
@patch("ocrsmith.core.engine.FontManager")
@patch("ocrsmith.core.engine.TextDataManager")
class TestOCRSmithEngine:
    """Tests for the main OCRSmithEngine class with all dependencies mocked."""

    def test_engine_initialization(
        self, mock_data, mock_font, mock_aug, mock_placement, mock_render, mock_bg, app_config
    ):
        """Test that the engine initializes all its managers correctly."""
        engine = OCRSmithEngine(app_config)

        assert engine.config == app_config
        mock_bg.assert_called_once_with(app_config)
        mock_render.assert_called_once_with(app_config)
        mock_placement.assert_called_once_with(app_config)
        mock_aug.assert_called_once_with(app_config)
        mock_font.assert_called_once_with(app_config)
        mock_data.assert_called_once()

    def test_generate_sample(
        self, mock_data, mock_font, mock_aug, mock_placement, mock_render, mock_bg, app_config
    ):
        """Test the logic of generating a single sample image."""
        # Ensure augmentations list is not empty for this specific test
        # This modification is safe because the fixture has function scope.
        app_config.augmentations = [Mock()]

        engine = OCRSmithEngine(app_config)

        # --- Setup Mocks for all manager instances ---
        engine.font_manager.get_random_font.return_value = Mock()

        mock_text_image = Image.new("RGBA", (100, 50))
        mock_render_context = Mock()
        mock_render_context.render_text.return_value = (mock_text_image, None, (100, 50))
        engine.text_rendering_manager.get_random_text_renderer.return_value = mock_render_context

        mock_background_image = Image.new("RGB", (200, 150))
        mock_bg_context = Mock()
        mock_bg_context.render.return_value = mock_background_image
        engine.background_manager.get_random_background.return_value = mock_bg_context

        mock_composed_image = Image.new("RGB", (200, 150))
        mock_placement_result = PlacementResult(
            composed_image=mock_composed_image, bbox=(10, 20, 110, 70), metadata={"placement_type": "random"}
        )
        mock_placement_context = Mock()
        mock_placement_context.place_text.return_value = mock_placement_result
        engine.text_placement_manager.get_random_placement.return_value = mock_placement_context

        # Pipeline-based augmentation now used in engine; return composed image
        engine.augmentation_manager.apply_pipeline.return_value = mock_composed_image

        # --- Call the method ---
        image, text, bbox, metadata = engine.generate_sample(text="Hello World")

        # --- Assertions ---
        assert isinstance(image, Image.Image)
        assert text == "Hello World"
        assert bbox == (10, 20, 110, 70)
        assert metadata.get("placement_type") == "random"

    def test_generate_sample_no_text_raises_error(
        self, mock_data, mock_font, mock_aug, mock_placement, mock_render, mock_bg, app_config
    ):
        """Test that generate_sample raises a ValueError if text is empty."""
        engine = OCRSmithEngine(app_config)
        with pytest.raises(ValueError, match="Text must be provided to generate a sample."):
            engine.generate_sample(text="")

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_generate_test_dataset(
        self,
        mock_json_dump,
        mock_file,
        mock_makedirs,
        mock_bg,
        mock_render,
        mock_placement,
        mock_aug,
        mock_font,
        mock_data,
        app_config,
        temp_dir,
    ):
        """Test the full dataset generation loop, including file I/O."""
        engine = OCRSmithEngine(app_config)

        mock_image = Mock(spec=Image.Image)
        engine.generate_sample = Mock(return_value=(mock_image, "test", (0, 0, 10, 10), {}))

        engine.text_data_manager.load_from_source = Mock()
        engine.text_data_manager.get_random_text.return_value = "test"

        # --- Call the method ---
        annotations = engine.generate_test_dataset(num_samples=3, output_dir=temp_dir)

        # --- Assertions ---
        assert engine.text_data_manager.load_from_source.called
        assert engine.generate_sample.call_count == 3
        mock_makedirs.assert_called_once_with(temp_dir, exist_ok=True)

        expected_save_paths = [
            os.path.join(temp_dir, "sample_0000.png"),
            os.path.join(temp_dir, "sample_0001.png"),
            os.path.join(temp_dir, "sample_0002.png"),
        ]
        saved_paths = [call.args[0] for call in mock_image.save.call_args_list]
        assert saved_paths == expected_save_paths

        assert len(annotations) == 3
        assert annotations[0]["text"] == "test"
        mock_json_dump.assert_called_once()
