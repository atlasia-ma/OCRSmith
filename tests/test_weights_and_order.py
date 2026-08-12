from unittest.mock import patch

import pytest
from PIL import Image

from ocrsmith.config.schema import AppConfig
from ocrsmith.core.AugmentationManager import AugmentationManager
from ocrsmith.core.BackgroundManager import BackgroundManager
from ocrsmith.core.TextRenderingManager import TextRenderingManager


def build_min_config(overrides=None):
    base = {
        "fonts": [{"path": "assets/fonts/dummy.ttf"}],
        "backgrounds": [
            {"type": "solid", "color": [255, 255, 255], "weight": 0.1},
            {"type": "solid", "color": [240, 240, 240], "weight": 0.9},
        ],
        "text_renderers": [
            {"type": "horizontal", "weight": 1.0},
            {"type": "simple", "weight": 2.0},
        ],
        "text_placements": [{"type": "center"}],
        "augmentations": [],
        "layout": {"type": "simple"},
        "output": {"images_dir": "out", "metadata_file": "out.jsonl"},
        "augmentation_order": "random",
    }
    if overrides:
        base.update(overrides)
    return AppConfig.model_validate(base)


def test_background_manager_weights_passed_to_choices():
    config = build_min_config()
    manager = BackgroundManager(config)
    with patch("random.choices") as mock_choices:
        mock_choices.return_value = [manager.background_configs[0]]
        _ = manager.get_random_background()
        # Ensure weights are correctly passed
        assert mock_choices.called
        args, kwargs = mock_choices.call_args
        weights = kwargs.get("weights") or (args[1] if len(args) > 1 else None)
        assert weights is not None
        assert pytest.approx(weights[0], rel=0, abs=0.0001) == 0.1
        assert pytest.approx(weights[1], rel=0, abs=0.0001) == 0.9


def test_text_rendering_manager_registers_simple_and_horizontal():
    config = build_min_config()
    manager = TextRenderingManager(config)
    types = {c["type"] for c in manager.text_configs}
    assert "horizontal" in types and "simple" in types
    ctx = manager.get_text_renderer_by_type("simple")
    assert type(ctx._strategy).__name__.lower().startswith("simple")


def test_augmentation_pipeline_order_fixed():
    # Build config with two augmentations
    from ocrsmith.config.schema import BlurAugmentationConfig, RotationAugmentationConfig

    blur_cfg = BlurAugmentationConfig(type="blur", blur_radius=1.0, probability=1.0)
    rot_cfg = RotationAugmentationConfig(type="rotation", max_angle=0.0, probability=1.0)

    # Create a real config instead of a mock
    cfg = build_min_config({"augmentations": [blur_cfg, rot_cfg], "augmentation_order": "fixed"})

    manager = AugmentationManager(cfg)

    # Replace registry with fakes to record order
    order = []

    class FirstAug:
        def __init__(self, **kwargs):
            pass

        def apply(self, image, **kwargs):
            order.append("first")
            return image

    class SecondAug:
        def __init__(self, **kwargs):
            pass

        def apply(self, image, **kwargs):
            order.append("second")
            return image

    # Replace the strategy registry with our fake classes
    manager.strategy_registry["blur"] = FirstAug
    manager.strategy_registry["rotation"] = SecondAug

    # Also replace the parsed configs to use our fake classes
    for cfg in manager.augmentation_configs:
        if cfg["type"] == "blur":
            cfg["strategy_class"] = FirstAug
        elif cfg["type"] == "rotation":
            cfg["strategy_class"] = SecondAug

    img = Image.new("RGB", (10, 10), "white")
    out = manager.apply_pipeline(img)
    assert out is not None
    assert order == ["first", "second"]
