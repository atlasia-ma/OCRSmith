"""Regression tests for rendering without explicit layout constraints.

`layout.max_width` / `layout.max_height` are optional in the config schema, so the
renderer must cope with `None` instead of raising `TypeError` while comparing pixel
widths against a missing bound.
"""

import pytest
from PIL import ImageFont

from ocrsmith.core.text_renderers.strategies.HorizontalRenderingStrategy import (
    HorizontalRenderingStrategy,
    wrap_text_by_pixels,
)


@pytest.fixture
def font():
    return ImageFont.load_default()


def test_wrap_without_constraints_keeps_all_words(font):
    lines = wrap_text_by_pixels(["alpha beta gamma"], font, None, None, spacing=1)

    assert " ".join(lines).split() == ["alpha", "beta", "gamma"]


def test_render_without_constraints_produces_an_image(font):
    strategy = HorizontalRenderingStrategy()

    image, mask, (width, height) = strategy.render_text(font, "hello world")

    assert width > 0 and height > 0
    assert image.size == mask.size


def test_render_whitespace_only_text_does_not_crash(font):
    strategy = HorizontalRenderingStrategy()

    image, _, (width, height) = strategy.render_text(font, "   ")

    assert image.size == (width, height)
    assert width >= 1 and height >= 1
