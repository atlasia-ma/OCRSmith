"""Contract for the shared working buffer and the fused shading field.

Both exist for speed, and both would be worth nothing if they changed the pixels. These
tests pin the properties that make them safe to use: a fused field is the same transform
as the stages it replaces, and the buffer does not quantise between stages.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from ocrsmith.core.degradations import DegradationPipeline
from ocrsmith.core.degradations.base import (
    ArrayDegradation,
    FieldDegradation,
    PhotometricDegradation,
)
from ocrsmith.core.degradations.buffer import Surface
from ocrsmith.domain.annotations import Page


def page_for(image: Image.Image) -> Page:
    return Page(image.width, image.height, ())


def noise_image(width: int = 160, height: int = 96, seed: int = 0) -> Image.Image:
    generator = np.random.default_rng(seed)
    return Image.fromarray(generator.integers(40, 215, (height, width, 3), dtype=np.uint8), "RGB")


class TestSurface:
    def test_round_trips_an_image_unchanged(self):
        image = noise_image()
        surface = Surface.of(image)

        assert np.array_equal(np.asarray(surface.as_image()), np.asarray(image))

    def test_array_view_is_writable(self):
        surface = Surface.of(noise_image())
        array = surface.as_array()
        array += 1.0  # must not raise on a read-only buffer

        assert array.dtype == np.float32

    def test_clips_only_when_becoming_an_image(self):
        surface = Surface.from_array(np.full((4, 4, 3), -50.0, dtype=np.float32))

        # The array still holds the out-of-range value...
        assert surface.as_array().min() == pytest.approx(-50.0)
        # ...and only the conversion to bytes clamps it.
        assert np.asarray(surface.as_image()).min() == 0

    def test_values_pushed_out_of_range_can_come_back(self):
        """A shadow then a glare must not lose what the shadow drove below zero.

        This is the whole reason intermediate stages share a float buffer instead of
        handing each other 8-bit images.
        """
        surface = Surface.from_array(np.full((4, 4, 3), 10.0, dtype=np.float32))
        array = surface.as_array()
        array -= 60.0  # a shadow
        array += 90.0  # a glare

        assert np.asarray(Surface.from_array(array).as_image())[0, 0, 0] == 40

    def test_rejects_being_built_from_both_or_neither(self):
        with pytest.raises(ValueError):
            Surface()
        with pytest.raises(ValueError):
            Surface(image=noise_image(), array=np.zeros((2, 2, 3), dtype=np.float32))


class _Scale(FieldDegradation):
    """A constant gain, as a field."""

    def __init__(self, factor: float):
        self.factor = factor
        self.probability = 1.0

    def field(self, width, height, rng):
        return np.full((height, width), self.factor, dtype=np.float32), None, {}


class _Offset(FieldDegradation):
    """A constant bias, as a field."""

    def __init__(self, amount: float):
        self.amount = amount
        self.probability = 1.0

    def field(self, width, height, rng):
        return None, np.full((height, width), self.amount, dtype=np.float32), {}


class _Mark(ArrayDegradation):
    """A non-field array stage, to prove fusing stops at one."""

    def shade(self, array, rng):
        array[0, 0] = 0.0
        return array, {}


class TestFieldFusion:
    def test_fused_gain_and_bias_match_sequential_application(self):
        """`x*g1 + b1` then `x*g2 + b2` must equal the single pair the pipeline fuses.

        Checked against the algebra rather than against a second implementation, so the
        test fails if the fold forgets to carry the first bias through the second gain —
        which is the one mistake this composition invites.
        """
        image = Image.new("RGB", (64, 32), (100, 100, 100))
        pipeline = DegradationPipeline([_Scale(0.5), _Offset(20.0), _Scale(2.0)])

        result, _page, records = pipeline.apply(image, page_for(image), random.Random(1))

        # 100 * 0.5 = 50; + 20 = 70; * 2 = 140.
        assert np.asarray(result)[16, 32, 0] == pytest.approx(140, abs=1)
        assert len(records) == 3

    def test_a_non_field_stage_splits_the_run(self):
        image = Image.new("RGB", (64, 32), (100, 100, 100))
        pipeline = DegradationPipeline([_Scale(0.5), _Mark(), _Offset(20.0)])

        result, _page, records = pipeline.apply(image, page_for(image), random.Random(1))

        assert np.asarray(result)[16, 32, 0] == pytest.approx(70, abs=1)
        assert len(records) == 3

    def test_a_field_stage_used_alone_matches_the_pipeline(self):
        image = Image.new("RGB", (64, 32), (100, 100, 100))

        alone, _ = _Scale(0.5).transform(image, random.Random(0))
        through, _page, _ = DegradationPipeline([_Scale(0.5)]).apply(image, page_for(image), random.Random(0))

        assert np.array_equal(np.asarray(alone), np.asarray(through))

    def test_probability_zero_contributes_no_field(self):
        image = Image.new("RGB", (64, 32), (100, 100, 100))
        skipped = _Scale(0.5)
        skipped.probability = 0.0

        result, _page, records = DegradationPipeline([skipped]).apply(
            image, page_for(image), random.Random(0)
        )

        assert np.asarray(result)[0, 0, 0] == 100
        assert records == ()


class TestRealFieldsStayFaithful:
    """The shipped field stages are built at reduced resolution; check they still land."""

    @pytest.mark.parametrize("name", ["Shadow", "Vignette", "Glare", "Stains", "Folds"])
    def test_field_stage_changes_the_image_without_destroying_it(self, name):
        from ocrsmith.core.degradations import photometric

        stage = getattr(photometric, name)(probability=1.0)
        image = noise_image(240, 160)

        result, _page, _ = DegradationPipeline([stage]).apply(image, page_for(image), random.Random(7))
        before = np.asarray(image, dtype=np.float32)
        after = np.asarray(result, dtype=np.float32)

        assert result.size == image.size
        assert not np.array_equal(before, after), f"{name} did nothing"
        # A shading field is smooth and bounded; none of these may black the page out.
        assert abs(after.mean() - before.mean()) < 120

    def test_a_field_is_smooth_enough_to_survive_being_built_small(self):
        """The reduced-resolution grid must not alias the sharpest field we draw.

        `Folds` has the narrowest feature of the five, so it is the one that would show
        banding first if `FIELD_DIVISOR` were raised.
        """
        from ocrsmith.core.degradations.photometric import Folds

        flat = Image.new("RGB", (512, 512), (200, 200, 200))
        result, _page, _ = DegradationPipeline([Folds(count=1, strength=0.3)]).apply(
            flat, page_for(flat), random.Random(3)
        )
        array = np.asarray(result, dtype=np.float32)[..., 0]

        # No adjacent pair of rows or columns may jump by more than a few levels.
        assert np.abs(np.diff(array, axis=0)).max() < 12
        assert np.abs(np.diff(array, axis=1)).max() < 12


class TestMorphologicalStagesStillSeeThePage:
    """A photometric stage that overrides `apply` must not be routed to `transform`.

    `InkSpread` and `InkErosion` read the body stroke width off the annotation to decide
    whether their kernel would erase the text. Dispatching them as plain photometric
    stages would skip that guard silently, and the symptom would be an unreadable page
    with a complete label — the exact failure this project is built to prevent.
    """

    def test_erosion_is_skipped_when_the_strokes_are_too_thin(self):
        from ocrsmith.core.degradations.photometric import InkErosion
        from ocrsmith.domain.annotations import Line, Region, RegionType
        from ocrsmith.domain.geometry import BBox

        image = noise_image(120, 60)
        # A 10-pixel line box implies a stem under a pixel wide.
        line = Line("x", BBox(0, 0, 40, 10), ())
        page = Page(120, 60, (Region(RegionType.PARAGRAPH, BBox(0, 0, 40, 10), (line,)),))

        _result, _page, records = DegradationPipeline([InkErosion(size=3)]).apply(
            image, page, random.Random(0)
        )

        assert records[0].params["size"] == 0
        assert "too thin" in records[0].params["skipped"]

    def test_it_is_still_a_photometric_degradation(self):
        from ocrsmith.core.degradations.photometric import InkErosion

        assert isinstance(InkErosion(), PhotometricDegradation)
