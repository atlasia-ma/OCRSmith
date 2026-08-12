"""Contract for degradations.

The invariant that matters more than any other: **pixels and boxes move together**. A
geometric degradation that shifts ink but leaves the annotation behind produces a dataset
that looks fine and trains a detector to be systematically wrong, so the tests here check
that the annotation actually tracks the ink rather than merely that it changed.
"""

import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocrsmith.core.degradations import (
    Bleedthrough,
    Blur,
    Brightness,
    Contrast,
    DegradationPipeline,
    Downscale,
    Folds,
    GaussianNoise,
    Glare,
    InkErosion,
    InkSpread,
    JpegArtifacts,
    MotionBlur,
    PaperGrain,
    PerspectiveWarp,
    Rotation,
    Shadow,
    Stains,
    Vignette,
    build_preset,
    preset_names,
)
from ocrsmith.domain import BBox, Line, Page, Region, RegionType, Word

PHOTOMETRIC = [
    GaussianNoise(sigma=8.0),
    PaperGrain(strength=10.0),
    Blur(radius=1.0),
    MotionBlur(length=5, angle=30.0),
    Brightness(factor=0.8),
    Contrast(factor=1.4),
    JpegArtifacts(quality=40),
    Downscale(scale=0.5),
    InkSpread(size=3),
    InkErosion(size=3),
    Bleedthrough(strength=0.15),
    Shadow(strength=0.3),
    Vignette(strength=0.3),
    Glare(strength=0.3),
    Stains(count=2, strength=0.2),
    Folds(count=2, strength=0.2),
]


def ink_box(image: Image.Image, threshold: int = 128) -> BBox:
    """Bounding box of the dark pixels — where the ink actually ended up."""
    grey = np.asarray(image.convert("L"))
    dark = grey < threshold
    ys, xs = np.nonzero(dark)
    if len(xs) == 0:
        raise AssertionError("image contains no ink")
    return BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


@pytest.fixture
def marked_page():
    """A white page with one black rectangle, annotated exactly where the rectangle is."""
    # A slightly off-white paper tint rather than pure white: a saturated page hides
    # tone-mapping degradations, which would make the tests pass for the wrong reason.
    image = Image.new("RGB", (400, 300), (247, 244, 238))
    box = BBox(120, 90, 260, 170)
    ImageDraw.Draw(image).rectangle(box.as_int(), fill=(12, 12, 16))
    word = Word("block", box)
    line = Line("block", box, (word,))
    page = Page(400, 300, (Region(RegionType.PARAGRAPH, box, (line,)),))
    return image, page


@pytest.fixture
def rng():
    return random.Random(20260812)


class TestPhotometricDegradations:
    @pytest.mark.parametrize("degradation", PHOTOMETRIC, ids=lambda d: d.name)
    def test_geometry_is_untouched(self, degradation, marked_page, rng):
        image, page = marked_page

        result, new_page, _ = degradation.apply(image, page, rng)

        assert result.size == image.size
        assert new_page.regions[0].bbox == page.regions[0].bbox

    @pytest.mark.parametrize("degradation", PHOTOMETRIC, ids=lambda d: d.name)
    def test_pixels_actually_change(self, degradation, marked_page, rng):
        image, page = marked_page

        result, _, _ = degradation.apply(image, page, rng)

        assert np.asarray(result.convert("RGB")).shape == np.asarray(image).shape
        assert not np.array_equal(np.asarray(result.convert("RGB")), np.asarray(image))

    @pytest.mark.parametrize("degradation", PHOTOMETRIC, ids=lambda d: d.name)
    def test_sampled_parameters_are_reported(self, degradation, marked_page, rng):
        image, page = marked_page

        _, _, params = degradation.apply(image, page, rng)

        assert isinstance(params, dict)
        assert params  # provenance must be able to explain the sample

    def test_ink_spread_darkens_and_erosion_lightens(self, marked_page, rng):
        image, page = marked_page
        original = np.asarray(image.convert("L")).mean()

        spread, _, _ = InkSpread(size=3).apply(image, page, rng)
        eroded, _, _ = InkErosion(size=3).apply(image, page, rng)

        assert np.asarray(spread.convert("L")).mean() < original
        assert np.asarray(eroded.convert("L")).mean() > original

    def test_jpeg_quality_is_clamped_to_a_valid_range(self, marked_page, rng):
        image, page = marked_page

        result, _, _ = JpegArtifacts(quality=999).apply(image, page, rng)

        assert result.size == image.size


class TestRotation:
    def test_boxes_follow_the_ink(self, marked_page, rng):
        image, page = marked_page

        rotated, new_page, params = Rotation(max_angle=8.0).apply(image, page, rng)

        assert abs(params["angle"]) > 0
        drawn = ink_box(rotated)
        annotated = new_page.regions[0].bbox
        assert annotated.iou(drawn) > 0.85

    def test_canvas_grows_to_hold_the_rotated_page(self, marked_page, rng):
        image, page = marked_page

        rotated, new_page, _ = Rotation(max_angle=8.0).apply(image, page, rng)

        assert rotated.size[0] >= image.size[0]
        assert (new_page.width, new_page.height) == rotated.size

    def test_boxes_stay_inside_the_new_canvas(self, marked_page, rng):
        image, page = marked_page

        rotated, new_page, _ = Rotation(max_angle=10.0).apply(image, page, rng)

        for region in new_page.regions:
            assert new_page.bbox.contains(region.bbox)

    def test_words_are_carried_along_with_their_line(self, marked_page, rng):
        image, page = marked_page

        _, new_page, _ = Rotation(max_angle=6.0).apply(image, page, rng)

        line = new_page.regions[0].lines[0]
        assert line.words
        assert line.bbox.contains(line.words[0].bbox)

    def test_a_rotated_word_gains_a_polygon(self, marked_page, rng):
        image, page = marked_page

        _, new_page, _ = Rotation(max_angle=6.0).apply(image, page, rng)

        assert new_page.regions[0].lines[0].words[0].polygon is not None

    def test_zero_angle_is_a_no_op(self, marked_page, rng):
        image, page = marked_page

        result, new_page, params = Rotation(max_angle=0.0).apply(image, page, rng)

        assert params["angle"] == 0.0
        assert result.size == image.size
        assert new_page.regions[0].bbox == page.regions[0].bbox


class TestPerspectiveWarp:
    def test_boxes_follow_the_ink(self, marked_page, rng):
        image, page = marked_page

        warped, new_page, _ = PerspectiveWarp(magnitude=0.05).apply(image, page, rng)

        drawn = ink_box(warped)
        annotated = new_page.regions[0].bbox
        assert annotated.iou(drawn) > 0.8

    def test_the_annotation_moves_at_all(self, marked_page, rng):
        image, page = marked_page

        _, new_page, _ = PerspectiveWarp(magnitude=0.06).apply(image, page, rng)

        assert new_page.regions[0].bbox != page.regions[0].bbox

    def test_corners_are_reported_for_provenance(self, marked_page, rng):
        image, page = marked_page

        _, _, params = PerspectiveWarp(magnitude=0.04).apply(image, page, rng)

        assert len(params["corners"]) == 4

    def test_boxes_stay_inside_the_warped_canvas(self, marked_page, rng):
        image, page = marked_page

        warped, new_page, _ = PerspectiveWarp(magnitude=0.06).apply(image, page, rng)

        assert (new_page.width, new_page.height) == warped.size
        for region in new_page.regions:
            assert new_page.bbox.contains(region.bbox)

    def test_a_zero_magnitude_warp_leaves_geometry_alone(self, marked_page, rng):
        image, page = marked_page

        _, new_page, _ = PerspectiveWarp(magnitude=0.0).apply(image, page, rng)

        assert new_page.regions[0].bbox.iou(page.regions[0].bbox) > 0.99


class TestPipeline:
    def test_records_what_it_applied(self, marked_page, rng):
        pipeline = DegradationPipeline([Blur(radius=1.0), GaussianNoise(sigma=5.0)])
        image, page = marked_page

        _, _, records = pipeline.apply(image, page, rng)

        assert [record.name for record in records] == ["Blur", "GaussianNoise"]
        assert records[0].to_dict()["radius"] == 1.0

    def test_probability_zero_never_fires(self, marked_page, rng):
        pipeline = DegradationPipeline([Blur(radius=1.0, probability=0.0)])
        image, page = marked_page

        result, _, records = pipeline.apply(image, page, rng)

        assert records == ()
        assert np.array_equal(np.asarray(result), np.asarray(image))

    def test_an_empty_pipeline_is_the_identity(self, marked_page, rng):
        image, page = marked_page

        result, new_page, records = DegradationPipeline().apply(image, page, rng)

        assert records == ()
        assert new_page is page
        assert result is image

    def test_same_seed_gives_the_same_result(self, marked_page):
        image, page = marked_page
        pipeline = build_preset("photo")

        first, first_page, first_records = pipeline.apply(image, page, random.Random(3))
        second, second_page, second_records = pipeline.apply(image, page, random.Random(3))

        assert np.array_equal(np.asarray(first), np.asarray(second))
        assert first_page.to_dict() == second_page.to_dict()
        assert [r.to_dict() for r in first_records] == [r.to_dict() for r in second_records]

    def test_different_seeds_diverge(self, marked_page):
        image, page = marked_page
        pipeline = build_preset("photo")

        first, _, _ = pipeline.apply(image, page, random.Random(1))
        second, _, _ = pipeline.apply(image, page, random.Random(2))

        assert not np.array_equal(np.asarray(first), np.asarray(second))

    def test_order_is_preserved_unless_shuffling_is_asked_for(self, marked_page, rng):
        ordered = DegradationPipeline([Blur(radius=1.0), Brightness(factor=0.9), Contrast(factor=1.1)])

        _, _, records = ordered.apply(*marked_page, rng)

        assert [r.name for r in records] == ["Blur", "Brightness", "Contrast"]


class TestPresets:
    def test_every_preset_is_named(self):
        assert set(preset_names()) == {"clean", "scan", "photo", "fax", "archive"}

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown degradation preset"):
            build_preset("underwater")

    def test_clean_leaves_the_page_untouched(self, marked_page, rng):
        image, page = marked_page

        result, new_page, records = build_preset("clean").apply(image, page, rng)

        assert records == ()
        assert result is image
        assert new_page is page

    @pytest.mark.parametrize("name", ["scan", "photo", "fax", "archive"])
    def test_presets_keep_the_annotation_consistent_with_the_canvas(self, name, marked_page):
        image, page = marked_page

        result, new_page, _ = build_preset(name).apply(image, page, random.Random(11))

        assert (new_page.width, new_page.height) == result.size
        for region in new_page.regions:
            assert new_page.bbox.contains(region.bbox)

    @pytest.mark.parametrize("name", ["scan", "photo", "fax", "archive"])
    def test_the_annotation_still_points_at_the_ink(self, name, marked_page):
        image, page = marked_page

        result, new_page, _ = build_preset(name).apply(image, page, random.Random(5))

        # Shadow and vignette darken the paper too, so a global ink threshold is not a
        # fair measure here. What must survive every preset is the relationship: whatever
        # the annotation points at is still markedly darker than the page around it.
        grey = np.asarray(result.convert("L"), dtype=np.float32)
        x0, y0, x1, y1 = new_page.regions[0].bbox.as_int()
        inside = grey[max(0, y0) : y1, max(0, x0) : x1]

        assert inside.size > 0
        assert inside.mean() < grey.mean() - 20

    def test_presets_are_independent_instances(self):
        first = build_preset("scan")
        first.degradations.clear()

        assert len(build_preset("scan")) > 0
