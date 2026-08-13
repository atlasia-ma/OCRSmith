"""Contract for physical degradations.

Paper is not flat and light is not even. These model both — and, being displacement
fields rather than colour transforms, they must carry the annotation with them. The
central test measures where the ink actually ended up and requires the boxes to be there.
"""

import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocrsmith.core.degradations import (
    IlluminationField,
    PageCurl,
    Wrinkles,
    build_preset,
)
from ocrsmith.core.degradations.photometric import _axes
from ocrsmith.domain import BBox, Line, Page, Region, RegionType, Word


def ink_box(image: Image.Image, threshold: int = 120) -> BBox:
    grey = np.asarray(image.convert("L"))
    ys, xs = np.nonzero(grey < threshold)
    if len(xs) == 0:
        raise AssertionError("no ink survived")
    return BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


@pytest.fixture
def striped_page():
    """Ruled text-like stripes, so ink and paper both appear inside the region."""
    image = Image.new("RGB", (400, 300), (248, 246, 240))
    box = BBox(120, 90, 260, 170)
    draw = ImageDraw.Draw(image)
    y = box.y0 + 2
    while y < box.y1 - 2:
        draw.rectangle((box.x0, y, box.x1, y + 3), fill=(10, 10, 10))
        y += 6
    word = Word("block", box)
    line = Line("block", box, (word,))
    page = Page(400, 300, (Region(RegionType.PARAGRAPH, box, (line,)),))
    return image, page


@pytest.fixture
def rng():
    return random.Random(20260813)


DEFORMATIONS = [PageCurl(strength=0.09), Wrinkles(strength=6.0)]


class TestDeformationsCarryTheAnnotation:
    @pytest.mark.parametrize("degradation", DEFORMATIONS, ids=lambda d: d.name)
    def test_boxes_follow_the_ink(self, degradation, striped_page, rng):
        image, page = striped_page

        warped, new_page, _ = degradation.apply(image, page, rng)

        assert new_page.regions[0].bbox.iou(ink_box(warped)) > 0.85

    @pytest.mark.parametrize("degradation", DEFORMATIONS, ids=lambda d: d.name)
    def test_the_annotation_actually_moves(self, degradation, striped_page, rng):
        image, page = striped_page

        _, new_page, _ = degradation.apply(image, page, rng)

        assert new_page.regions[0].bbox != page.regions[0].bbox

    @pytest.mark.parametrize("degradation", DEFORMATIONS, ids=lambda d: d.name)
    def test_boxes_stay_inside_the_canvas(self, degradation, striped_page, rng):
        image, page = striped_page

        warped, new_page, _ = degradation.apply(image, page, rng)

        assert (new_page.width, new_page.height) == warped.size
        for region in new_page.regions:
            assert new_page.bbox.contains(region.bbox)

    @pytest.mark.parametrize("degradation", DEFORMATIONS, ids=lambda d: d.name)
    def test_words_gain_polygons(self, degradation, striped_page, rng):
        image, page = striped_page

        _, new_page, _ = degradation.apply(image, page, rng)

        assert new_page.regions[0].lines[0].words[0].polygon is not None

    @pytest.mark.parametrize("degradation", DEFORMATIONS, ids=lambda d: d.name)
    def test_parameters_are_reported(self, degradation, striped_page, rng):
        _, _, params = degradation.apply(*striped_page, rng)

        assert params and "strength" in params


class TestPageCurl:
    def test_the_curled_edge_is_recorded(self, striped_page, rng):
        _, _, params = PageCurl(strength=0.08, edge="left").apply(*striped_page, rng)

        assert params["edge"] == "left"

    def test_curling_darkens_towards_the_bend(self, striped_page, rng):
        image, page = striped_page

        curled, _, _ = PageCurl(strength=0.12, edge="left").apply(image, page, rng)

        grey = np.asarray(curled.convert("L"), dtype=np.float32)
        assert grey[:, :40].mean() < grey[:, -40:].mean()

    def test_a_negligible_curl_barely_moves_anything(self, striped_page, rng):
        image, page = striped_page

        _, new_page, _ = PageCurl(strength=0.0005).apply(image, page, rng)

        assert new_page.regions[0].bbox.iou(page.regions[0].bbox) > 0.95


class TestWrinkles:
    def test_ridges_are_shaded(self, striped_page, rng):
        image, page = striped_page
        flat = np.asarray(image.convert("L"), dtype=np.float32).mean()

        wrinkled, _, _ = Wrinkles(strength=5.0, shading=0.3).apply(image, page, rng)

        assert np.asarray(wrinkled.convert("L"), dtype=np.float32).mean() < flat

    def test_displacement_is_smooth_not_per_pixel(self, striped_page, rng):
        # A per-pixel field would shred the stripes into noise; a smooth one keeps them
        # recognisable as lines.
        image, page = striped_page

        wrinkled, _, _ = Wrinkles(strength=4.0).apply(image, page, rng)

        grey = np.asarray(wrinkled.convert("L"), dtype=np.float32)
        row_variation = np.abs(np.diff(grey.mean(axis=1))).mean()
        assert row_variation > 1.0, "stripes should still alternate"


class TestIlluminationField:
    def test_geometry_is_untouched(self, striped_page, rng):
        image, page = striped_page

        lit, new_page, _ = IlluminationField(strength=0.3).apply(image, page, rng)

        assert lit.size == image.size
        assert new_page.regions[0].bbox == page.regions[0].bbox

    def test_lighting_varies_across_the_page(self, striped_page, rng):
        image, page = striped_page

        lit, _, _ = IlluminationField(strength=0.4).apply(image, page, rng)

        grey = np.asarray(lit.convert("L"), dtype=np.float32)
        quadrants = [
            grey[:150, :200].mean(),
            grey[:150, 200:].mean(),
            grey[150:, :200].mean(),
            grey[150:, 200:].mean(),
        ]
        assert max(quadrants) - min(quadrants) > 3.0

    def test_it_differs_from_a_flat_brightness_change(self, striped_page, rng):
        image, page = striped_page

        lit, _, _ = IlluminationField(strength=0.35).apply(image, page, random.Random(2))

        grey = np.asarray(lit.convert("L"), dtype=np.float32)
        columns = grey.mean(axis=0)
        assert columns.std() > 1.0, "a uniform change would leave no column-wise spread"


class TestPresets:
    def test_photo_now_models_an_uneven_surface(self):
        names = {type(d).__name__ for d in build_preset("photo").degradations}

        assert {"PageCurl", "Wrinkles", "IlluminationField"} <= names

    def test_archive_wrinkles_its_paper(self):
        names = {type(d).__name__ for d in build_preset("archive").degradations}

        assert "Wrinkles" in names

    @pytest.mark.parametrize("name", ["photo", "archive"])
    def test_presets_keep_the_annotation_consistent(self, name, striped_page):
        image, page = striped_page

        result, new_page, _ = build_preset(name).apply(image, page, random.Random(9))

        assert (new_page.width, new_page.height) == result.size
        for region in new_page.regions:
            assert new_page.bbox.contains(region.bbox)


class TestFieldConstruction:
    """Fields are built from broadcast axis vectors rather than two full-page grids.

    `np.mgrid[0:h, 0:w]` costs 60 MB of int64 for an A4 page at 200 dpi before any
    arithmetic, and every field here is either separable or depends on one axis. The
    optimisation is only legitimate if the arithmetic is unchanged, so that is what is
    asserted: the axes must equal the grids they replaced, exactly.
    """

    @pytest.mark.parametrize("size", [(7, 5), (256, 129)])
    def test_axes_equal_the_grid_they_replace(self, size):
        width, height = size

        rows, columns = _axes(width, height)
        grid_rows, grid_columns = np.mgrid[0:height, 0:width]

        assert np.array_equal(np.broadcast_to(rows, (height, width)), grid_rows)
        assert np.array_equal(np.broadcast_to(columns, (height, width)), grid_columns)

    def test_axes_do_not_materialise_the_page(self):
        rows, columns = _axes(4000, 3000)

        assert rows.shape == (3000, 1)
        assert columns.shape == (1, 4000)

    def test_a_single_axis_field_still_maps_the_annotation(self, striped_page):
        # PageCurl varies along one axis, so its field is a row or column vector. The
        # annotation mapper indexes it per point, which broke before it was broadcast.
        image, page = striped_page

        result, new_page, _ = PageCurl(strength=0.1, edge="left").apply(image, page, random.Random(3))

        assert (new_page.width, new_page.height) == result.size
        assert new_page.regions[0].bbox != page.regions[0].bbox, "the curl must move the ink"
