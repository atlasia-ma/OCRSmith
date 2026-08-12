"""Contract for the text block renderer.

These are property tests against real fonts: exact pixel positions depend on the FreeType
build, but the invariants that make a dataset usable do not.

The invariants:

* the annotation transcribes exactly what was drawn — no more, no less;
* every word box lies inside the canvas and inside its line box;
* word boxes on a line do not overlap;
* the label stays in logical order even when the pixels are right-to-left.
"""

import random
from pathlib import Path

import pytest
from PIL import ImageFont

from ocrsmith.core.rendering import Alignment, TextBlockRenderer, TextStyle
from ocrsmith.text import Direction

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
ARABIC_FONT_PATH = FONT_DIR / "NotoSansArabic-Regular.ttf"

pytestmark = pytest.mark.skipif(not ARABIC_FONT_PATH.exists(), reason="bundled fonts unavailable")

ARABIC = "مرحبا بالعالم من أطلسيا"
LATIN = "the quick brown fox jumps"


@pytest.fixture
def font():
    return ImageFont.truetype(str(ARABIC_FONT_PATH), size=28)


@pytest.fixture
def renderer():
    return TextBlockRenderer()


def all_words(rendered):
    return [word for line in rendered.lines for word in line.words]


class TestTranscription:
    def test_label_matches_the_drawn_text(self, renderer, font):
        rendered = renderer.render(LATIN, font)

        assert rendered.text == LATIN

    def test_arabic_label_stays_in_logical_order(self, renderer, font):
        rendered = renderer.render(ARABIC, font)

        assert rendered.text == ARABIC

    def test_word_annotations_follow_logical_order(self, renderer, font):
        rendered = renderer.render(ARABIC, font)

        assert [word.text for word in all_words(rendered)] == ARABIC.split()

    def test_wrapped_text_still_transcribes_completely(self, renderer, font):
        rendered = renderer.render(LATIN, font, max_width=120)

        assert len(rendered.lines) > 1
        assert rendered.text.split() == LATIN.split()

    def test_paragraphs_become_separate_lines(self, renderer, font):
        rendered = renderer.render("first\nsecond", font)

        assert [line.text for line in rendered.lines] == ["first", "second"]

    def test_height_budget_drops_lines_and_says_so(self, renderer, font):
        rendered = renderer.render(LATIN, font, max_width=100, max_height=60)

        assert rendered.dropped_lines > 0
        # The label must describe only what is actually on the canvas.
        assert len(rendered.text.split()) < len(LATIN.split())

    def test_empty_text_renders_nothing_but_does_not_crash(self, renderer, font):
        rendered = renderer.render("   ", font)

        assert rendered.lines == ()
        assert rendered.text == ""


class TestGeometry:
    def test_every_word_box_is_inside_the_canvas(self, renderer, font):
        rendered = renderer.render(ARABIC, font, max_width=200)
        width, height = rendered.size

        for word in all_words(rendered):
            assert word.bbox.x0 >= 0 and word.bbox.x1 <= width
            assert word.bbox.y0 >= 0 and word.bbox.y1 <= height

    def test_every_word_box_is_inside_its_line_box(self, renderer, font):
        rendered = renderer.render(ARABIC, font, max_width=200)

        for line in rendered.lines:
            for word in line.words:
                assert line.bbox.contains(word.bbox)

    def test_word_boxes_on_a_line_do_not_overlap(self, renderer, font):
        rendered = renderer.render(LATIN, font)

        for line in rendered.lines:
            boxes = sorted((w.bbox for w in line.words), key=lambda b: b.x0)
            for left, right in zip(boxes, boxes[1:], strict=False):
                assert left.x1 <= right.x0 + 1  # a pixel of tolerance for bearings

    def test_word_boxes_are_not_degenerate(self, renderer, font):
        rendered = renderer.render(ARABIC, font)

        for word in all_words(rendered):
            assert word.bbox.width > 0
            assert word.bbox.height > 0

    def test_lines_advance_down_the_page(self, renderer, font):
        rendered = renderer.render(LATIN, font, max_width=120)

        tops = [line.bbox.y0 for line in rendered.lines]
        assert tops == sorted(tops)

    def test_baseline_sits_inside_the_line(self, renderer, font):
        rendered = renderer.render(LATIN, font)

        line = rendered.lines[0]
        assert line.bbox.y0 <= line.baseline <= line.bbox.y1 + font.size


class TestDirection:
    def test_arabic_is_detected_as_right_to_left(self, renderer, font):
        rendered = renderer.render(ARABIC, font)

        assert rendered.lines[0].direction is Direction.RTL

    def test_arabic_first_word_is_drawn_rightmost(self, renderer, font):
        rendered = renderer.render(ARABIC, font)
        words = rendered.lines[0].words

        assert words[0].bbox.x0 > words[-1].bbox.x0

    def test_latin_first_word_is_drawn_leftmost(self, renderer, font):
        rendered = renderer.render(LATIN, font)
        words = rendered.lines[0].words

        assert words[0].bbox.x0 < words[-1].bbox.x0

    def test_direction_can_be_forced(self, renderer, font):
        rendered = renderer.render(LATIN, font, direction=Direction.RTL)

        assert rendered.lines[0].direction is Direction.RTL


class TestStyle:
    def test_right_alignment_pushes_short_lines_to_the_edge(self, renderer, font):
        left = renderer.render("hi", font, TextStyle(align=Alignment.LEFT), max_width=400)
        right = renderer.render("hi", font, TextStyle(align=Alignment.RIGHT), max_width=400)

        assert right.lines[0].bbox.x0 > left.lines[0].bbox.x0

    def test_centre_alignment_sits_between_the_two(self, renderer, font):
        style = TextStyle(align=Alignment.CENTER)
        centered = renderer.render("hi", font, style, max_width=400)
        left = renderer.render("hi", font, TextStyle(align=Alignment.LEFT), max_width=400)
        right = renderer.render("hi", font, TextStyle(align=Alignment.RIGHT), max_width=400)

        assert left.lines[0].bbox.x0 < centered.lines[0].bbox.x0 < right.lines[0].bbox.x0

    def test_natural_alignment_follows_the_script(self, renderer, font):
        arabic = renderer.render("مرحبا", font, TextStyle(align=Alignment.NATURAL), max_width=400)
        latin = renderer.render("hi", font, TextStyle(align=Alignment.NATURAL), max_width=400)

        assert arabic.lines[0].bbox.x1 > latin.lines[0].bbox.x1

    def test_line_spacing_changes_the_block_height(self, renderer, font):
        tight = renderer.render(LATIN, font, TextStyle(line_spacing=1.0), max_width=120)
        loose = renderer.render(LATIN, font, TextStyle(line_spacing=2.0), max_width=120)

        assert loose.size[1] > tight.size[1]

    def test_stroke_widens_the_word_boxes(self, renderer, font):
        plain = renderer.render("hi", font)
        stroked = renderer.render("hi", font, TextStyle(stroke_width=3))

        assert stroked.lines[0].words[0].bbox.width > plain.lines[0].words[0].bbox.width

    def test_underline_does_not_change_the_transcription(self, renderer, font):
        rendered = renderer.render(LATIN, font, TextStyle(underline=True))

        assert rendered.text == LATIN

    def test_synthetic_italic_widens_the_canvas(self, renderer, font):
        upright = renderer.render("hi", font)
        oblique = renderer.render("hi", font, TextStyle(synthetic_italic=0.25))

        assert oblique.size[0] > upright.size[0]

    def test_canvas_is_transparent_where_nothing_was_drawn(self, renderer, font):
        rendered = renderer.render("hi", font)

        assert rendered.image.mode == "RGBA"
        assert rendered.image.getpixel((0, 0))[3] == 0


class TestDeterminism:
    def test_same_seed_gives_identical_geometry(self, renderer, font):
        style = TextStyle(baseline_jitter=3, word_spacing_jitter=4)

        first = renderer.render(LATIN, font, style, rng=random.Random(11))
        second = renderer.render(LATIN, font, style, rng=random.Random(11))

        assert [w.bbox for w in all_words(first)] == [w.bbox for w in all_words(second)]

    def test_different_seeds_move_things_around(self, renderer, font):
        style = TextStyle(baseline_jitter=4, word_spacing_jitter=6)

        first = renderer.render(LATIN, font, style, rng=random.Random(1))
        second = renderer.render(LATIN, font, style, rng=random.Random(2))

        assert [w.bbox for w in all_words(first)] != [w.bbox for w in all_words(second)]

    def test_no_jitter_means_no_randomness(self, renderer, font):
        first = renderer.render(LATIN, font, rng=random.Random(1))
        second = renderer.render(LATIN, font, rng=random.Random(999))

        assert [w.bbox for w in all_words(first)] == [w.bbox for w in all_words(second)]


class TestTranslation:
    def test_translated_annotation_moves_with_the_paste(self, renderer, font):
        rendered = renderer.render(LATIN, font)

        moved = rendered.translated(100, 50)

        original = rendered.lines[0].words[0].bbox
        assert moved[0].words[0].bbox.x0 == original.x0 + 100
        assert moved[0].words[0].bbox.y0 == original.y0 + 50
