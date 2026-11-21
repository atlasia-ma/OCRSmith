"""Contract for bidi/shaping.

The label a model is trained on is *logical* order; the pixels on the page are
*visual* order. Confusing the two silently poisons an Arabic dataset, so the
distinction is modelled explicitly and asserted here.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ocrsmith.text import (
    Direction,
    ReshaperBidiShaper,
    Script,
    ShapedText,
    TransparentShaper,
    detect_direction,
    detect_script,
    resolve_shaper,
)

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

HELLO_AR = "مرحبا"
# Presentation forms: what a non-Raqm renderer needs in order to draw joined Arabic.
PRESENTATION_RANGE = range(0xFE70, 0xFF00)


class TestScriptDetection:
    @pytest.mark.parametrize(
        ("text", "script"),
        [
            ("مرحبا بالعالم", Script.ARABIC),
            ("hello world", Script.LATIN),
            ("مرحبا hello", Script.MIXED),
            ("12345", Script.NEUTRAL),
            ("", Script.NEUTRAL),
        ],
    )
    def test_detect_script(self, text, script):
        assert detect_script(text) is script

    @pytest.mark.parametrize(
        ("text", "direction"),
        [
            ("مرحبا بالعالم", Direction.RTL),
            ("hello world", Direction.LTR),
            ("2024", Direction.LTR),
            # First strong character wins, as in the Unicode bidi algorithm.
            ("2024 مرحبا", Direction.RTL),
            ("hello مرحبا", Direction.LTR),
        ],
    )
    def test_detect_direction(self, text, direction):
        assert detect_direction(text) is direction


class TestTransparentShaper:
    """Used when the renderer itself (Raqm/HarfBuzz) performs shaping and bidi."""

    def test_visual_equals_logical(self):
        shaped = TransparentShaper().shape(HELLO_AR)

        assert shaped.logical == HELLO_AR
        assert shaped.visual == HELLO_AR
        assert shaped.direction is Direction.RTL

    def test_latin_is_left_to_right(self):
        assert TransparentShaper().shape("hello").direction is Direction.LTR


class TestReshaperBidiShaper:
    """Used when the renderer cannot shape, so we hand it presentation forms."""

    def test_logical_text_is_preserved_for_the_label(self):
        assert ReshaperBidiShaper().shape(HELLO_AR).logical == HELLO_AR

    def test_visual_text_uses_presentation_forms(self):
        visual = ReshaperBidiShaper().shape(HELLO_AR).visual

        assert any(ord(ch) in PRESENTATION_RANGE for ch in visual)

    def test_visual_text_is_reversed_for_rtl(self):
        visual = ReshaperBidiShaper().shape(HELLO_AR).visual

        # The last logical letter is drawn leftmost, so it comes first in visual order.
        assert visual[0] != HELLO_AR[0]
        assert len(visual) == len(HELLO_AR)

    def test_latin_passes_through_unchanged(self):
        shaped = ReshaperBidiShaper().shape("hello world")

        assert shaped.visual == "hello world"

    def test_embedded_latin_keeps_its_own_direction(self):
        shaped = ReshaperBidiShaper().shape("سنة 2024 OCR")

        assert "OCR" in shaped.visual
        assert "2024" in shaped.visual

    def test_empty_text_is_safe(self):
        shaped = ReshaperBidiShaper().shape("")

        assert shaped.visual == ""
        assert shaped.logical == ""


class TestShapedText:
    def test_is_immutable(self):
        shaped = ShapedText(logical="a", visual="a", direction=Direction.LTR)

        with pytest.raises(FrozenInstanceError):
            shaped.visual = "b"

    def test_reports_whether_reshaping_happened(self):
        assert not TransparentShaper().shape(HELLO_AR).was_reshaped
        assert ReshaperBidiShaper().shape(HELLO_AR).was_reshaped


class TestShaperResolution:
    def test_auto_selects_a_working_shaper(self):
        from ocrsmith.text.shaping import HarfBuzzShaper

        shaper = resolve_shaper("auto")

        assert isinstance(shaper, (HarfBuzzShaper, TransparentShaper, ReshaperBidiShaper))

    def test_auto_prefers_harfbuzz_when_it_is_available(self):
        """Backend choice is by how much of the font pool each one can draw.

        The reshaper path needs Arabic presentation forms, and the modern faces — every
        handwriting and calligraphic one among them — do not implement that block.
        """
        from ocrsmith.text.shaping import HarfBuzzShaper, harfbuzz_available

        if not harfbuzz_available():
            pytest.skip("uharfbuzz / freetype-py not installed")

        assert isinstance(resolve_shaper("auto"), HarfBuzzShaper)

    def test_explicit_backends_can_be_requested(self):
        from ocrsmith.text.shaping import HarfBuzzShaper

        assert isinstance(resolve_shaper("raqm"), TransparentShaper)
        assert isinstance(resolve_shaper("reshaper"), ReshaperBidiShaper)
        assert isinstance(resolve_shaper("harfbuzz"), HarfBuzzShaper)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown shaping backend"):
            resolve_shaper("magic")

    def test_any_shaper_keeps_the_label_intact(self):
        for backend in ("raqm", "reshaper"):
            assert resolve_shaper(backend).shape(HELLO_AR).logical == HELLO_AR


class TestShapingCache:
    """Shaping is cached because arabic-reshaper is pathologically slow per call.

    Caching a pure function is only safe if it really is pure, so these pin the
    behaviour the cache must not change.
    """

    def test_repeated_shaping_is_stable(self):
        shaper = ReshaperBidiShaper()

        first = shaper.shape(HELLO_AR)
        second = shaper.shape(HELLO_AR)

        assert first == second

    def test_the_cache_does_not_bleed_between_inputs(self):
        shaper = ReshaperBidiShaper()

        one = shaper.shape("مرحبا").visual
        two = shaper.shape("بالعالم").visual

        assert one != two
        assert shaper.shape("مرحبا").visual == one

    def test_lam_alef_still_forms_a_ligature(self):
        # The reshaper's ligature table is what the cache-warming workaround touches, so
        # a lost ligature is the failure mode to watch for.
        shaped = ReshaperBidiShaper().shape("لا")

        assert len(shaped.visual) == 1
        assert shaped.logical == "لا"

    def test_a_shaped_word_keeps_its_character_count_otherwise(self):
        shaped = ReshaperBidiShaper().shape("بالعالم")

        assert len(shaped.visual) == len("بالعالم")


class TestDiacriticsSurviveShaping:
    """The fallback shaper must not delete the marks it was handed.

    arabic-reshaper defaults `delete_harakat` to True, which removes every tashkeel mark
    before reshaping. Without Raqm — the default on a Windows or conda Pillow — the marks
    then never reach the rasteriser while the label still carries them, so `keep`,
    `partial` and `strip` all render identically and the first two assert vowels that are
    not in the pixels. Nothing raises; the dataset just teaches a model to hallucinate.
    """

    VOCALISED = "دابا كايدوّز بزّاف"
    SHADDA = "ّ"

    def test_the_shaped_form_still_carries_the_marks(self):
        from ocrsmith.text.shaping import resolve_shaper

        visual = resolve_shaper().shape(self.VOCALISED).visual

        assert self.SHADDA in visual

    def test_mark_count_is_preserved(self):
        from ocrsmith.text.shaping import resolve_shaper

        visual = resolve_shaper().shape(self.VOCALISED).visual

        assert visual.count(self.SHADDA) == self.VOCALISED.count(self.SHADDA)

    def test_a_stripped_string_shapes_without_them(self):
        """The marks in the output come from the input, not from the shaper."""
        from ocrsmith.text.normalization import strip_diacritics
        from ocrsmith.text.shaping import resolve_shaper

        visual = resolve_shaper().shape(strip_diacritics(self.VOCALISED)).visual

        assert self.SHADDA not in visual

    def test_keeping_and_stripping_produce_different_pixels(self):
        """The end-to-end property: `text.diacritics` has to change the image."""
        import random

        import numpy as np

        from ocrsmith.core.fonts import load_font
        from ocrsmith.core.lines import LineRenderer, LineStrip
        from ocrsmith.text.normalization import strip_diacritics

        font = load_font(FONT_DIR / "NotoNaskhArabic-Regular.ttf", 36)
        strip = LineStrip(text_height=46, pad_left=14, pad_right=14, pad_top=12, pad_bottom=12)
        renderer = LineRenderer()

        kept = renderer.render(self.VOCALISED, font, None, strip, rng=random.Random(0)).image
        bare = renderer.render(
            strip_diacritics(self.VOCALISED), font, None, strip, rng=random.Random(0)
        ).image

        assert kept.size == bare.size, "the marks changed the advance widths, not just the ink"
        ink = lambda image: int((np.asarray(image.convert("L")) < 128).sum())  # noqa: E731
        assert ink(kept) > ink(bare), "the vocalised render drew no extra ink"
