"""Contract for bidi/shaping.

The label a model is trained on is *logical* order; the pixels on the page are
*visual* order. Confusing the two silently poisons an Arabic dataset, so the
distinction is modelled explicitly and asserted here.
"""

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

        with pytest.raises(Exception):
            shaped.visual = "b"

    def test_reports_whether_reshaping_happened(self):
        assert not TransparentShaper().shape(HELLO_AR).was_reshaped
        assert ReshaperBidiShaper().shape(HELLO_AR).was_reshaped


class TestShaperResolution:
    def test_auto_selects_a_working_shaper(self):
        shaper = resolve_shaper("auto")

        assert isinstance(shaper, (TransparentShaper, ReshaperBidiShaper))

    def test_explicit_backends_can_be_requested(self):
        assert isinstance(resolve_shaper("raqm"), TransparentShaper)
        assert isinstance(resolve_shaper("reshaper"), ReshaperBidiShaper)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown shaping backend"):
            resolve_shaper("magic")

    def test_any_shaper_keeps_the_label_intact(self):
        for backend in ("raqm", "reshaper"):
            assert resolve_shaper(backend).shape(HELLO_AR).logical == HELLO_AR
