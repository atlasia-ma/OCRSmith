"""Contract for coverage-aware font selection.

The bug these pin: coverage was checked against the *logical* string while the renderer
drew *presentation forms*. Without Raqm, arabic-reshaper turns `أمينة` into U+FE94 U+FEE8
… and a modern OpenType face like Fustat or Mada covers the base Arabic block while
carrying no presentation-form glyphs at all — it joins via GSUB instead. Probing the
logical string reported 100% coverage for such a face, and every glyph then rendered as an
empty box while the label still claimed the text.
"""

from pathlib import Path

import pytest

from ocrsmith.core.fonts import FontPool
from ocrsmith.text.shaping import ReshaperBidiShaper, TransparentShaper

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

# Covers base Arabic *and* presentation forms.
FULL_FORMS = "NotoSansArabic-Regular.ttf"
# Covers base Arabic but has no presentation forms.
GSUB_ONLY = "Fustat-Medium.ttf"

ARABIC = "أمينة كتب الدرس"

pytestmark = pytest.mark.skipif(not (FONT_DIR / GSUB_ONLY).exists(), reason="bundled fonts unavailable")


@pytest.fixture
def pool():
    return FontPool(
        [FONT_DIR / FULL_FORMS, FONT_DIR / GSUB_ONLY],
        shaper=ReshaperBidiShaper(),
    )


class TestPresentationFormCoverage:
    def test_a_font_without_presentation_forms_is_rejected(self, pool):
        eligible = {path.name for path in pool.supporting(ARABIC)}

        assert FULL_FORMS in eligible
        assert GSUB_ONLY not in eligible, "font would render tofu under the reshaper backend"

    def test_choose_never_returns_the_unusable_font(self, pool):
        import random

        for seed in range(20):
            assert pool.choose(ARABIC, random.Random(seed)).name == FULL_FORMS

    def test_latin_is_unaffected(self, pool):
        eligible = {path.name for path in pool.supporting("hello world")}

        assert eligible == {FULL_FORMS, GSUB_ONLY}

    def test_a_raqm_backend_judges_the_logical_form_instead(self):
        # With Raqm, HarfBuzz applies GSUB and never emits presentation forms, so a
        # GSUB-only face is perfectly usable and must not be excluded.
        pool = FontPool(
            [FONT_DIR / FULL_FORMS, FONT_DIR / GSUB_ONLY],
            shaper=TransparentShaper(),
        )

        eligible = {path.name for path in pool.supporting(ARABIC)}

        assert eligible == {FULL_FORMS, GSUB_ONLY}

    def test_disabling_the_requirement_still_returns_everything(self):
        pool = FontPool(
            [FONT_DIR / FULL_FORMS, FONT_DIR / GSUB_ONLY],
            require_full_coverage=False,
            shaper=ReshaperBidiShaper(),
        )

        assert len(pool.supporting(ARABIC)) == 2


class TestProbeCompleteness:
    def test_table_and_list_text_reach_the_coverage_probe(self):
        from ocrsmith.core.documents import DocumentBuilder

        content = (
            DocumentBuilder().paragraph("short").list(["ONE", "TWO"]).table([["HEAD"], ["CELL"]]).build()
        )

        probe = content.all_text

        # `text` alone misses both, which is how an invoice picked a font on the strength
        # of four words of prose and then drew its whole table as tofu.
        assert "CELL" in probe and "HEAD" in probe
        assert "ONE" in probe and "TWO" in probe
        assert "CELL" not in content.text


class TestHarfBuzzEngineUnlocksTheModernPool:
    """The reason the engine exists, as a test rather than a claim.

    A face that joins through `GSUB` carries no Arabic presentation forms, so the
    reshaper path cannot draw it and the coverage gate correctly refuses it. That is
    every handwriting and calligraphic Arabic face, so without the engine the
    `handwritten` code path has nothing to select.
    """

    HANDS = Path(__file__).resolve().parents[1] / "assets" / "fonts_hands"
    ARABIC = "دابا كايدوز بزاف تلوقت فدار"

    def test_a_modern_face_has_no_presentation_forms(self):
        """The premise. If this ever fails, the rest of the reasoning is stale."""
        from ocrsmith.text.coverage import drawable_codepoints

        face = self.HANDS / "arefruqaa" / "ArefRuqaa-Regular.ttf"
        if not face.exists():
            pytest.skip("run: ocrsmith fetch-fonts --subset arabic --hands")
        drawable = drawable_codepoints(face)

        assert len({c for c in drawable if 0x0600 <= c <= 0x06FF}) > 80, "no base Arabic"
        assert not {c for c in drawable if 0xFE70 <= c <= 0xFEFF}, "unexpectedly has FExx"

    def test_the_engine_draws_it_and_the_reshaper_cannot(self):
        from ocrsmith.text.shaping import HarfBuzzShaper, ReshaperBidiShaper, harfbuzz_available

        face = self.HANDS / "arefruqaa" / "ArefRuqaa-Regular.ttf"
        if not face.exists():
            pytest.skip("run: ocrsmith fetch-fonts --subset arabic --hands")
        if not harfbuzz_available():
            pytest.skip("uharfbuzz / freetype-py not installed")

        with_engine = FontPool([face], shaper=HarfBuzzShaper())
        with_reshaper = FontPool([face], shaper=ReshaperBidiShaper())

        assert with_engine.supporting(self.ARABIC), "the engine should accept a GSUB face"
        assert not with_reshaper.supporting(self.ARABIC), "the reshaper should refuse it"

    def test_the_engine_actually_puts_ink_on_the_page(self):
        """Coverage says a face *can* be drawn; this says it *was*."""
        import random

        import numpy as np

        from ocrsmith.core.fonts import load_font
        from ocrsmith.core.rendering.text_renderer import TextBlockRenderer
        from ocrsmith.text.shaping import harfbuzz_available

        face = self.HANDS / "arefruqaa" / "ArefRuqaa-Regular.ttf"
        if not face.exists():
            pytest.skip("run: ocrsmith fetch-fonts --subset arabic --hands")
        if not harfbuzz_available():
            pytest.skip("uharfbuzz / freetype-py not installed")

        rendered = TextBlockRenderer().render(self.ARABIC, load_font(face, 40), rng=random.Random(0))
        alpha = np.asarray(rendered.image.split()[-1])

        assert rendered.lines, "nothing was laid out"
        assert len(rendered.lines[0].words) == len(self.ARABIC.split())
        assert (alpha > 128).sum() > 400, "the glyphs drew nothing"
