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
