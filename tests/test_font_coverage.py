"""Contract for glyph-coverage checking.

Uses the fonts shipped in `assets/fonts`, which are Arabic-first: they carry Arabic and
basic Latin but not, say, CJK.
"""

from pathlib import Path

import pytest

from ocrsmith.text import (
    fonts_supporting,
    has_glyph,
    missing_glyphs,
    supports_text,
)

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
ARABIC_FONT = FONT_DIR / "NotoSansArabic-Regular.ttf"
LATIN_ONLY_FONT = FONT_DIR / "NotoSansMono-Light.ttf"

pytestmark = pytest.mark.skipif(not ARABIC_FONT.exists(), reason="bundled fonts unavailable")


class TestHasGlyph:
    def test_arabic_font_covers_arabic(self):
        assert has_glyph(ARABIC_FONT, "م")

    def test_arabic_font_covers_latin(self):
        assert has_glyph(ARABIC_FONT, "A")

    def test_arabic_font_does_not_cover_cjk(self):
        assert not has_glyph(ARABIC_FONT, "漢")

    def test_whitespace_is_always_considered_covered(self):
        assert has_glyph(ARABIC_FONT, " ")

    def test_empty_string_is_covered(self):
        assert has_glyph(ARABIC_FONT, "")


class TestMissingGlyphs:
    def test_reports_only_uncovered_characters(self):
        assert missing_glyphs(ARABIC_FONT, "مرحبا 漢字") == ("漢", "字")

    def test_deduplicates_and_preserves_order(self):
        assert missing_glyphs(ARABIC_FONT, "漢漢字漢") == ("漢", "字")

    def test_fully_covered_text_reports_nothing(self):
        assert missing_glyphs(ARABIC_FONT, "مرحبا بالعالم") == ()


class TestCoverageReport:
    def test_complete_report_is_truthy(self):
        report = supports_text(ARABIC_FONT, "مرحبا")

        assert report.is_complete
        assert bool(report) is True
        assert report.ratio == 1.0

    def test_partial_report_counts_covered_characters(self):
        report = supports_text(ARABIC_FONT, "ab漢")

        assert report.missing == ("漢",)
        assert report.covered == 2
        assert report.total == 3
        assert 0.0 < report.ratio < 1.0

    def test_empty_text_is_complete(self):
        report = supports_text(ARABIC_FONT, "")

        assert report.is_complete
        assert report.ratio == 1.0

    def test_whitespace_does_not_count_towards_the_total(self):
        assert supports_text(ARABIC_FONT, "a b").total == 2


class TestFontsSupporting:
    def test_filters_out_fonts_missing_the_script(self):
        candidates = [ARABIC_FONT, LATIN_ONLY_FONT]

        eligible = fonts_supporting(candidates, "مرحبا")

        assert str(ARABIC_FONT) in eligible
        assert str(LATIN_ONLY_FONT) not in eligible

    def test_all_fonts_qualify_for_plain_latin(self):
        eligible = fonts_supporting([ARABIC_FONT, LATIN_ONLY_FONT], "abc")

        assert len(eligible) == 2

    def test_unreadable_paths_are_skipped_rather_than_raising(self):
        assert fonts_supporting([FONT_DIR / "does-not-exist.ttf"], "abc") == []

    def test_min_ratio_allows_partial_coverage(self):
        eligible = fonts_supporting([LATIN_ONLY_FONT], "abcم", min_ratio=0.5)

        assert eligible == [str(LATIN_ONLY_FONT)]
