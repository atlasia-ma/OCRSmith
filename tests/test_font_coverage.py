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


class TestHollowGlyphs:
    """A code point in the `cmap` is not proof the font draws anything.

    Estedad maps the Arabic presentation forms to glyphs with no outline - empty shapes
    carrying a normal advance - because it supplies the real joining shapes through GSUB
    instead. On a build without Raqm the renderer draws the reshaped presentation forms
    directly, lands on those stubs, and produces a page of gaps whose label still claims
    every word. Coverage must reject such a face, not merely note the code point exists.
    """

    ESTEDAD = (
        Path(__file__).resolve().parents[1] / "assets" / "fonts_google" / "estedad" / "Estedad[wght].ttf"
    )
    # Presentation forms the reshaper emits for "كنتي": initial kaf, medial noon,
    # medial teh, final yeh. All four are in Estedad's cmap and all four are hollow.
    HOLLOW_FORMS = "ﻛﻨﺘﻲ"

    @pytest.mark.skipif(not ESTEDAD.exists(), reason="Estedad not fetched")
    def test_hollow_presentation_forms_count_as_missing(self):
        report = supports_text(self.ESTEDAD, self.HOLLOW_FORMS)

        assert not report.is_complete
        assert len(report.missing) == len(self.HOLLOW_FORMS)

    @pytest.mark.skipif(not ESTEDAD.exists(), reason="Estedad not fetched")
    def test_hollow_glyphs_are_not_reported_as_drawable(self):
        assert not has_glyph(self.ESTEDAD, self.HOLLOW_FORMS[0])

    @pytest.mark.skipif(not ESTEDAD.exists(), reason="Estedad not fetched")
    def test_a_face_with_real_forms_still_qualifies(self):
        eligible = fonts_supporting([ARABIC_FONT, self.ESTEDAD], self.HOLLOW_FORMS)

        assert str(ARABIC_FONT) in eligible
        assert str(self.ESTEDAD) not in eligible

    def test_the_base_letters_themselves_are_unaffected(self):
        # The bug is specific to the presentation-form stubs; the base block draws.
        assert has_glyph(ARABIC_FONT, "ك")
        assert supports_text(ARABIC_FONT, "كنتي").is_complete


class TestCoverageCache:
    """Coverage is answered from a disk cache keyed by the font's bytes.

    It is the reason a hundred-face pool costs milliseconds instead of seconds in every
    worker process, and it is also the one place where a wrong answer would be both
    persistent and invisible — so the key has to track the file, and a cache that cannot
    be used has to degrade to computing the answer rather than to returning a guess.
    """

    def test_a_cold_cache_writes_an_entry_and_a_warm_one_reuses_it(self, tmp_path, monkeypatch):
        from ocrsmith.text import coverage

        monkeypatch.setenv("OCRSMITH_CACHE_DIR", str(tmp_path))
        coverage.clear_coverage_cache()
        first = coverage.drawable_codepoints(ARABIC_FONT)
        entries = list(tmp_path.glob("*.json"))

        coverage.clear_coverage_cache()
        second = coverage.drawable_codepoints(ARABIC_FONT)

        assert len(entries) == 1
        assert first == second
        assert list(tmp_path.glob("*.json")) == entries

    def test_the_cached_answer_matches_an_uncached_one(self, tmp_path, monkeypatch):
        from ocrsmith.text import coverage

        monkeypatch.setenv("OCRSMITH_CACHE_DIR", "")
        coverage.clear_coverage_cache()
        uncached = coverage.drawable_codepoints(ARABIC_FONT)

        monkeypatch.setenv("OCRSMITH_CACHE_DIR", str(tmp_path))
        coverage.clear_coverage_cache()
        coverage.drawable_codepoints(ARABIC_FONT)
        coverage.clear_coverage_cache()

        assert coverage.drawable_codepoints(ARABIC_FONT) == uncached

    def test_an_edited_font_is_not_served_from_the_cache(self, tmp_path, monkeypatch):
        """The key carries size and mtime, so a replaced file is a different question."""
        from ocrsmith.text import coverage

        monkeypatch.setenv("OCRSMITH_CACHE_DIR", str(tmp_path / "cache"))
        face = tmp_path / "face.ttf"
        face.write_bytes(ARABIC_FONT.read_bytes())
        coverage.clear_coverage_cache()
        arabic = coverage.drawable_codepoints(face)

        face.write_bytes(LATIN_ONLY_FONT.read_bytes())
        coverage.clear_coverage_cache()

        assert coverage.drawable_codepoints(face) != arabic

    def test_an_empty_cache_dir_setting_disables_the_disk_cache(self, tmp_path, monkeypatch):
        from ocrsmith.text import coverage

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OCRSMITH_CACHE_DIR", "")
        coverage.clear_coverage_cache()
        coverage.drawable_codepoints(ARABIC_FONT)

        assert coverage.cache_dir() is None
        assert list(tmp_path.rglob("*.json")) == []

    def test_an_unwritable_cache_dir_is_a_slow_run_not_a_failed_one(self, tmp_path, monkeypatch):
        from ocrsmith.text import coverage

        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setenv("OCRSMITH_CACHE_DIR", str(blocker))
        coverage.clear_coverage_cache()

        assert coverage.drawable_codepoints(ARABIC_FONT)

    def test_warming_reports_how_many_faces_were_readable(self, tmp_path, monkeypatch):
        from ocrsmith.text.coverage import warm_coverage_cache

        monkeypatch.setenv("OCRSMITH_CACHE_DIR", str(tmp_path))
        missing = FONT_DIR / "does-not-exist.ttf"

        assert warm_coverage_cache([ARABIC_FONT, LATIN_ONLY_FONT, missing]) == 2


class TestRequiredCodepoints:
    def test_whitespace_is_not_something_a_font_must_draw(self):
        from ocrsmith.text.coverage import required_codepoints

        assert required_codepoints("a b") == frozenset({ord("a"), ord("b")})

    def test_zero_width_marks_are_exempt_too(self):
        from ocrsmith.text.coverage import required_codepoints

        assert required_codepoints("‌‍﻿") == frozenset()

    def test_a_space_does_not_make_every_font_ineligible(self):
        """The regression this guards: requiring an outline for U+0020 rejects the whole
        pool for any string containing a space, and the run then falls back to a single
        face without reporting anything."""
        assert fonts_supporting([ARABIC_FONT], "مرحبا بالعالم") == [str(ARABIC_FONT)]
