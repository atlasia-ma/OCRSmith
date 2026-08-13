"""Contract for diacritics handling.

The governing rule: marks are only ever **removed**, never invented. Adding vocalisation
to bare text would need a diacritiser model and would make the label assert vowels no
human wrote — a fabricated ground truth that looks perfectly plausible.
"""

import random

import pytest

from ocrsmith.text import (
    DiacriticsMode,
    DiacriticsPolicy,
    apply_diacritics,
    count_diacritics,
    diacritic_ratio,
    strip_diacritics,
    strip_partial,
)

VOCALISED = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
BARE = "بسم الله الرحمن الرحيم"


class TestMeasurement:
    def test_counts_marks(self):
        assert count_diacritics(VOCALISED) > 10
        assert count_diacritics(BARE) == 0

    def test_ratio_separates_vocalised_from_bare(self):
        assert diacritic_ratio(VOCALISED) > 0.4
        assert diacritic_ratio(BARE) == 0.0

    def test_ratio_of_empty_text_is_zero(self):
        assert diacritic_ratio("") == 0.0

    def test_latin_carries_no_marks(self):
        assert count_diacritics("hello world") == 0


class TestPartialRemoval:
    def test_keeping_everything_is_the_identity(self):
        assert strip_partial(VOCALISED, 1.0, random.Random(1)) == VOCALISED

    def test_keeping_nothing_matches_a_full_strip(self):
        assert strip_partial(VOCALISED, 0.0, random.Random(1)) == strip_diacritics(VOCALISED)

    def test_a_partial_strip_lands_between_the_two(self):
        partial = strip_partial(VOCALISED, 0.5, random.Random(7))

        assert 0 < count_diacritics(partial) < count_diacritics(VOCALISED)

    def test_the_consonant_skeleton_is_never_touched(self):
        # Whatever happens to the marks, the letters must survive untouched — this is what
        # keeps a partially stripped label truthful.
        partial = strip_partial(VOCALISED, 0.4, random.Random(3))

        assert strip_diacritics(partial) == strip_diacritics(VOCALISED)

    def test_removal_is_reproducible(self):
        first = strip_partial(VOCALISED, 0.5, random.Random(11))
        second = strip_partial(VOCALISED, 0.5, random.Random(11))

        assert first == second


class TestPolicy:
    def test_keep_leaves_the_source_alone(self):
        text, kept = apply_diacritics(VOCALISED, DiacriticsPolicy(DiacriticsMode.KEEP), random.Random(1))

        assert text == VOCALISED
        assert kept == 1.0

    def test_strip_removes_everything(self):
        text, kept = apply_diacritics(VOCALISED, DiacriticsPolicy(DiacriticsMode.STRIP), random.Random(1))

        assert count_diacritics(text) == 0
        assert kept == 0.0

    def test_partial_reports_the_fraction_it_kept(self):
        policy = DiacriticsPolicy(DiacriticsMode.PARTIAL, keep_range=(0.3, 0.3))

        _, kept = apply_diacritics(VOCALISED, policy, random.Random(5))

        assert kept == pytest.approx(0.3)

    def test_bare_text_is_untouched_by_every_mode(self):
        for mode in DiacriticsMode:
            text, kept = apply_diacritics(BARE, DiacriticsPolicy(mode), random.Random(2))
            assert text == BARE
            assert kept == 1.0

    def test_mixed_produces_a_spread_across_a_corpus(self):
        policy = DiacriticsPolicy(DiacriticsMode.MIXED)

        ratios = {round(apply_diacritics(VOCALISED, policy, random.Random(seed))[1], 2) for seed in range(40)}

        # A mixed corpus must contain fully marked, partly marked and bare documents.
        assert 1.0 in ratios
        assert 0.0 in ratios
        assert any(0.0 < value < 1.0 for value in ratios)

    def test_mixed_weights_are_honoured(self):
        policy = DiacriticsPolicy(DiacriticsMode.MIXED, mixed_weights=(0.0, 0.0, 1.0))

        kept = {apply_diacritics(VOCALISED, policy, random.Random(s))[1] for s in range(15)}

        assert kept == {0.0}

    def test_nothing_is_ever_invented(self):
        # The strongest guarantee in this module: no mode may add marks to bare text.
        for mode in DiacriticsMode:
            for seed in range(10):
                text, _ = apply_diacritics(BARE, DiacriticsPolicy(mode), random.Random(seed))
                assert count_diacritics(text) == 0


class TestPipelineIntegration:
    def test_provenance_records_how_vocalised_the_sample_was(self, tmp_path):
        from pathlib import Path

        from ocrsmith.config import GenerationConfig
        from ocrsmith.pipeline import SampleFactory

        font_dir = Path(__file__).resolve().parents[1] / "assets" / "fonts"
        if not font_dir.exists():
            pytest.skip("bundled fonts unavailable")

        config = GenerationConfig.model_validate(
            {
                "seed": 5,
                "fonts": {"paths": [str(font_dir)], "include": ["NotoSansArabic-"], "size_range": [16, 18]},
                "text": {
                    "source": {"type": "inline", "sentences": [VOCALISED + " " + VOCALISED]},
                    "diacritics": {"mode": "strip"},
                },
                "page": {"papers": {"a6": 1.0}, "dpi_range": [90, 90], "max_pages": 1},
                "templates": {"weights": {"article": 1.0}},
                "degradations": {"presets": {"clean": 1.0}},
                "output": {"dir": str(tmp_path)},
            }
        )

        sample = next(iter(SampleFactory(config).create(0)))

        assert sample.provenance.extra["diacritics_kept"] == 0.0
        assert count_diacritics(sample.text) == 0
