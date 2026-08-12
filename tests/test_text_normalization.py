"""Contract for the Arabic-aware text normalisation layer.

Normalisation decides what the *label* says, so every transform here is part of the
dataset's ground truth and must be explicit, reversible in intent, and idempotent.
"""

import pytest

from ocrsmith.text import (
    NormalizationPolicy,
    NumeralSystem,
    normalize_text,
    strip_diacritics,
    strip_tatweel,
    to_numeral_system,
)

BASMALA = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
BASMALA_BARE = "بسم الله الرحمن الرحيم"


class TestDiacritics:
    def test_strip_diacritics_removes_tashkeel(self):
        assert strip_diacritics(BASMALA) == BASMALA_BARE

    def test_strip_diacritics_is_idempotent(self):
        once = strip_diacritics(BASMALA)
        assert strip_diacritics(once) == once

    def test_strip_diacritics_leaves_latin_untouched(self):
        assert strip_diacritics("Café naïve") == "Café naïve"

    def test_strip_diacritics_preserves_word_count(self):
        assert len(strip_diacritics(BASMALA).split()) == len(BASMALA.split())


class TestTatweel:
    def test_strip_tatweel_removes_kashida(self):
        assert strip_tatweel("مـــرحبا") == "مرحبا"

    def test_strip_tatweel_noop_without_kashida(self):
        assert strip_tatweel("مرحبا") == "مرحبا"


class TestNumerals:
    @pytest.mark.parametrize(
        ("system", "expected"),
        [
            (NumeralSystem.WESTERN, "2024 سنة"),
            (NumeralSystem.ARABIC_INDIC, "٢٠٢٤ سنة"),
            (NumeralSystem.EASTERN_ARABIC_INDIC, "۲۰۲۴ سنة"),
        ],
    )
    def test_conversion_between_numeral_systems(self, system, expected):
        assert to_numeral_system("2024 سنة", system) == expected

    def test_conversion_round_trips_through_western(self):
        arabic = to_numeral_system("42", NumeralSystem.ARABIC_INDIC)
        assert to_numeral_system(arabic, NumeralSystem.WESTERN) == "42"

    def test_keep_leaves_every_digit_alone(self):
        mixed = "42 و ٤٢"
        assert to_numeral_system(mixed, NumeralSystem.KEEP) == mixed


class TestNormalizationPolicy:
    def test_default_policy_only_collapses_whitespace(self):
        assert normalize_text("  a\t\tb \n c  ", NormalizationPolicy()) == "a b c"

    def test_default_policy_preserves_diacritics(self):
        assert normalize_text(BASMALA, NormalizationPolicy()) == BASMALA

    def test_policy_can_strip_diacritics_and_convert_numerals(self):
        policy = NormalizationPolicy(
            strip_diacritics=True,
            numerals=NumeralSystem.ARABIC_INDIC,
        )
        assert normalize_text("سنة 2024 " + BASMALA, policy) == "سنة ٢٠٢٤ " + BASMALA_BARE

    def test_policy_can_normalize_alef_and_ya_variants(self):
        policy = NormalizationPolicy(unify_alef=True, unify_ya=True)
        assert normalize_text("أحمد إلى آية على", policy) == "احمد الي اية علي"

    def test_policy_can_be_disabled_entirely(self):
        raw = "  a\t\tb  "
        assert normalize_text(raw, NormalizationPolicy(collapse_whitespace=False)) == raw

    def test_normalize_text_is_idempotent(self):
        policy = NormalizationPolicy(strip_diacritics=True, unify_alef=True)
        once = normalize_text("  أ " + BASMALA, policy)
        assert normalize_text(once, policy) == once

    def test_line_breaks_survive_when_requested(self):
        policy = NormalizationPolicy(preserve_line_breaks=True)
        assert normalize_text("a  b\n\n c ", policy) == "a b\nc"
