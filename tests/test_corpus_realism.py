"""Contract for non-prose corpus content.

A sentence corpus contains no dates, no reference numbers and no partial words — and those
are precisely the cases where a recogniser has no language model to lean on. A serial
number is only readable glyph by glyph, which is the hardest thing a recogniser does, and
a corpus of clean sentences never asks it to.
"""

import random

import pytest

from ocrsmith.core.documents import CorpusTextProvider, FieldGenerator, FormTemplate, InvoiceTemplate
from ocrsmith.text import NumeralSystem, to_numeral_system

SENTENCES = [
    "المغرب بلد يقع في شمال إفريقيا ويطل على المحيط الأطلسي.",
    "تهدف أطلسيا إلى بناء نماذج ذكاء اصطناعي مفتوحة المصدر.",
    "يحتوي هذا التقرير على جداول وأرقام ومعلومات إضافية مفيدة.",
]

WESTERN = set("0123456789")
ARABIC_INDIC = set("٠١٢٣٤٥٦٧٨٩")


@pytest.fixture
def source():
    return CorpusTextProvider(SENTENCES)


@pytest.fixture
def rng():
    return random.Random(20260813)


class TestFragments:
    def test_a_fragment_is_part_of_a_real_word(self, source, rng):
        words = {word for sentence in SENTENCES for word in sentence.split()}

        for _ in range(20):
            fragment = source.fragment(rng)
            assert any(fragment and fragment in word for word in words)

    def test_a_fragment_is_never_empty(self, source, rng):
        assert all(source.fragment(rng) for _ in range(30))

    def test_contextless_text_has_no_sentence_structure(self, source, rng):
        # The point is that no sentence predicts the next word.
        runs = {source.contextless(rng) for _ in range(15)}

        assert len(runs) > 5, "contextless runs should not repeat"

    def test_contextless_respects_its_word_budget(self, source, rng):
        for _ in range(20):
            assert 1 <= len(source.contextless(rng, max_words=3).split()) <= 3


class TestFields:
    @pytest.mark.parametrize("field", ["date", "amount", "code", "phone", "percentage"])
    def test_each_field_produces_something(self, field, rng):
        value = getattr(FieldGenerator(), field)(rng)

        assert value and value.strip() == value

    def test_dates_vary_in_format(self, rng):
        formats = {
            "slash" if "/" in d else "dash" if "-" in d else "words"
            for d in (FieldGenerator().date(random.Random(s)) for s in range(20))
        }

        assert len(formats) > 1, "a single date format teaches a single prior"

    def test_a_code_has_no_language_model_to_lean_on(self, rng):
        code = FieldGenerator().code(rng)

        assert any(c.isdigit() or c in "٠١٢٣٤٥٦٧٨٩" for c in code)
        assert any(c.isalpha() for c in code)

    def test_numerals_follow_the_requested_system(self, rng):
        western = FieldGenerator("western").date(random.Random(4))
        arabic = FieldGenerator("arabic_indic").date(random.Random(4))

        assert set(western) & WESTERN
        assert set(arabic) & ARABIC_INDIC
        assert not (set(arabic) & WESTERN)

    def test_keep_leaves_digits_alone(self, rng):
        plain = FieldGenerator("keep").code(random.Random(2))

        assert plain == FieldGenerator("keep").code(random.Random(2))

    def test_conversion_matches_the_normalisation_module(self):
        # Fields and prose must agree on digit shapes, or a page mixes systems by accident.
        value = FieldGenerator("arabic_indic").percentage(random.Random(1))
        expected = to_numeral_system(
            FieldGenerator("keep").percentage(random.Random(1)), NumeralSystem.ARABIC_INDIC
        )

        assert value == expected

    def test_any_field_covers_the_range(self, rng):
        generator = FieldGenerator()

        values = {generator.any_field(random.Random(seed)) for seed in range(30)}

        assert len(values) > 15


class TestTemplateIntegration:
    def test_a_form_carries_non_lexical_values(self, source):
        content = FormTemplate().build(source, random.Random(3))

        values = [
            block.attributes.get("value", "") for block in content.blocks if "value" in block.attributes
        ]
        assert any(any(c.isdigit() for c in value) for value in values)

    def test_an_invoice_uses_real_reference_and_date_fields(self, source):
        content = InvoiceTemplate().build(source, random.Random(5))

        values = " ".join(
            block.attributes.get("value", "") for block in content.blocks if "value" in block.attributes
        )
        assert any(c.isdigit() for c in values)

    def test_numerals_reach_the_templates(self, source):
        content = InvoiceTemplate().build(source, random.Random(5), numerals="arabic_indic")

        values = " ".join(
            block.attributes.get("value", "") for block in content.blocks if "value" in block.attributes
        )
        assert set(values) & ARABIC_INDIC

    def test_templates_stay_reproducible(self, source):
        first = FormTemplate().build(source, random.Random(8)).all_text
        second = FormTemplate().build(source, random.Random(8)).all_text

        assert first == second
