"""Contract for the line (recognition-strip) renderer and its factory.

The invariant is the project's: the annotation describes exactly what was drawn. On a
strip that has two specific teeth. The label must be the logical-order text of the one
line the strip is *about*, and the neighbour ink deliberately clipped into the margins
must never reach the annotated line — ink the label does not mention, sitting on top of
ink it does, is a wrong label rather than a realistic crop.
"""

from __future__ import annotations

import collections
import random
from pathlib import Path

import pytest

from ocrsmith.config.schema import GenerationConfig
from ocrsmith.core.fonts import load_font
from ocrsmith.core.lines import LineRenderer, LineStrip
from ocrsmith.pipeline import LineSampleFactory, build_factory

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
ARABIC_FONT = FONT_DIR / "NotoNaskhArabic-Regular.ttf"

pytestmark = pytest.mark.skipif(not ARABIC_FONT.exists(), reason="bundled fonts unavailable")

STRIP = LineStrip(text_height=40, pad_left=12, pad_right=12, pad_top=8, pad_bottom=8)


def config(**overrides) -> GenerationConfig:
    data = {"render": {"mode": "line"}, "run": {"num_samples": 8}}
    data.update(overrides)
    return GenerationConfig(**data)


class TestLineRenderer:
    def test_label_is_the_text_that_was_drawn(self):
        rendered = LineRenderer().render(
            "مرحبا بالعالم", load_font(ARABIC_FONT, 30), None, STRIP, rng=random.Random(0)
        )

        assert rendered.text == "مرحبا بالعالم"

    def test_strip_is_wider_than_it_is_tall_for_a_sentence(self):
        rendered = LineRenderer().render(
            "مرحبا بالعالم من المغرب", load_font(ARABIC_FONT, 30), None, STRIP, rng=random.Random(0)
        )
        width, height = rendered.image.size

        assert width > height

    def test_words_carry_their_own_boxes(self):
        rendered = LineRenderer().render(
            "مرحبا بالعالم", load_font(ARABIC_FONT, 30), None, STRIP, rng=random.Random(0)
        )

        assert len(rendered.line.words) == 2
        assert all(word.bbox.width > 0 for word in rendered.line.words)

    def test_boxes_lie_inside_the_strip(self):
        rendered = LineRenderer().render(
            "مرحبا بالعالم", load_font(ARABIC_FONT, 30), None, STRIP, rng=random.Random(0)
        )
        width, height = rendered.image.size

        for word in rendered.line.words:
            assert word.bbox.x0 >= 0 and word.bbox.x1 <= width
            assert word.bbox.y0 >= 0 and word.bbox.y1 <= height

    def test_as_page_is_a_one_line_page(self):
        rendered = LineRenderer().render(
            "مرحبا بالعالم", load_font(ARABIC_FONT, 30), None, STRIP, rng=random.Random(0)
        )
        page = rendered.as_page()

        assert (page.width, page.height) == rendered.image.size
        assert len(page.regions) == 1
        assert len(page.regions[0].lines) == 1
        assert page.text.strip() == "مرحبا بالعالم"

    def test_neighbour_bleed_never_reaches_the_annotated_line(self):
        """The whole point of slicing the neighbour instead of placing it off-canvas."""
        renderer = LineRenderer()
        strip = LineStrip(text_height=40, pad_left=10, pad_right=10, pad_top=14, pad_bottom=14)

        for seed in range(25):
            rendered = renderer.render(
                "مرحبا بالعالم",
                load_font(ARABIC_FONT, 30),
                None,
                strip,
                neighbours=("سطر فوق النص", "سطر تحت النص"),
                rng=random.Random(seed),
            )
            box = rendered.line.bbox
            height = rendered.image.size[1]
            # The neighbour may only occupy the margins, strictly outside the line box.
            assert box.y0 >= 1, f"seed {seed}: line starts at the very top edge"
            assert box.y1 <= height - 1, f"seed {seed}: line runs to the bottom edge"

    def test_neighbour_text_is_not_in_the_label(self):
        rendered = LineRenderer().render(
            "مرحبا بالعالم",
            load_font(ARABIC_FONT, 30),
            None,
            STRIP,
            neighbours=("نص الجار", "نص الجار"),
            rng=random.Random(4),
        )

        assert rendered.text == "مرحبا بالعالم"

    def test_empty_text_yields_an_empty_annotation_rather_than_raising(self):
        rendered = LineRenderer().render("   ", load_font(ARABIC_FONT, 30), None, STRIP, rng=random.Random(0))

        assert rendered.text == ""
        assert rendered.line.words == ()


class TestLineSampleFactory:
    def test_yields_exactly_one_sample_per_index(self):
        factory = LineSampleFactory(config())

        assert len(list(factory.create(0))) == 1

    def test_is_reproducible_from_the_seed_alone(self):
        first = list(LineSampleFactory(config()).create(41))[0]
        second = list(LineSampleFactory(config()).create(41))[0]

        assert first.text == second.text
        assert first.image.size == second.image.size
        assert first.image.tobytes() == second.image.tobytes()

    def test_different_indices_give_different_strips(self):
        factory = LineSampleFactory(config())
        texts = {list(factory.create(index))[0].text for index in range(20)}

        assert len(texts) > 10

    def test_label_matches_the_words_in_the_annotation(self):
        factory = LineSampleFactory(config())

        for index in range(15):
            for sample in factory.create(index):
                words = [word.text for word in sample.page.iter_words()]
                assert " ".join(words) == sample.text

    def test_text_height_is_honoured_within_the_configured_range(self):
        factory = LineSampleFactory(config(line={"text_height_range": (30, 34)}))

        for index in range(12):
            for sample in factory.create(index):
                assert 30 <= sample.provenance.extra["text_height"] <= 34

    def test_provenance_names_the_renderer_and_the_crop_shape(self):
        sample = list(LineSampleFactory(config()).create(2))[0]

        assert sample.provenance.renderer == "line"
        assert sample.provenance.extra["content"] in {
            "sentence",
            "window",
            "phrase",
            "contextless",
            "numeric",
            "fragment",
        }

    def test_the_corpus_is_not_set_in_one_face(self):
        """Font monotony is invisible in any single sample and ruins a corpus.

        It is also easy to reintroduce: anything that makes the coverage check reject
        most faces collapses the whole run onto whichever one survives, with no error.
        """
        factory = LineSampleFactory(config())
        faces = collections.Counter(
            Path(list(factory.create(index))[0].provenance.font_path).name for index in range(120)
        )

        assert len(faces) > 20, f"only {len(faces)} distinct faces used"
        assert faces.most_common(1)[0][1] < 30, f"one face dominates: {faces.most_common(3)}"

    def test_numeric_strips_appear(self):
        """Digits, dates and references carry the value on a document and are the
        hardest strings to read, because no language prior helps. A corpus of prose
        contains almost none of them, so the factory has to inject them."""
        factory = LineSampleFactory(config())
        shapes = collections.Counter(
            list(factory.create(index))[0].provenance.extra["content"] for index in range(120)
        )

        assert shapes["numeric"] > 0


class TestRenderModeDispatch:
    def test_build_factory_follows_the_configured_mode(self):
        from ocrsmith.pipeline import SampleFactory

        assert isinstance(build_factory(config()), LineSampleFactory)
        assert isinstance(build_factory(GenerationConfig()), SampleFactory)

    def test_document_remains_the_default(self):
        assert GenerationConfig().render.mode == "document"


class TestNumeralPolicy:
    """`text.normalization.numerals` has to reach the digits the factory invents.

    The corpus is converted at load time, but a `numeric` strip's digits are generated
    here — so without this they came out western on every such strip, which is precisely
    the sample whose entire content is digits.
    """

    def test_synthetic_numbers_follow_the_configured_system(self):
        import random

        arabic_indic = set("٠١٢٣٤٥٦٧٨٩")
        factory = LineSampleFactory(config(text={"normalization": {"numerals": "arabic_indic"}}))
        rng = random.Random(3)

        digits = set()
        for _ in range(80):
            kind, text = factory._content(rng)
            if kind == "numeric":
                digits |= {char for char in text if char.isdigit() or char in arabic_indic}

        assert digits, "no numeric strips were sampled"
        assert digits <= arabic_indic, f"western digits leaked through: {sorted(digits)}"

    def test_keep_leaves_the_digits_alone(self):
        import random

        factory = LineSampleFactory(config(text={"normalization": {"numerals": "keep"}}))
        rng = random.Random(3)

        texts = [text for _ in range(80) for kind, text in [factory._content(rng)] if kind == "numeric"]

        assert any(char in "0123456789" for text in texts for char in text)


class TestInlineSourceIsNormalisedToo:
    def test_the_inline_fallback_honours_the_numeral_setting(self):
        """A policy that holds for a CSV corpus and not for the inline one is a corpus
        whose labels depend on where the text was read from."""
        from ocrsmith.pipeline.factory import load_corpus_texts

        spec = GenerationConfig(
            text={
                "source": {"type": "inline", "sentences": ["في سنة 2024 صدر التقرير"]},
                "normalization": {"numerals": "arabic_indic"},
            }
        )

        assert load_corpus_texts(spec) == ["في سنة ٢٠٢٤ صدر التقرير"]

    def test_keep_still_round_trips_inline_text_unchanged(self):
        from ocrsmith.pipeline.factory import load_corpus_texts

        spec = GenerationConfig(text={"source": {"type": "inline", "sentences": ["في سنة 2024 صدر التقرير"]}})

        assert load_corpus_texts(spec) == ["في سنة 2024 صدر التقرير"]
