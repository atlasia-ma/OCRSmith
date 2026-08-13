"""Contract for the added document genres and handwriting setting.

Genre coverage is not decoration. A model trained only on flowing prose miscounts dot
leaders, misreads slide-sized type, and transfers poorly to the handwriting-heavy Arabic
benchmarks. Each template here exists because a benchmark asks for that shape.
"""

import random
from pathlib import Path

import pytest

from ocrsmith.core.documents import (
    ContentsTemplate,
    CorpusTextProvider,
    DocumentRenderer,
    NotesTemplate,
    PageSpec,
    SlideTemplate,
    TypographySampler,
    default_registry,
)
from ocrsmith.core.documents.typography import _looks_handwritten
from ocrsmith.domain import RegionType

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

pytestmark = pytest.mark.skipif(not FONT_DIR.exists(), reason="bundled fonts unavailable")

SENTENCES = [
    "المغرب بلد يقع في شمال إفريقيا ويطل على المحيط الأطلسي.",
    "تهدف أطلسيا إلى بناء نماذج ذكاء اصطناعي مفتوحة المصدر.",
    "يحتوي هذا التقرير على جداول وأرقام ومعلومات إضافية مفيدة.",
    "تم إنشاء هذه الصفحة تلقائيا لأغراض التدريب والتقييم.",
]


@pytest.fixture
def source():
    return CorpusTextProvider(SENTENCES)


@pytest.fixture
def typography():
    fonts = sorted(FONT_DIR.glob("NotoSansArabic-*.ttf"))
    return TypographySampler(fonts, body_size_range=(16, 18)).sample(random.Random(2))


class TestRegistry:
    def test_the_new_genres_are_registered(self):
        assert {"contents", "slide", "notes"} <= set(default_registry().names())

    def test_the_registry_still_samples_reproducibly(self):
        registry = default_registry()

        assert registry.sample(random.Random(8)).name == registry.sample(random.Random(8)).name


class TestContents:
    def test_entries_carry_dot_leaders_and_a_page_number(self, source):
        content = ContentsTemplate().build(source, random.Random(1))

        entries = [b for b in content.blocks if b.type is RegionType.PARAGRAPH]
        assert entries
        assert all(".." in block.text for block in entries)
        assert all(block.text.rstrip()[-1].isdigit() for block in entries)

    def test_page_numbers_increase_down_the_list(self, source):
        content = ContentsTemplate().build(source, random.Random(4))

        numbers = [
            int(b.text.rsplit(" ", 1)[1])
            for b in content.blocks
            if b.type is RegionType.PARAGRAPH and b.text.rsplit(" ", 1)[-1].isdigit()
        ]
        assert numbers == sorted(numbers)

    def test_leader_runs_vary_in_length(self, source):
        content = ContentsTemplate().build(source, random.Random(7))

        lengths = {block.text.count(".") for block in content.blocks if "." in block.text}
        assert len(lengths) > 1, "identical leaders would not exercise run-length counting"


class TestSlide:
    def test_a_slide_is_a_headline_and_bullets(self, source):
        content = SlideTemplate().build(source, random.Random(3))

        types = [block.type for block in content.blocks]
        assert RegionType.TITLE in types
        assert RegionType.LIST in types

    def test_a_slide_is_sparse(self, source):
        content = SlideTemplate().build(source, random.Random(5))

        # Slides are a distinct visual regime: very large type, very little of it.
        assert len(content.all_text.split()) < 90


class TestNotes:
    def test_notes_are_marked_handwritten(self, source):
        content = NotesTemplate().build(source, random.Random(2))

        assert content.metadata["handwritten"] is True

    def test_notes_carry_prose(self, source):
        content = NotesTemplate().build(source, random.Random(6))

        assert any(block.type is RegionType.PARAGRAPH for block in content.blocks)


class TestHandwrittenTypography:
    def test_handwriting_faces_are_detected_by_name(self):
        assert _looks_handwritten("ArefRuqaa")
        assert _looks_handwritten("NotoKufiArabic")
        assert not _looks_handwritten("IBMPlexSansArabic")

    def test_a_handwritten_page_gets_baseline_and_gap_jitter(self):
        sampler = TypographySampler(sorted(FONT_DIR.glob("*.ttf")), body_size_range=(18, 20))

        printed = sampler.sample(random.Random(3), handwritten=False)
        written = sampler.sample(random.Random(3), handwritten=True)

        # A hand holds neither a constant baseline nor an even word gap.
        assert printed.body.style.baseline_jitter == 0.0
        assert written.body.style.baseline_jitter > 0.0
        assert written.body.style.word_spacing_jitter > 0.0

    def test_handwriting_prefers_a_handwriting_family(self):
        sampler = TypographySampler(sorted(FONT_DIR.glob("*.ttf")), body_size_range=(18, 20))
        if not sampler.handwriting_families:
            pytest.skip("no handwriting-like family in the bundled fonts")

        names = {
            Path(sampler.sample(random.Random(seed), handwritten=True).body.font.path).stem
            for seed in range(6)
        }

        assert all(_looks_handwritten(name) for name in names)

    def test_it_falls_back_gracefully_when_no_handwriting_face_exists(self):
        sampler = TypographySampler(
            sorted(FONT_DIR.glob("IBMPlexSansArabic-*.ttf")), body_size_range=(18, 20)
        )

        assert sampler.handwriting_families == ()
        assert sampler.sample(random.Random(1), handwritten=True) is not None

    def test_handwriting_is_set_more_loosely(self):
        sampler = TypographySampler(sorted(FONT_DIR.glob("*.ttf")), body_size_range=(18, 20))

        printed = sampler.sample(random.Random(11), handwritten=False)
        written = sampler.sample(random.Random(11), handwritten=True)

        assert written.body.style.line_spacing > printed.body.style.line_spacing


class TestRendering:
    @pytest.mark.parametrize(
        "template", [ContentsTemplate(), SlideTemplate(), NotesTemplate()], ids=lambda t: t.name
    )
    def test_each_genre_renders_to_a_valid_page(self, template, source, typography):
        content = template.build(source, random.Random(3))

        pages = DocumentRenderer().render(
            content, PageSpec.from_paper("a5", dpi=100), typography, rng=random.Random(1)
        )

        assert pages
        for page in pages:
            assert page.page.regions
            for region in page.page.regions:
                assert page.page.bbox.contains(region.bbox)
