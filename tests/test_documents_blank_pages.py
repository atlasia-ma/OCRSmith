"""A page that describes nothing must never be emitted.

A block that fits in no column of a given page shape used to produce a blank image on
every remaining page, each carrying an annotation that claimed content the image did not
have. The block is dropped instead, because a page with a confident label and no ink is
worse than a page that is missing.
"""

import random
from pathlib import Path

import pytest

from ocrsmith.core.documents import (
    DocumentBuilder,
    DocumentRenderer,
    PageSpec,
    TypographySampler,
)

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

pytestmark = pytest.mark.skipif(not FONT_DIR.exists(), reason="bundled fonts unavailable")


@pytest.fixture
def typography():
    fonts = sorted(FONT_DIR.glob("NotoSansArabic-*.ttf"))
    return TypographySampler(fonts, body_size_range=(16, 18)).sample(random.Random(3))


@pytest.fixture
def renderer():
    return DocumentRenderer()


def test_every_emitted_page_carries_at_least_one_region(renderer, typography):
    content = DocumentBuilder().title("Title").paragraph("Body text goes here.").build()

    pages = renderer.render(content, PageSpec.from_paper("a6", dpi=90), typography, max_pages=10)

    assert pages
    for page in pages:
        assert page.page.regions


def test_an_unplaceable_block_is_dropped_rather_than_blanking_pages(renderer, typography):
    # A table far too wide for the column fits nowhere, and tables never split.
    rows = [[f"column-{index}" * 4 for index in range(12)] for _ in range(3)]
    content = DocumentBuilder().paragraph("Readable prose.").table(rows).build()

    pages = renderer.render(content, PageSpec.from_paper("a6", dpi=80), typography, max_pages=10)

    assert pages
    assert all(page.page.regions for page in pages)


def test_page_numbers_stay_dense_when_a_block_is_dropped(renderer, typography):
    rows = [[f"wide-{index}" * 5 for index in range(10)] for _ in range(2)]
    content = DocumentBuilder().paragraph("First.").table(rows).paragraph("Second.").build()

    pages = renderer.render(content, PageSpec.from_paper("a6", dpi=80), typography, max_pages=10)

    assert [page.number for page in pages] == list(range(1, len(pages) + 1))


def test_a_document_of_nothing_but_unplaceable_content_yields_no_pages(renderer, typography):
    rows = [[f"enormous-{index}" * 8 for index in range(14)]]
    content = DocumentBuilder().table(rows).build()

    pages = renderer.render(content, PageSpec.from_paper("a6", dpi=80), typography, max_pages=10)

    assert pages == []
