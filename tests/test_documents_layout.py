"""Contract for page geometry, content building and flow layout.

The invariants that make document-level ground truth trustworthy:

* content is never clipped — a block that does not fit moves or splits;
* nothing overlaps, and everything sits inside its column;
* the annotation's markup describes exactly the blocks that were drawn;
* the same content laid out on one column or two produces the same markup.
"""

import random
from pathlib import Path

import pytest

from ocrsmith.core.documents import (
    ArticleTemplate,
    CorpusTextProvider,
    DocumentBuilder,
    DocumentRenderer,
    InvoiceTemplate,
    PageSpec,
    TypographySampler,
    default_registry,
    group_font_families,
)
from ocrsmith.domain import RegionType
from ocrsmith.text import Direction

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

pytestmark = pytest.mark.skipif(not FONT_DIR.exists(), reason="bundled fonts unavailable")

ARABIC_SENTENCES = [
    "هذا نص تجريبي لتوليد المستندات الاصطناعية باللغة العربية.",
    "تهدف أطلسيا إلى بناء نماذج تعرف ضوئي عالية الجودة للدارجة المغربية.",
    "يحتوي هذا التقرير على جداول وأرقام ومعلومات إضافية مفيدة.",
    "تم إنشاء هذه الصفحة تلقائيا لأغراض التدريب والتقييم.",
]

LATIN_SENTENCES = [
    "This is a synthetic document rendered for OCR training purposes.",
    "The layout engine pours blocks down each column in turn.",
    "Nothing may be clipped, and every box must describe drawn pixels.",
    "Tables, figures and headings move whole rather than splitting.",
]


@pytest.fixture
def rng():
    return random.Random(20260812)


@pytest.fixture
def typography(rng):
    fonts = sorted(FONT_DIR.glob("NotoSansArabic-*.ttf"))
    return TypographySampler(fonts, body_size_range=(16, 18)).sample(rng)


@pytest.fixture
def renderer():
    return DocumentRenderer()


@pytest.fixture
def arabic_source():
    return CorpusTextProvider(ARABIC_SENTENCES)


@pytest.fixture
def latin_source():
    return CorpusTextProvider(LATIN_SENTENCES)


class TestPageSpec:
    def test_a4_at_150_dpi_is_about_1240_by_1754(self):
        spec = PageSpec.from_paper("a4", dpi=150)

        assert 1230 <= spec.width <= 1250
        assert 1740 <= spec.height <= 1765

    def test_landscape_swaps_the_axes(self):
        portrait = PageSpec.from_paper("a4", dpi=100)
        landscape = PageSpec.from_paper("a4", dpi=100, landscape=True)

        assert (landscape.width, landscape.height) == (portrait.height, portrait.width)

    def test_unknown_paper_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown paper size"):
            PageSpec.from_paper("a9")

    def test_margins_that_swallow_the_page_are_rejected(self):
        with pytest.raises(ValueError, match="no room"):
            PageSpec(width=100, height=100, margin_left=60, margin_right=60)

    def test_columns_tile_the_content_box_without_overlapping(self):
        spec = PageSpec(width=1000, height=1000, columns=3, column_gap=20)

        boxes = spec.column_boxes()

        assert len(boxes) == 3
        ordered = sorted(boxes, key=lambda b: b.x0)
        for left, right in zip(ordered, ordered[1:], strict=False):
            assert left.x1 <= right.x0
        assert ordered[0].x0 == spec.content_box.x0
        assert ordered[-1].x1 == pytest.approx(spec.content_box.x1)

    def test_right_to_left_pages_start_with_the_rightmost_column(self):
        spec = PageSpec(width=1000, height=1000, columns=2, direction=Direction.RTL)

        first, second = spec.column_boxes()

        assert first.x0 > second.x0

    def test_header_and_footer_bands_are_carved_out_of_the_content_box(self):
        spec = PageSpec(width=800, height=1000, header_height=50, footer_height=40)

        assert spec.content_box.y0 == spec.margin_top + 50
        assert spec.content_box.y1 == spec.height - spec.margin_bottom - 40

    def test_scaling_preserves_proportions(self):
        spec = PageSpec.from_paper("a4", dpi=100)

        doubled = spec.scaled(2.0)

        assert doubled.width == pytest.approx(spec.width * 2, abs=2)
        assert doubled.dpi == 200


class TestDocumentBuilder:
    def test_blocks_keep_the_order_they_were_added(self):
        content = DocumentBuilder().title("T").paragraph("P").heading("H").build()

        assert [b.type for b in content.blocks] == [
            RegionType.TITLE,
            RegionType.PARAGRAPH,
            RegionType.HEADING,
        ]

    def test_table_builds_a_full_cell_grid(self):
        content = DocumentBuilder().table([["a", "b"], ["c", "d"]]).build()
        table = content.blocks[0].table

        assert (table.rows, table.cols) == (2, 2)
        assert len(table.cells) == 4
        assert table.cell_at(0, 0).is_header

    def test_ragged_rows_are_padded_to_a_rectangle(self):
        table = DocumentBuilder().table([["a", "b", "c"], ["d"]]).build().blocks[0].table

        assert table.cols == 3
        assert table.cell_at(1, 2).text == ""

    def test_key_values_become_one_block_each(self):
        content = DocumentBuilder().key_values([("k1", "v1"), ("k2", "v2")]).build()

        assert len(content.blocks) == 2
        assert content.blocks[0].text == "k1: v1"
        assert content.blocks[0].attributes["key"] == "k1"

    def test_figure_with_a_caption_adds_both_blocks(self):
        content = DocumentBuilder().figure(100, 80, caption="a chart").build()

        assert [b.type for b in content.blocks] == [RegionType.FIGURE, RegionType.CAPTION]

    def test_direction_is_inferred_from_the_content(self):
        assert DocumentBuilder().title("مرحبا بالعالم").build().direction is Direction.RTL
        assert DocumentBuilder().title("hello world").build().direction is Direction.LTR

    def test_direction_can_be_forced(self):
        content = DocumentBuilder(Direction.RTL).title("hello").build()

        assert content.direction is Direction.RTL

    def test_prose_splits_but_structure_does_not(self):
        content = DocumentBuilder().paragraph("p").heading("h").table([["a"]]).build()
        paragraph, heading, table = content.blocks

        assert paragraph.is_splittable
        assert not heading.is_splittable
        assert not table.is_splittable


class TestFlowLayout:
    def test_a_short_document_fits_on_one_page(self, renderer, typography, rng):
        content = DocumentBuilder().title("Title").paragraph("Short body text.").build()

        pages = renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng)

        assert len(pages) == 1
        assert pages[0].number == 1

    def test_a_long_document_overflows_onto_further_pages(self, renderer, typography, rng, latin_source):
        content = ArticleTemplate(min_sections=8, max_sections=10).build(latin_source, rng)

        pages = renderer.render(content, PageSpec.from_paper("a6", dpi=100), typography, rng=rng)

        assert len(pages) > 1
        assert [p.number for p in pages] == list(range(1, len(pages) + 1))

    def test_every_region_stays_inside_the_canvas(self, renderer, typography, rng, arabic_source):
        content = ArticleTemplate().build(arabic_source, rng, direction=Direction.RTL)

        for page in renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng):
            for region in page.page.regions:
                assert page.page.bbox.contains(region.bbox)

    def test_body_regions_stay_inside_the_content_area(self, renderer, typography, rng, arabic_source):
        # Content must respect the margins and the header/footer bands, not merely the
        # canvas: text spilling into the footer is what a clipped layout looks like.
        spec = PageSpec.from_paper("a5", dpi=100, header_height=30, footer_height=30)
        content = ArticleTemplate(min_sections=4, max_sections=4).build(
            arabic_source, rng, direction=Direction.RTL
        )
        furniture = {RegionType.HEADER, RegionType.FOOTER, RegionType.PAGE_NUMBER}

        for page in renderer.render(content, spec, typography, rng=rng, max_pages=20):
            for region in page.page.regions:
                if region.type in furniture:
                    continue
                assert region.bbox.y1 <= spec.content_box.y1 + 2
                assert region.bbox.y0 >= spec.content_box.y0 - 2
                assert region.bbox.x0 >= spec.content_box.x0 - 2
                assert region.bbox.x1 <= spec.content_box.x1 + 2

    def test_regions_on_a_page_do_not_overlap(self, renderer, typography, rng, latin_source):
        content = ArticleTemplate().build(latin_source, rng)

        for page in renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng):
            boxes = [r.bbox for r in page.page.regions if r.type is not RegionType.SEPARATOR]
            for i, left in enumerate(boxes):
                for right in boxes[i + 1 :]:
                    assert left.iou(right) < 0.05

    def test_reading_order_is_dense_and_increasing(self, renderer, typography, rng, latin_source):
        content = ArticleTemplate().build(latin_source, rng)

        page = renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng)[0]

        orders = [r.reading_order for r in page.page.ordered_regions()]
        assert orders == sorted(orders)
        assert len(set(orders)) == len(orders)

    def test_no_text_is_lost_across_page_breaks(self, renderer, typography, rng, latin_source):
        content = ArticleTemplate(min_sections=6, max_sections=6).build(latin_source, rng)
        expected = len(" ".join(b.text for b in content.blocks).split())

        pages = renderer.render(
            content, PageSpec.from_paper("a6", dpi=100), typography, rng=rng, max_pages=40
        )
        rendered_words = sum(len(p.page.text.split()) for p in pages)

        # Bullet markers and page furniture can add tokens; nothing may go missing.
        assert rendered_words >= expected

    def test_two_columns_hold_more_per_page_than_one(self, renderer, typography, rng, latin_source):
        content = ArticleTemplate(min_sections=5, max_sections=5).build(latin_source, rng)
        one = PageSpec.from_paper("a5", dpi=100, columns=1)
        two = PageSpec.from_paper("a5", dpi=100, columns=2)

        single = renderer.render(content, one, typography, rng=random.Random(1), max_pages=40)
        double = renderer.render(content, two, typography, rng=random.Random(1), max_pages=40)

        assert len(double) <= len(single)

    def test_right_to_left_documents_fill_the_right_column_first(
        self, renderer, typography, rng, arabic_source
    ):
        content = ArticleTemplate(min_sections=3, max_sections=3).build(
            arabic_source, rng, direction=Direction.RTL
        )
        spec = PageSpec.from_paper("a5", dpi=100, columns=2, direction=Direction.RTL)

        page = renderer.render(content, spec, typography, rng=rng)[0]

        first = page.page.ordered_regions()[0]
        assert first.bbox.x0 > spec.width / 2

    def test_tables_are_placed_whole_with_cell_boxes(self, renderer, typography, rng):
        content = DocumentBuilder().table([["a", "b"], ["1", "2"], ["3", "4"]]).build()

        page = renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng)[0]

        tables = page.page.regions_of(RegionType.TABLE)
        assert len(tables) == 1
        cells = tables[0].table.cells
        assert all(cell.bbox is not None for cell in cells)
        assert all(tables[0].bbox.contains(cell.bbox) for cell in cells)

    def test_figures_are_annotated_even_though_they_are_placeholders(self, renderer, typography, rng):
        content = DocumentBuilder().figure(200, 120, caption="cap").build()

        page = renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng)[0]

        figures = page.page.regions_of(RegionType.FIGURE)
        assert len(figures) == 1
        assert figures[0].bbox.width > 0

    def test_running_header_and_page_number_are_drawn_in_their_bands(
        self, renderer, typography, rng, latin_source
    ):
        content = DocumentBuilder(metadata_footer=None).header("Running head").paragraph("Body").build()
        content = type(content)(content.blocks, content.direction, {"footer": "page {page}"})
        spec = PageSpec.from_paper("a5", dpi=100, header_height=40, footer_height=40)

        page = renderer.render(content, spec, typography, rng=rng)[0]

        assert page.page.regions_of(RegionType.HEADER)
        numbers = page.page.regions_of(RegionType.PAGE_NUMBER)
        assert numbers and "1" in numbers[0].text

    def test_pages_can_be_streamed_one_at_a_time(self, renderer, typography, rng, latin_source):
        content = ArticleTemplate(min_sections=6, max_sections=6).build(latin_source, rng)

        stream = renderer.iter_pages(
            content, PageSpec.from_paper("a6", dpi=100), typography, rng=rng, max_pages=40
        )

        assert next(stream).number == 1

    def test_max_pages_bounds_the_output(self, renderer, typography, rng, latin_source):
        content = ArticleTemplate(min_sections=10, max_sections=10).build(latin_source, rng)

        pages = renderer.render(content, PageSpec.from_paper("a6", dpi=100), typography, rng=rng, max_pages=2)

        assert len(pages) == 2


class TestMarkupGroundTruth:
    def test_html_reflects_the_blocks_that_were_drawn(self, renderer, typography, rng):
        content = DocumentBuilder().title("The Title").paragraph("Body text here.").build()

        page = renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng)[0]

        html = page.page.to_html()
        assert "<h1>The Title</h1>" in html
        assert "<p>Body text here.</p>" in html

    def test_markdown_is_produced_for_the_same_page(self, renderer, typography, rng):
        content = DocumentBuilder().title("T").paragraph("B").build()

        page = renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng)[0]

        assert page.page.to_markdown().startswith("# T")

    def test_tables_serialise_to_html_and_otsl(self, renderer, typography, rng):
        content = DocumentBuilder().table([["h1", "h2"], ["a", "b"]]).build()

        page = renderer.render(content, PageSpec.from_paper("a5", dpi=100), typography, rng=rng)[0]
        table = page.page.regions_of(RegionType.TABLE)[0].table

        assert "<th>h1</th>" in table.to_html()
        assert table.to_otsl() == "fcel fcel nl fcel fcel nl"


class TestTemplates:
    def test_default_registry_exposes_every_genre(self):
        registry = default_registry()

        assert set(registry.names()) >= {"article", "report", "invoice", "form", "letter"}

    def test_registry_sampling_is_reproducible(self, latin_source):
        registry = default_registry()

        first = registry.sample(random.Random(4)).name
        second = registry.sample(random.Random(4)).name

        assert first == second

    def test_unknown_template_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown template"):
            default_registry().get("origami")

    @pytest.mark.parametrize("template", list(default_registry()))
    def test_every_template_produces_usable_content(self, template, latin_source):
        content = template.build(latin_source, random.Random(3))

        assert len(content) > 0
        assert content.metadata["template"] == template.name

    def test_invoice_contains_a_table_and_key_values(self, latin_source):
        content = InvoiceTemplate().build(latin_source, random.Random(9))

        types = {block.type for block in content.blocks}
        assert RegionType.TABLE in types
        assert RegionType.KEY_VALUE in types

    def test_templates_are_deterministic_for_a_given_seed(self, latin_source):
        first = ArticleTemplate().build(latin_source, random.Random(5))
        second = ArticleTemplate().build(latin_source, random.Random(5))

        assert first.text == second.text


class TestTypography:
    def test_font_files_are_grouped_into_families(self):
        families = group_font_families(["Amiri-Bold.ttf", "Amiri-Regular.ttf", "Mada-Light.ttf"])

        by_name = {family.name: family for family in families}
        assert set(by_name) == {"Amiri", "Mada"}
        assert len(by_name["Amiri"].faces) == 2

    def test_faces_are_ranked_from_light_to_bold(self):
        family = group_font_families(["X-Bold.ttf", "X-Light.ttf", "X-Regular.ttf"])[0]

        assert family.light.stem.endswith("Light")
        assert family.bold.stem.endswith("Bold")

    def test_a_document_uses_one_family_throughout(self, rng):
        sampler = TypographySampler(sorted(FONT_DIR.glob("*.ttf")))

        typography = sampler.sample(rng)

        families = {
            Path(role.font.path).stem.split("-")[0] for role in [typography.body, *typography.roles.values()]
        }
        assert len(families) == 1

    def test_headings_are_larger_than_body_text(self, rng):
        sampler = TypographySampler(sorted(FONT_DIR.glob("*.ttf")))

        typography = sampler.sample(rng)

        assert typography.for_(RegionType.TITLE).font.size > typography.body.font.size
        assert typography.for_(RegionType.HEADING).font.size > typography.body.font.size

    def test_unknown_roles_fall_back_to_body(self, typography):
        assert typography.for_(RegionType.MARGINALIA) is typography.body

    def test_sampler_needs_fonts(self):
        with pytest.raises(ValueError, match="at least one font"):
            TypographySampler([])
