"""Contract for the annotation model.

These objects are the ground truth. If a box drifts, a reading order flips or a table
serialises wrongly, the dataset teaches the wrong thing — so the invariants are pinned
here rather than discovered during training.
"""

import pytest
from PIL import Image

from ocrsmith.domain import (
    BBox,
    Line,
    Page,
    Provenance,
    Region,
    RegionType,
    Sample,
    Table,
    TableCell,
    Word,
    assign_reading_order,
)
from ocrsmith.text import Direction


def word(text, x0, y0, x1, y1):
    return Word(text, BBox(x0, y0, x1, y1))


def line(text, x0, y0, x1, y1, words=(), direction=Direction.LTR):
    return Line(text, BBox(x0, y0, x1, y1), tuple(words), direction)


def region(kind, x0, y0, x1, y1, lines=(), order=0, table=None, attributes=None):
    return Region(kind, BBox(x0, y0, x1, y1), tuple(lines), table, order, attributes or {})


class TestWordAndLine:
    def test_translate_moves_word_and_its_box(self):
        moved = word("مرحبا", 0, 0, 10, 10).translate(5, 7)

        assert moved.bbox == BBox(5, 7, 15, 17)
        assert moved.text == "مرحبا"

    def test_translate_moves_nested_words(self):
        source = line("a b", 0, 0, 20, 10, [word("a", 0, 0, 8, 10), word("b", 12, 0, 20, 10)])

        moved = source.translate(0, 100)

        assert [w.bbox.y0 for w in moved.words] == [100, 100]

    def test_line_serialisation_includes_words_and_direction(self):
        data = line("مرحبا", 0, 0, 10, 10, [word("مرحبا", 0, 0, 10, 10)], Direction.RTL).to_dict()

        assert data["direction"] == "rtl"
        assert data["words"][0]["text"] == "مرحبا"
        assert data["bbox"] == [0, 0, 10, 10]

    def test_line_serialisation_omits_absent_optional_fields(self):
        assert "words" not in line("x", 0, 0, 1, 1).to_dict()
        assert "baseline" not in line("x", 0, 0, 1, 1).to_dict()


class TestRegion:
    def test_text_joins_lines_with_newlines(self):
        block = region(
            RegionType.PARAGRAPH, 0, 0, 10, 30, [line("one", 0, 0, 10, 10), line("two", 0, 10, 10, 20)]
        )

        assert block.text == "one\ntwo"

    def test_words_are_flattened_across_lines(self):
        block = region(
            RegionType.PARAGRAPH,
            0,
            0,
            20,
            20,
            [line("a b", 0, 0, 20, 10, [word("a", 0, 0, 8, 10), word("b", 12, 0, 20, 10)])],
        )

        assert [w.text for w in block.words] == ["a", "b"]

    def test_translate_is_deep(self):
        block = region(RegionType.PARAGRAPH, 0, 0, 10, 10, [line("x", 0, 0, 10, 10)]).translate(3, 4)

        assert block.bbox == BBox(3, 4, 13, 14)
        assert block.lines[0].bbox == BBox(3, 4, 13, 14)


class TestTable:
    @pytest.fixture
    def simple_table(self):
        cells = (
            TableCell(0, 0, "المؤشر", is_header=True),
            TableCell(0, 1, "القيمة", is_header=True),
            TableCell(1, 0, "A"),
            TableCell(1, 1, "42"),
        )
        return Table(rows=2, cols=2, cells=cells, has_header_row=True)

    def test_cell_lookup(self, simple_table):
        assert simple_table.cell_at(1, 1).text == "42"
        assert simple_table.cell_at(5, 5) is None

    def test_html_marks_header_cells(self, simple_table):
        html = simple_table.to_html()

        assert html.startswith("<table><tr><th>")
        assert "<td>42</td>" in html
        assert html.count("<tr>") == 2

    def test_html_escapes_cell_text(self):
        table = Table(1, 1, (TableCell(0, 0, "a <b> & c"),))

        assert "&lt;b&gt;" in table.to_html()
        assert "&amp;" in table.to_html()

    def test_html_emits_span_attributes(self):
        table = Table(2, 2, (TableCell(0, 0, "wide", col_span=2), TableCell(1, 0, "x"), TableCell(1, 1, "y")))

        assert 'colspan="2"' in table.to_html()

    def test_otsl_describes_the_grid_row_by_row(self, simple_table):
        assert simple_table.to_otsl() == "fcel fcel nl fcel fcel nl"

    def test_otsl_marks_empty_cells(self):
        table = Table(1, 2, (TableCell(0, 0, "x"), TableCell(0, 1, "")))

        assert table.to_otsl() == "fcel ecel nl"

    def test_otsl_marks_horizontal_spans(self):
        table = Table(1, 3, (TableCell(0, 0, "wide", col_span=2), TableCell(0, 2, "z")))

        assert table.to_otsl() == "fcel lcel fcel nl"

    def test_otsl_marks_vertical_spans(self):
        table = Table(2, 1, (TableCell(0, 0, "tall", row_span=2),))

        assert table.to_otsl() == "fcel nl ucel nl"

    def test_markdown_includes_a_header_rule(self, simple_table):
        assert "| --- | --- |" in simple_table.to_markdown()


class TestPageText:
    @pytest.fixture
    def article(self):
        return Page(
            width=600,
            height=800,
            regions=(
                region(RegionType.TITLE, 0, 0, 600, 40, [line("العنوان", 0, 0, 600, 40)], order=0),
                region(
                    RegionType.PARAGRAPH,
                    0,
                    50,
                    600,
                    90,
                    [line("فقرة أولى", 0, 50, 600, 90)],
                    order=1,
                ),
            ),
            direction=Direction.RTL,
        )

    def test_text_follows_reading_order(self, article):
        assert article.text == "العنوان\nفقرة أولى"

    def test_text_ignores_declaration_order(self):
        page = Page(
            10,
            10,
            (
                region(RegionType.PARAGRAPH, 0, 5, 10, 10, [line("second", 0, 5, 10, 10)], order=1),
                region(RegionType.TITLE, 0, 0, 10, 5, [line("first", 0, 0, 10, 5)], order=0),
            ),
        )

        assert page.text == "first\nsecond"

    def test_iter_words_walks_the_whole_page(self, article):
        page = article.with_regions(
            [
                region(
                    RegionType.PARAGRAPH,
                    0,
                    0,
                    10,
                    10,
                    [line("a b", 0, 0, 10, 10, [word("a", 0, 0, 4, 10), word("b", 6, 0, 10, 10)])],
                )
            ]
        )

        assert [w.text for w in page.iter_words()] == ["a", "b"]

    def test_regions_of_filters_by_type(self, article):
        assert len(article.regions_of(RegionType.TITLE)) == 1
        assert len(article.regions_of(RegionType.TABLE)) == 0


class TestPageMarkup:
    def test_html_maps_region_types_to_tags(self):
        page = Page(
            100,
            100,
            (
                region(RegionType.TITLE, 0, 0, 100, 10, [line("Title", 0, 0, 100, 10)], order=0),
                region(RegionType.PARAGRAPH, 0, 10, 100, 20, [line("Body", 0, 10, 100, 20)], order=1),
                region(
                    RegionType.LIST,
                    0,
                    20,
                    100,
                    40,
                    [line("one", 0, 20, 100, 30), line("two", 0, 30, 100, 40)],
                    order=2,
                ),
            ),
        )

        html = page.to_html()

        assert "<h1>Title</h1>" in html
        assert "<p>Body</p>" in html
        assert "<ul><li>one</li><li>two</li></ul>" in html

    def test_html_delegates_tables(self):
        table = Table(1, 1, (TableCell(0, 0, "x"),))
        page = Page(10, 10, (region(RegionType.TABLE, 0, 0, 10, 10, table=table),))

        assert page.to_html() == "<table><tr><td>x</td></tr></table>"

    def test_html_escapes_text_content(self):
        page = Page(10, 10, (region(RegionType.PARAGRAPH, 0, 0, 10, 10, [line("a <b>", 0, 0, 10, 10)]),))

        assert page.to_html() == "<p>a &lt;b&gt;</p>"

    def test_markdown_uses_heading_prefixes(self):
        page = Page(
            100,
            100,
            (
                region(RegionType.TITLE, 0, 0, 100, 10, [line("Title", 0, 0, 100, 10)], order=0),
                region(RegionType.HEADING, 0, 10, 100, 20, [line("Section", 0, 10, 100, 20)], order=1),
                region(RegionType.PARAGRAPH, 0, 20, 100, 30, [line("Body", 0, 20, 100, 30)], order=2),
            ),
        )

        assert page.to_markdown() == "# Title\n\n## Section\n\nBody"

    def test_markdown_renders_lists_and_rules(self):
        page = Page(
            100,
            100,
            (
                region(RegionType.LIST, 0, 0, 100, 20, [line("one", 0, 0, 100, 10)], order=0),
                region(RegionType.SEPARATOR, 0, 20, 100, 21, order=1),
            ),
        )

        assert page.to_markdown() == "- one\n\n---"


class TestPageGeometry:
    def test_clipped_trims_boxes_to_the_canvas(self):
        page = Page(
            100, 100, (region(RegionType.PARAGRAPH, -10, -10, 150, 50, [line("x", -10, -10, 150, 50)]),)
        )

        clipped = page.clipped()

        assert clipped.regions[0].bbox == BBox(0, 0, 100, 50)
        assert clipped.regions[0].lines[0].bbox == BBox(0, 0, 100, 50)

    def test_clipped_drops_regions_entirely_off_page(self):
        page = Page(100, 100, (region(RegionType.PARAGRAPH, 200, 200, 300, 300),))

        assert page.clipped().regions == ()

    def test_page_bbox_covers_the_canvas(self):
        assert Page(640, 480).bbox == BBox(0, 0, 640, 480)


class TestReadingOrder:
    def test_left_to_right_pages_read_left_then_down(self):
        regions = [
            region(RegionType.PARAGRAPH, 50, 0, 90, 10, [line("right", 50, 0, 90, 10)]),
            region(RegionType.PARAGRAPH, 0, 0, 40, 10, [line("left", 0, 0, 40, 10)]),
            region(RegionType.PARAGRAPH, 0, 20, 40, 30, [line("below", 0, 20, 40, 30)]),
        ]

        ordered = assign_reading_order(regions, Direction.LTR)

        assert [r.text for r in ordered] == ["left", "right", "below"]

    def test_right_to_left_pages_read_right_then_down(self):
        regions = [
            region(RegionType.PARAGRAPH, 0, 0, 40, 10, [line("left", 0, 0, 40, 10)]),
            region(RegionType.PARAGRAPH, 50, 0, 90, 10, [line("right", 50, 0, 90, 10)]),
        ]

        ordered = assign_reading_order(regions, Direction.RTL)

        assert [r.text for r in ordered] == ["right", "left"]

    def test_reading_order_indices_are_dense_and_sequential(self):
        regions = [region(RegionType.PARAGRAPH, 0, y, 10, y + 5) for y in (30, 0, 15)]

        ordered = assign_reading_order(regions, Direction.LTR)

        assert [r.reading_order for r in ordered] == [0, 1, 2]


class TestSample:
    @pytest.fixture
    def sample(self):
        page = Page(20, 10, (region(RegionType.PARAGRAPH, 0, 0, 20, 10, [line("hi", 0, 0, 20, 10)]),))
        return Sample(
            id="sample_000001",
            image=Image.new("RGB", (20, 10)),
            page=page,
            provenance=Provenance(seed=7, font_path="a.ttf", renderer="horizontal"),
        )

    def test_text_delegates_to_the_page(self, sample):
        assert sample.text == "hi"

    def test_size_comes_from_the_page(self, sample):
        assert sample.size == (20, 10)

    def test_serialisation_excludes_pixels(self, sample):
        data = sample.to_dict(image_path="images/sample_000001.png")

        assert data["image_path"] == "images/sample_000001.png"
        assert data["provenance"]["seed"] == 7
        assert "image" not in data

    def test_serialisation_omits_unset_provenance_fields(self, sample):
        assert "background" not in sample.to_dict()["provenance"]

    def test_replace_page_keeps_identity_and_pixels(self, sample):
        replaced = sample.replace_page(Page(20, 10))

        assert replaced.id == sample.id
        assert replaced.image is sample.image
        assert replaced.page.regions == ()
