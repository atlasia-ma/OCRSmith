"""Contract for validation and dataset statistics.

Validators exist to catch failures that do not raise. Each test therefore builds the
specific broken sample the validator is meant to notice, and checks it is *not* accepted.
"""

import pytest
from PIL import Image, ImageDraw

from ocrsmith.domain import (
    BBox,
    Line,
    Page,
    Provenance,
    Region,
    RegionType,
    Sample,
    Word,
    page_from_dict,
)
from ocrsmith.quality import (
    BoxesInsidePage,
    DatasetStats,
    LegibleLineHeight,
    MinContrast,
    MinInkCoverage,
    NonEmptyText,
    NoOverlappingRegions,
    ValidationPipeline,
    default_validators,
)
from ocrsmith.text import Direction


def build_sample(
    *,
    text: str = "مرحبا بالعالم من أطلسيا",
    box: BBox = BBox(20, 20, 260, 60),
    size: tuple[int, int] = (320, 200),
    ink: tuple[int, int, int] = (10, 10, 10),
    paper: tuple[int, int, int] = (250, 248, 244),
    extra_regions: tuple[Region, ...] = (),
    template: str = "article",
    preset: str = "scan",
) -> Sample:
    image = Image.new("RGB", size, paper)
    if text.strip():
        # Stripes rather than a solid block: a real text region contains ink *and* paper,
        # and a validator measuring local contrast needs to see both.
        draw = ImageDraw.Draw(image)
        y = box.y0 + 2
        while y < box.y1 - 2:
            draw.rectangle((box.x0, y, box.x1, min(y + 3, box.y1)), fill=ink)
            y += 6
    words = tuple(
        Word(token, BBox(box.x0 + index * 10, box.y0, box.x0 + index * 10 + 8, box.y1))
        for index, token in enumerate(text.split())
    )
    lines = (Line(text, box, words, Direction.RTL),) if text.strip() else ()
    regions = (Region(RegionType.PARAGRAPH, box, lines, None, 0), *extra_regions)
    return Sample(
        id="00000001_01",
        image=image,
        page=Page(size[0], size[1], regions, Direction.RTL),
        provenance=Provenance(
            seed=1,
            template=template,
            background="paper",
            font_path="assets/fonts/X-Regular.ttf",
            extra={"index": 1, "page": 1, "preset": preset},
        ),
    )


def build_banded_sample(inks: tuple[tuple[int, int, int], ...]) -> Sample:
    """A page of stacked text blocks, one per ink, so blocks can differ in legibility."""
    paper = (250, 248, 244)
    image = Image.new("RGB", (320, 60 * len(inks) + 20), paper)
    draw = ImageDraw.Draw(image)
    regions = []
    for index, ink in enumerate(inks):
        box = BBox(20, 20 + index * 60, 300, 60 + index * 60)
        y = box.y0 + 2
        while y < box.y1 - 2:
            draw.rectangle((box.x0, y, box.x1, min(y + 3, box.y1)), fill=ink)
            y += 6
        words = (Word("كلمة", box),)
        line = Line("كلمة", box, words, Direction.RTL)
        regions.append(Region(RegionType.PARAGRAPH, box, (line,), None, 0))
    return Sample(
        id="00000001_01",
        image=image,
        page=Page(image.width, image.height, tuple(regions), Direction.RTL),
        provenance=Provenance(
            seed=1,
            template="article",
            background="paper",
            font_path="assets/fonts/X-Regular.ttf",
            extra={"index": 1, "page": 1, "preset": "archive"},
        ),
    )


class TestNonEmptyText:
    def test_accepts_a_normal_page(self):
        assert NonEmptyText().check(build_sample()).passed

    def test_rejects_an_empty_page(self):
        assert not NonEmptyText().check(build_sample(text="")).passed

    def test_rejects_a_single_short_word(self):
        assert not NonEmptyText().check(build_sample(text="ok")).passed

    def test_explains_itself(self):
        verdict = NonEmptyText().check(build_sample(text=""))

        assert "characters" in verdict.reason
        assert verdict.validator == "NonEmptyText"


class TestMinInkCoverage:
    def test_accepts_a_page_with_text_on_it(self):
        assert MinInkCoverage().check(build_sample()).passed

    def test_rejects_a_blank_page(self):
        blank = build_sample(text="")
        assert not MinInkCoverage().check(blank).passed

    def test_reports_the_measured_fraction(self):
        verdict = MinInkCoverage().check(build_sample())

        assert verdict.value is not None and 0 < verdict.value < 1


class TestMinContrast:
    def test_accepts_dark_text_on_light_paper(self):
        assert MinContrast().check(build_sample()).passed

    def test_rejects_text_the_same_shade_as_the_paper(self):
        washed = build_sample(ink=(246, 244, 240), paper=(250, 248, 244))

        assert not MinContrast().check(washed).passed

    def test_rejects_a_washed_out_body_under_a_heading_that_survived(self):
        """A faded scan keeps its heading dark long after the body is unrecoverable.

        Judging the page by its best block accepted exactly this, with a full
        transcription attached to text no reader could make out.
        """
        dark, faded = (10, 10, 10), (246, 244, 240)
        sample = build_banded_sample((dark, faded, faded, faded))

        assert not MinContrast().check(sample).passed

    def test_accepts_a_page_where_only_one_block_faded(self):
        dark, faded = (10, 10, 10), (246, 244, 240)
        sample = build_banded_sample((dark, dark, dark, faded))

        assert MinContrast().check(sample).passed


class TestBoxesInsidePage:
    def test_accepts_boxes_within_the_canvas(self):
        assert BoxesInsidePage().check(build_sample()).passed

    def test_rejects_a_region_hanging_off_the_page(self):
        sample = build_sample()
        rogue = Region(RegionType.PARAGRAPH, BBox(300, 10, 400, 40), (), None, 1)
        broken = sample.replace_page(sample.page.with_regions([*sample.page.regions, rogue]))

        assert not BoxesInsidePage().check(broken).passed

    def test_rejects_a_word_outside_the_page(self):
        sample = build_sample()
        line = Line("x", BBox(0, 0, 10, 10), (Word("x", BBox(-50, 0, -10, 10)),))
        region = Region(RegionType.PARAGRAPH, BBox(0, 0, 10, 10), (line,), None, 1)
        broken = sample.replace_page(sample.page.with_regions([region]))

        assert not BoxesInsidePage().check(broken).passed


class TestNoOverlappingRegions:
    def test_accepts_a_clean_layout(self):
        assert NoOverlappingRegions().check(build_sample()).passed

    def test_rejects_stacked_blocks(self):
        overlapping = Region(
            RegionType.PARAGRAPH,
            BBox(20, 20, 260, 60),
            (Line("again", BBox(20, 20, 260, 60)),),
            None,
            1,
        )
        sample = build_sample(extra_regions=(overlapping,))

        assert not NoOverlappingRegions().check(sample).passed


class TestLegibleLineHeight:
    def test_accepts_normal_lines(self):
        assert LegibleLineHeight().check(build_sample()).passed

    def test_rejects_text_too_small_to_resolve(self):
        tiny = build_sample(box=BBox(20, 20, 260, 24))

        assert not LegibleLineHeight(min_height=8.0).check(tiny).passed


class TestValidationPipeline:
    def test_a_good_sample_passes_every_check(self):
        report = default_validators().check(build_sample())

        assert report.passed
        assert report.failures == ()

    def test_a_broken_sample_names_the_check_that_failed(self):
        report = default_validators().check(build_sample(text=""))

        assert not report.passed
        assert report.failures[0].validator in {"NonEmptyText", "MinInkCoverage"}

    def test_fail_fast_stops_at_the_first_failure(self):
        pipeline = ValidationPipeline([NonEmptyText(), MinInkCoverage()], fail_fast=True)

        report = pipeline.check(build_sample(text=""))

        assert len(report.verdicts) == 1

    def test_full_run_collects_every_verdict(self):
        pipeline = ValidationPipeline([NonEmptyText(), MinInkCoverage()], fail_fast=False)

        report = pipeline.check(build_sample(text=""))

        assert len(report.verdicts) == 2

    def test_filter_drops_bad_samples_and_reports_them(self):
        pipeline = default_validators()
        rejected = []

        kept = list(pipeline.filter([build_sample(), build_sample(text="")], on_reject=rejected.append))

        assert len(kept) == 1
        assert len(rejected) == 1

    def test_reports_serialise_for_logging(self):
        data = default_validators().check(build_sample(text="")).to_dict()

        assert data["passed"] is False
        assert data["failures"][0]["reason"]


class TestDatasetStats:
    def test_counts_pages_documents_and_text(self):
        stats = DatasetStats().add(build_sample()).add(build_sample())

        data = stats.to_dict()
        assert data["pages"] == 2
        assert data["documents"] == 1  # both pages came from document index 1
        assert data["words"] > 0
        assert data["characters"] > 0

    def test_tracks_the_distributions_that_define_a_corpus(self):
        stats = DatasetStats()
        stats.add(build_sample(template="article", preset="scan"))
        stats.add(build_sample(template="invoice", preset="photo"))

        data = stats.to_dict()
        assert data["templates"] == {"article": 1, "invoice": 1}
        assert data["degradation_presets"] == {"scan": 1, "photo": 1}
        assert data["directions"] == {"rtl": 2}

    def test_reports_the_alphabet_it_has_seen(self):
        stats = DatasetStats().add(build_sample(text="ab ab"))

        assert stats.alphabet_size == 3  # 'a', 'b', ' '

    def test_line_height_percentiles_are_ordered(self):
        stats = DatasetStats()
        for height in (10, 20, 30, 40):
            stats.add(build_sample(box=BBox(20, 20, 260, 20 + height)))

        percentiles = stats.line_height_percentiles
        assert percentiles["p5"] <= percentiles["p50"] <= percentiles["p95"]

    def test_markdown_summary_is_pasteable(self):
        markdown = DatasetStats().add(build_sample()).to_markdown()

        assert markdown.startswith("## Dataset statistics")
        assert "Capture conditions" in markdown

    def test_accumulates_from_serialised_records_too(self):
        sample = build_sample()
        record = sample.to_dict(image_path="images/x.png")

        stats = DatasetStats().add_record(record)

        assert stats.pages == 1
        assert stats.words == len(sample.text.split())

    def test_empty_stats_do_not_divide_by_zero(self):
        assert DatasetStats().to_dict()["pages"] == 0
        assert DatasetStats().line_height_percentiles == {}


class TestPageRoundTrip:
    def test_a_page_survives_serialisation(self):
        page = build_sample().page

        restored = page_from_dict(page.to_dict())

        assert restored.to_dict() == page.to_dict()

    def test_words_and_direction_survive(self):
        page = build_sample().page

        restored = page_from_dict(page.to_dict())

        assert [w.text for w in restored.iter_words()] == [w.text for w in page.iter_words()]
        assert restored.direction is Direction.RTL

    def test_tables_survive(self):
        from ocrsmith.domain import Table, TableCell

        table = Table(1, 2, (TableCell(0, 0, "a", BBox(0, 0, 5, 5)), TableCell(0, 1, "b")), True)
        page = Page(50, 50, (Region(RegionType.TABLE, BBox(0, 0, 20, 20), (), table, 0),))

        restored = page_from_dict(page.to_dict())

        assert restored.regions[0].table.to_otsl() == table.to_otsl()

    def test_an_unknown_region_type_is_rejected_loudly(self):
        data = build_sample().page.to_dict()
        data["regions"][0]["type"] = "hieroglyph"

        with pytest.raises(ValueError):
            page_from_dict(data)
