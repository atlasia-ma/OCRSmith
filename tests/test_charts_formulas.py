"""Contract for charts and formulas.

Both follow the same principle: the **data is the input**, and the drawing is derived from
it. Neither is produced by parsing the other, so the picture and its ground truth cannot
drift apart — which is exactly the failure that makes hand-annotated chart and formula
corpora unreliable.
"""

import json
import random
from pathlib import Path

import pytest

from ocrsmith.core.documents import (
    Chart,
    ChartKind,
    ChartRenderer,
    ChartSeries,
    DocumentBuilder,
    DocumentRenderer,
    FormulaRenderer,
    PageSpec,
    TypographySampler,
    sample_chart,
    sample_formula,
)
from ocrsmith.core.documents.formulas import (
    MATH_GLYPHS,
    BigOperator,
    Fraction,
    Row,
    Sqrt,
    Subscript,
    Superscript,
    Symbol,
    choose_math_font,
)
from ocrsmith.domain import RegionType

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

pytestmark = pytest.mark.skipif(not FONT_DIR.exists(), reason="bundled fonts unavailable")


@pytest.fixture
def math_font():
    return choose_math_font(sorted(FONT_DIR.glob("*.ttf")))


@pytest.fixture
def text_font():
    from ocrsmith.core.fonts import load_font

    return load_font(FONT_DIR / "NotoSansArabic-Regular.ttf", 18)


class TestFormulaLatex:
    """LaTeX is derived from the tree, so it always describes what was drawn."""

    def test_fraction(self):
        assert Fraction(Symbol("a"), Symbol("b")).latex() == r"\frac{a}{b}"

    def test_superscript_and_subscript(self):
        assert Superscript(Symbol("x"), Symbol("2")).latex() == "x^{2}"
        assert Subscript(Symbol("a"), Symbol("i")).latex() == "a_{i}"

    def test_root(self):
        assert Sqrt(Symbol("2")).latex() == r"\sqrt{2}"

    def test_big_operator_carries_its_limits(self):
        node = BigOperator("sum", Symbol("x"), Symbol("i=1"), Symbol("n"))

        assert node.latex() == r"\sum_{i=1}^{n} x"

    def test_nesting_composes(self):
        node = Row(
            (Superscript(Symbol("a"), Symbol("2")), Symbol("+"), Sqrt(Fraction(Symbol("b"), Symbol("c"))))
        )

        assert node.latex() == r"a^{2} + \sqrt{\frac{b}{c}}"


class TestFormulaRendering:
    def test_a_formula_produces_pixels_and_latex_together(self, math_font):
        rendered = FormulaRenderer(math_font, 26).render(Fraction(Symbol("1"), Symbol("2")))

        assert rendered.size[0] > 4 and rendered.size[1] > 4
        assert rendered.latex == r"\frac{1}{2}"

    def test_a_fraction_is_taller_than_its_parts(self, math_font):
        renderer = FormulaRenderer(math_font, 26)

        plain = renderer.render(Symbol("1")).size[1]
        fraction = renderer.render(Fraction(Symbol("1"), Symbol("2"))).size[1]

        assert fraction > plain

    def test_a_sum_with_limits_is_taller_than_a_bare_symbol(self, math_font):
        renderer = FormulaRenderer(math_font, 24)

        bare = renderer.render(Symbol("x")).size[1]
        summed = renderer.render(BigOperator("sum", Symbol("x"), Symbol("i=1"), Symbol("n"))).size[1]

        assert summed > bare

    def test_rendering_is_reproducible(self, math_font):
        renderer = FormulaRenderer(math_font, 24)
        node = sample_formula(random.Random(4))

        assert renderer.render(node).latex == renderer.render(node).latex

    def test_sampling_is_reproducible(self):
        assert sample_formula(random.Random(9)).latex() == sample_formula(random.Random(9)).latex()


class TestMathFontSelection:
    def test_a_font_covering_the_operators_is_chosen(self, math_font):
        from ocrsmith.text.coverage import supports_text

        assert supports_text(math_font, MATH_GLYPHS).is_complete

    def test_choosing_from_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="draw mathematics"):
            choose_math_font([])

    def test_the_best_available_face_wins_when_none_is_complete(self):
        # Selection must degrade to the least-bad face rather than the first one.
        mono = FONT_DIR / "NotoSansMono-Light.ttf"
        arabic = FONT_DIR / "IBMPlexSansArabic-Regular.ttf"
        if not mono.exists() or not arabic.exists():
            pytest.skip("expected fonts unavailable")

        assert Path(choose_math_font([mono, arabic])).name != mono.name


class TestChartData:
    def test_the_json_target_names_every_series(self):
        chart = Chart(
            ChartKind.BAR,
            "Sales",
            ("Q1", "Q2"),
            (ChartSeries("north", (1.0, 2.0)), ChartSeries("south", (3.0, 4.0))),
        )

        data = chart.to_dict()

        assert data["type"] == "bar"
        assert [s["name"] for s in data["series"]] == ["north", "south"]
        assert data["series"][1]["values"] == [3.0, 4.0]

    def test_the_json_round_trips(self):
        chart = sample_chart(random.Random(2), ["a", "b", "c"], title="t")

        assert json.loads(json.dumps(chart.to_dict(), ensure_ascii=False)) == chart.to_dict()

    def test_sampling_is_reproducible(self):
        first = sample_chart(random.Random(6), ["a", "b"], title="t").to_dict()
        second = sample_chart(random.Random(6), ["a", "b"], title="t").to_dict()

        assert first == second

    @pytest.mark.parametrize("kind", list(ChartKind))
    def test_every_kind_draws_something(self, kind, text_font):
        chart = sample_chart(random.Random(1), ["alpha", "beta", "gamma"], title="T", kind=kind)

        rendered = ChartRenderer().render(chart, text_font, width=300, height=220)

        assert rendered.size == (300, 220)
        assert rendered.chart.kind is kind

    def test_labels_are_real_annotated_text(self, text_font):
        chart = sample_chart(random.Random(3), ["alpha", "beta"], title="Report", kind=ChartKind.BAR)

        rendered = ChartRenderer().render(chart, text_font, width=340, height=240)

        # A chart contributes to recognition supervision too, not only chart-to-JSON.
        assert rendered.lines
        assert any("Report" in line.text for line in rendered.lines)

    def test_a_tiny_canvas_degrades_rather_than_crashing(self, text_font):
        # A canvas with no room for a plot must still produce a valid image; the renderer
        # enforces a floor of 32px per side rather than raising.
        chart = sample_chart(random.Random(1), ["a"], title="t")

        rendered = ChartRenderer().render(chart, text_font, width=40, height=30)

        assert rendered.size == (40, 32)
        assert rendered.chart is chart


class TestPageIntegration:
    @pytest.fixture
    def typography(self):
        fonts = sorted(FONT_DIR.glob("IBMPlexSansArabic-*.ttf"))
        return TypographySampler(fonts, body_size_range=(18, 20)).sample(random.Random(1))

    def test_a_chart_becomes_a_region_carrying_its_data(self, typography):
        chart = sample_chart(random.Random(4), ["a", "b", "c"], title="Chart")
        content = DocumentBuilder().paragraph("Intro.").chart(chart, 300, 220).build()

        page = DocumentRenderer().render(
            content, PageSpec.from_paper("a5", dpi=110), typography, rng=random.Random(1)
        )[0]

        charts = page.page.regions_of(RegionType.CHART)
        assert len(charts) == 1
        assert charts[0].attributes["chart"]["type"] == chart.kind.value

    def test_chart_json_reaches_the_markdown_ground_truth(self, typography):
        chart = sample_chart(random.Random(4), ["a", "b"], title="Chart")
        content = DocumentBuilder().chart(chart, 280, 200).build()

        page = DocumentRenderer().render(
            content, PageSpec.from_paper("a5", dpi=110), typography, rng=random.Random(1)
        )[0]

        markdown = page.page.to_markdown()
        assert "```json" in markdown
        assert json.loads(markdown.split("```json")[1].split("```")[0])["title"] == "Chart"

    def test_a_formula_becomes_a_region_carrying_its_latex(self, typography):
        node = Fraction(Symbol("a"), Symbol("b"))
        content = DocumentBuilder().paragraph("Intro.").formula(node).build()

        page = DocumentRenderer().render(
            content, PageSpec.from_paper("a5", dpi=110), typography, rng=random.Random(1)
        )[0]

        formulas = page.page.regions_of(RegionType.FORMULA)
        assert len(formulas) == 1
        assert formulas[0].attributes["latex"] == r"\frac{a}{b}"

    def test_formula_latex_reaches_the_markdown_ground_truth(self, typography):
        content = DocumentBuilder().formula(Sqrt(Symbol("2"))).build()

        page = DocumentRenderer().render(
            content, PageSpec.from_paper("a5", dpi=110), typography, rng=random.Random(1)
        )[0]

        assert r"$$\n\sqrt{2}\n$$".replace("\\n", "\n") in page.page.to_markdown()

    def test_regions_are_serialisable(self, typography):
        content = (
            DocumentBuilder()
            .chart(sample_chart(random.Random(1), ["a", "b"], title="C"), 260, 190)
            .formula(Superscript(Symbol("x"), Symbol("2")))
            .build()
        )

        page = DocumentRenderer().render(
            content, PageSpec.from_paper("a5", dpi=110), typography, rng=random.Random(1)
        )[0]

        # Everything on the page must survive a trip through JSON, or no writer can emit it.
        assert json.loads(json.dumps(page.page.to_dict(), ensure_ascii=False))
