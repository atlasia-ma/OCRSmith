"""Charts, drawn with the data that produced them.

Chart-to-JSON is a first-class task in the Arabic document benchmarks, and no synthetic
generator covers it for Arabic. That makes it the clearest differentiator available: a
chart rendered from known data is perfect supervision for a task where real annotation is
expensive and error-prone, because the ground truth is the input rather than something a
human squinted at afterwards.

Charts are drawn with plain Pillow rather than matplotlib. That keeps the dependency
footprint honest, but more importantly it keeps the *series values* the source of truth —
the drawing is derived from them, so the JSON cannot drift from the picture.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont

from ...domain.annotations import Line
from ...domain.geometry import BBox
from ...text.script import Direction
from ..rendering.metrics import metrics_for
from ..rendering.style import Alignment, TextStyle
from ..rendering.text_renderer import TextBlockRenderer

__all__ = ["Chart", "ChartKind", "ChartSeries", "RenderedChart", "ChartRenderer"]

#: Colour-blind-safe qualitative palette, so a chart stays legible in greyscale scans too.
_PALETTE = (
    (57, 106, 177),
    (218, 124, 48),
    (62, 150, 81),
    (204, 37, 41),
    (107, 76, 154),
    (146, 36, 40),
    (148, 139, 61),
)


class ChartKind(str, Enum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    PIE = "pie"


@dataclass(frozen=True, slots=True)
class ChartSeries:
    """One named series of values."""

    name: str
    values: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"name": self.name, "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class Chart:
    """A chart's data — the ground truth, from which the drawing is derived."""

    kind: ChartKind
    title: str
    categories: tuple[str, ...]
    series: tuple[ChartSeries, ...]
    x_label: str = ""
    y_label: str = ""

    def to_dict(self) -> dict:
        """The structured target a chart-to-JSON model is asked to predict."""
        return {
            "type": self.kind.value,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "categories": list(self.categories),
            "series": [item.to_dict() for item in self.series],
        }

    @property
    def maximum(self) -> float:
        return max((value for item in self.series for value in item.values), default=1.0) or 1.0


@dataclass(frozen=True, slots=True)
class RenderedChart:
    """A drawn chart, its data, and the text it contains."""

    image: Image.Image
    chart: Chart
    lines: tuple[Line, ...] = field(default_factory=tuple)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def translated(self, dx: float, dy: float) -> RenderedChart:
        return RenderedChart(self.image, self.chart, tuple(line.translate(dx, dy) for line in self.lines))


class ChartRenderer:
    """Draws a `Chart`, returning the text it contains as annotated lines.

    Labels are real text and are annotated like any other text on the page, so a chart
    contributes to recognition and detection supervision as well as to chart-to-JSON.
    """

    def __init__(self, text_renderer: TextBlockRenderer | None = None):
        self.text_renderer = text_renderer or TextBlockRenderer()

    def render(
        self,
        chart: Chart,
        font: FreeTypeFont,
        *,
        width: int,
        height: int,
        direction: Direction = Direction.LTR,
        rng: random.Random | None = None,
        ink: tuple[int, int, int] = (30, 30, 34),
    ) -> RenderedChart:
        rng = rng or random.Random()
        image = Image.new("RGBA", (max(32, width), max(32, height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        metrics = metrics_for(font, self.text_renderer.shaper)
        lines: list[Line] = []

        pad = max(8, int(min(width, height) * 0.06))
        title_height = int(metrics.natural_line_height * 1.4) if chart.title else 0
        if chart.title:
            lines.extend(
                self._label(image, chart.title, font, direction, pad, pad // 2, width - 2 * pad, rng)
            )

        plot = BBox(pad * 2, pad + title_height, width - pad, height - pad * 2)
        if plot.width < 16 or plot.height < 16:
            return RenderedChart(image, chart, tuple(lines))

        if chart.kind is ChartKind.PIE:
            self._draw_pie(draw, chart, plot)
        elif chart.kind is ChartKind.LINE:
            self._draw_axes(draw, plot, ink)
            self._draw_line_series(draw, chart, plot)
        elif chart.kind is ChartKind.HORIZONTAL_BAR:
            self._draw_axes(draw, plot, ink)
            self._draw_bars(draw, chart, plot, horizontal=True)
        else:
            self._draw_axes(draw, plot, ink)
            self._draw_bars(draw, chart, plot, horizontal=False)

        lines.extend(self._category_labels(image, chart, font, plot, direction, rng))
        return RenderedChart(image, chart, tuple(lines))

    # -- drawing primitives ------------------------------------------------

    @staticmethod
    def _draw_axes(draw: ImageDraw.ImageDraw, plot: BBox, ink) -> None:
        colour = (*ink, 255)
        draw.line((plot.x0, plot.y0, plot.x0, plot.y1), fill=colour, width=2)
        draw.line((plot.x0, plot.y1, plot.x1, plot.y1), fill=colour, width=2)

    @staticmethod
    def _draw_bars(draw: ImageDraw.ImageDraw, chart: Chart, plot: BBox, *, horizontal: bool) -> None:
        groups = max(1, len(chart.categories))
        per_group = max(1, len(chart.series))
        maximum = chart.maximum

        if horizontal:
            slot = plot.height / groups
            thickness = slot / (per_group + 1)
            for index, series in enumerate(chart.series):
                colour = (*_PALETTE[index % len(_PALETTE)], 255)
                for position, value in enumerate(series.values[:groups]):
                    length = (value / maximum) * (plot.width - 4)
                    top = plot.y0 + position * slot + index * thickness + thickness * 0.25
                    draw.rectangle(
                        (plot.x0 + 2, top, plot.x0 + 2 + length, top + thickness * 0.8), fill=colour
                    )
        else:
            slot = plot.width / groups
            thickness = slot / (per_group + 1)
            for index, series in enumerate(chart.series):
                colour = (*_PALETTE[index % len(_PALETTE)], 255)
                for position, value in enumerate(series.values[:groups]):
                    length = (value / maximum) * (plot.height - 4)
                    left = plot.x0 + position * slot + index * thickness + thickness * 0.25
                    draw.rectangle(
                        (left, plot.y1 - 2 - length, left + thickness * 0.8, plot.y1 - 2), fill=colour
                    )

    @staticmethod
    def _draw_line_series(draw: ImageDraw.ImageDraw, chart: Chart, plot: BBox) -> None:
        groups = max(1, len(chart.categories))
        maximum = chart.maximum
        step = plot.width / max(1, groups - 1) if groups > 1 else plot.width
        for index, series in enumerate(chart.series):
            colour = (*_PALETTE[index % len(_PALETTE)], 255)
            points = [
                (plot.x0 + position * step, plot.y1 - (value / maximum) * (plot.height - 4))
                for position, value in enumerate(series.values[:groups])
            ]
            if len(points) > 1:
                draw.line(points, fill=colour, width=3, joint="curve")
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)

    @staticmethod
    def _draw_pie(draw: ImageDraw.ImageDraw, chart: Chart, plot: BBox) -> None:
        values = list(chart.series[0].values) if chart.series else []
        total = sum(values) or 1.0
        diameter = min(plot.width, plot.height)
        cx, cy = plot.center
        box = (cx - diameter / 2, cy - diameter / 2, cx + diameter / 2, cy + diameter / 2)
        start = -90.0
        for index, value in enumerate(values):
            sweep = 360.0 * value / total
            draw.pieslice(box, start, start + sweep, fill=(*_PALETTE[index % len(_PALETTE)], 255))
            start += sweep

    # -- text --------------------------------------------------------------

    def _label(self, image, text, font, direction, x, y, width, rng) -> tuple[Line, ...]:
        rendered = self.text_renderer.render(
            text,
            font,
            TextStyle(align=Alignment.CENTER),
            max_width=width,
            direction=direction,
            rng=rng,
        )
        if not rendered.lines:
            return ()
        return rendered.place(image, x, y)

    def _category_labels(self, image, chart, font, plot, direction, rng) -> list[Line]:
        """Axis labels, drawn as real text so they supervise recognition too."""
        if chart.kind is ChartKind.PIE or not chart.categories:
            return []
        lines: list[Line] = []
        groups = len(chart.categories)
        slot = plot.width / groups
        for index, category in enumerate(chart.categories):
            if slot < 18:
                break  # no room to write anything legible
            rendered = self.text_renderer.render(
                category,
                font,
                TextStyle(align=Alignment.CENTER),
                max_width=slot,
                direction=direction,
                rng=rng,
            )
            if rendered.lines:
                lines.extend(rendered.place(image, plot.x0 + index * slot, plot.y1 + 4))
        return lines


def sample_chart(
    rng: random.Random,
    labels: list[str],
    *,
    title: str = "",
    kind: ChartKind | None = None,
) -> Chart:
    """Build a plausible chart from a pool of category labels."""
    kind = kind or rng.choice(list(ChartKind))
    count = rng.randint(3, 6) if kind is not ChartKind.PIE else rng.randint(3, 5)
    categories = tuple((labels * count)[:count])
    series_count = 1 if kind is ChartKind.PIE else rng.randint(1, 3)
    series = tuple(
        ChartSeries(
            name=labels[index % len(labels)],
            values=tuple(round(rng.uniform(5, 100), 1) for _ in range(count)),
        )
        for index in range(series_count)
    )
    return Chart(kind=kind, title=title, categories=categories, series=series)
