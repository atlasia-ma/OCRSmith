"""Table rendering with cell-level ground truth.

Tables are where document models fail first, so the annotation has to be complete: every
cell carries its own box and its own text lines, and the same `Table` object serialises to
HTML and OTSL. Column widths are derived from content and then shrunk proportionally to
fit the available width, with cell text wrapping rather than overflowing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont

from ...domain.annotations import Line, Table, TableCell
from ...domain.geometry import BBox
from ...text.script import Direction
from ..rendering.metrics import FontMetrics, metrics_for
from ..rendering.style import TextStyle
from ..rendering.text_renderer import TextBlockRenderer
from ..rendering.wrapping import wrap_paragraph

__all__ = ["BorderStyle", "RenderedTable", "TableRenderer", "TableStyle"]


class BorderStyle(str, Enum):
    """Which rules a table draws. Real tables vary a lot, and models overfit to grids."""

    ALL = "all"
    HORIZONTAL = "horizontal"
    OUTER = "outer"
    HEADER_ONLY = "header_only"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class TableStyle:
    border: BorderStyle = BorderStyle.ALL
    border_width: int = 1
    border_color: tuple[int, int, int] = (40, 40, 40)
    cell_padding: int = 8
    header_fill: tuple[int, int, int] | None = None
    zebra_fill: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class RenderedTable:
    """A drawn table plus the annotation describing it."""

    image: Image.Image
    table: Table
    lines: tuple[Line, ...]

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def translated(self, dx: float, dy: float) -> RenderedTable:
        return RenderedTable(
            self.image,
            self.table.translate(dx, dy),
            tuple(line.translate(dx, dy) for line in self.lines),
        )


class TableRenderer:
    """Draws a `Table` and returns it with per-cell boxes filled in."""

    def __init__(self, text_renderer: TextBlockRenderer | None = None):
        self.text_renderer = text_renderer or TextBlockRenderer()

    def render(
        self,
        table: Table,
        font: FreeTypeFont,
        *,
        max_width: float,
        style: TableStyle | None = None,
        text_style: TextStyle | None = None,
        direction: Direction = Direction.LTR,
        rng: random.Random | None = None,
    ) -> RenderedTable:
        style = style or TableStyle()
        text_style = text_style or TextStyle()
        rng = rng or random.Random()
        metrics = metrics_for(font, self.text_renderer.shaper)

        col_widths = self._column_widths(table, metrics, style, max_width)
        wrapped = self._wrap_cells(table, metrics, style, col_widths)
        row_heights = self._row_heights(table, wrapped, metrics, text_style, style)

        width = int(round(sum(col_widths))) + style.border_width
        height = int(round(sum(row_heights))) + style.border_width
        image = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))

        x_edges = self._edges(col_widths, direction, total=width)
        y_edges = self._edges(row_heights, Direction.LTR, total=height)

        self._fill_rows(image, table, style, x_edges, y_edges)
        self._draw_borders(image, table, style, x_edges, y_edges)

        cells, lines = self._draw_cells(
            image, table, wrapped, font, style, text_style, direction, x_edges, y_edges, rng
        )
        return RenderedTable(
            image,
            Table(table.rows, table.cols, tuple(cells), table.has_header_row),
            tuple(lines),
        )

    # -- measurement -------------------------------------------------------

    def _column_widths(
        self, table: Table, metrics: FontMetrics, style: TableStyle, max_width: float
    ) -> list[float]:
        padding = style.cell_padding * 2 + style.border_width
        natural = [padding] * table.cols
        for cell in table.cells:
            if cell.col_span > 1:
                continue  # spanning cells do not constrain a single column
            natural[cell.col] = max(natural[cell.col], metrics.line_advance(cell.text) + padding)

        total = sum(natural)
        if total <= max_width or total <= 0:
            return natural
        scale = max_width / total
        floor = padding
        return [max(floor, width * scale) for width in natural]

    def _wrap_cells(
        self, table: Table, metrics: FontMetrics, style: TableStyle, col_widths: list[float]
    ) -> dict[tuple[int, int], list[str]]:
        inner_padding = style.cell_padding * 2 + style.border_width
        wrapped: dict[tuple[int, int], list[str]] = {}
        for cell in table.cells:
            span_width = sum(col_widths[cell.col : cell.col + cell.col_span])
            available = max(1.0, span_width - inner_padding)
            wrapped[(cell.row, cell.col)] = wrap_paragraph(cell.text, metrics.line_advance, available)
        return wrapped

    def _row_heights(
        self,
        table: Table,
        wrapped: dict[tuple[int, int], list[str]],
        metrics: FontMetrics,
        text_style: TextStyle,
        style: TableStyle,
    ) -> list[float]:
        line_height = metrics.line_height(text_style.line_spacing)
        padding = style.cell_padding * 2 + style.border_width
        heights = [line_height + padding] * table.rows
        for (row, _col), lines in wrapped.items():
            heights[row] = max(heights[row], line_height * max(1, len(lines)) + padding)
        return heights

    @staticmethod
    def _edges(sizes: list[float], direction: Direction, total: float) -> list[float]:
        """Cumulative edges, mirrored horizontally for right-to-left tables."""
        edges = [0.0]
        for size in sizes:
            edges.append(edges[-1] + size)
        if direction.is_rtl:
            return [total - edge for edge in edges]
        return edges

    # -- drawing -----------------------------------------------------------

    @staticmethod
    def _cell_rect(x_edges: list[float], y_edges: list[float], cell: TableCell) -> BBox:
        return BBox(
            x_edges[cell.col],
            y_edges[cell.row],
            x_edges[min(cell.col + cell.col_span, len(x_edges) - 1)],
            y_edges[min(cell.row + cell.row_span, len(y_edges) - 1)],
        )

    def _fill_rows(self, image, table, style, x_edges, y_edges) -> None:
        if not style.header_fill and not style.zebra_fill:
            return
        draw = ImageDraw.Draw(image)
        for row in range(table.rows):
            fill = None
            if row == 0 and table.has_header_row and style.header_fill:
                fill = style.header_fill
            elif style.zebra_fill and row % 2 == 1:
                fill = style.zebra_fill
            if fill is None:
                continue
            box = BBox(min(x_edges), y_edges[row], max(x_edges), y_edges[row + 1])
            draw.rectangle(box.as_int(), fill=(*fill, 255))

    def _draw_borders(self, image, table, style, x_edges, y_edges) -> None:
        if style.border is BorderStyle.NONE:
            return
        draw = ImageDraw.Draw(image)
        colour = (*style.border_color, 255)
        width = style.border_width
        left, right = min(x_edges), max(x_edges)
        top, bottom = y_edges[0], y_edges[-1]

        def horizontal(y: float) -> None:
            draw.line((left, y, right, y), fill=colour, width=width)

        def vertical(x: float) -> None:
            draw.line((x, top, x, bottom), fill=colour, width=width)

        if style.border is BorderStyle.HEADER_ONLY:
            horizontal(y_edges[0])
            horizontal(y_edges[1] if len(y_edges) > 1 else y_edges[-1])
            horizontal(y_edges[-1])
            return

        horizontal(top)
        horizontal(bottom)
        if style.border in (BorderStyle.ALL, BorderStyle.HORIZONTAL):
            for y in y_edges[1:-1]:
                horizontal(y)
        if style.border in (BorderStyle.ALL, BorderStyle.OUTER):
            vertical(left)
            vertical(right)
        if style.border is BorderStyle.ALL:
            for x in x_edges[1:-1]:
                vertical(x)

    def _draw_cells(
        self, image, table, wrapped, font, style, text_style, direction, x_edges, y_edges, rng
    ) -> tuple[list[TableCell], list[Line]]:
        cells: list[TableCell] = []
        lines: list[Line] = []
        for cell in table.cells:
            rect = self._cell_rect(x_edges, y_edges, cell)
            text = "\n".join(wrapped.get((cell.row, cell.col), []))
            if not text.strip():
                cells.append(
                    TableCell(
                        cell.row,
                        cell.col,
                        cell.text,
                        rect,
                        cell.row_span,
                        cell.col_span,
                        cell.is_header,
                    )
                )
                continue

            inner_width = max(1.0, rect.width - 2 * style.cell_padding)
            rendered = self.text_renderer.render(
                text,
                font,
                text_style,
                max_width=inner_width,
                direction=direction,
                rng=rng,
            )
            x = rect.x0 + style.cell_padding
            y = rect.y0 + style.cell_padding
            cell_lines = rendered.place(image, x, y)
            lines.extend(cell_lines)
            cells.append(
                TableCell(
                    cell.row,
                    cell.col,
                    cell.text,
                    rect,
                    cell.row_span,
                    cell.col_span,
                    cell.is_header,
                    tuple(cell_lines),
                )
            )
        return cells, lines
