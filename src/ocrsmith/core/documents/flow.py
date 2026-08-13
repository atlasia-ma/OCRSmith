"""Flow layout: content in, pages out.

Blocks are poured down a column, then into the next column, then onto the next page —
the same model a word processor uses. Two rules keep the ground truth honest:

* **nothing is drawn that the annotation does not describe, and nothing is described that
  was not drawn.** A block that does not fit is moved or split, never clipped.
* **prose splits, structure does not.** A paragraph can continue on the next page; a
  heading, a table or a figure moves whole, because half a table is a bug rather than a
  document.

The previous composer laid every section out at a fixed offset and grew the canvas to fit,
which meant overlapping text on long inputs and no notion of a page at all.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from PIL import Image, ImageDraw

from ...domain.annotations import Page, Region, RegionType
from ...domain.geometry import BBox
from ...text.script import Direction
from ..rendering.metrics import metrics_for
from ..rendering.style import Alignment
from ..rendering.text_renderer import TextBlockRenderer
from ..rendering.wrapping import wrap_paragraph
from .charts import ChartRenderer
from .content import ContentBlock, DocumentContent
from .formulas import FormulaRenderer, choose_math_font
from .page_spec import PageSpec
from .table_renderer import RenderedTable, TableRenderer, TableStyle
from .typography import Typography

__all__ = ["DocumentRenderer", "RenderedPage"]

#: Factory producing a page background of the requested size.
BackgroundFactory = Callable[[int, int], Image.Image]


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """One composed page: the image and the annotation that describes it."""

    image: Image.Image
    page: Page
    number: int = 1

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


class DocumentRenderer:
    """Lays `DocumentContent` out on `PageSpec` pages and draws it."""

    def __init__(
        self,
        text_renderer: TextBlockRenderer | None = None,
        table_renderer: TableRenderer | None = None,
        chart_renderer: ChartRenderer | None = None,
    ):
        self.text_renderer = text_renderer or TextBlockRenderer()
        self.table_renderer = table_renderer or TableRenderer(self.text_renderer)
        self.chart_renderer = chart_renderer or ChartRenderer(self.text_renderer)

    def render(
        self,
        content: DocumentContent,
        spec: PageSpec,
        typography: Typography,
        *,
        background: BackgroundFactory | None = None,
        table_style: TableStyle | None = None,
        rng: random.Random | None = None,
        max_pages: int = 8,
    ) -> list[RenderedPage]:
        """Render `content`, returning one `RenderedPage` per page it occupies."""
        return list(
            self.iter_pages(
                content,
                spec,
                typography,
                background=background,
                table_style=table_style,
                rng=rng,
                max_pages=max_pages,
            )
        )

    def iter_pages(
        self,
        content: DocumentContent,
        spec: PageSpec,
        typography: Typography,
        *,
        background: BackgroundFactory | None = None,
        table_style: TableStyle | None = None,
        rng: random.Random | None = None,
        max_pages: int = 8,
    ) -> Iterator[RenderedPage]:
        """Stream pages as they are composed, so long documents never sit in memory whole."""
        rng = rng or random.Random()
        direction = spec.direction if spec.direction.is_rtl else content.direction
        pending: list[ContentBlock] = [b for b in content.blocks if b.type is not RegionType.HEADER]
        header = next((b for b in content.blocks if b.type is RegionType.HEADER), None)
        footer_template = content.metadata.get("footer")

        page_number = 0
        while pending and page_number < max_pages:
            image = self._new_canvas(spec, background)
            regions: list[Region] = []
            furniture = 0

            if header is not None and spec.header_height:
                regions.extend(self._draw_band(image, header, spec.header_box, typography, direction, rng))
            if footer_template and spec.footer_height:
                block = ContentBlock(
                    RegionType.PAGE_NUMBER,
                    text=str(footer_template).replace("{page}", str(page_number + 1)),
                )
                regions.extend(self._draw_band(image, block, spec.footer_box, typography, direction, rng))
            furniture = len(regions)

            pending = self._fill_columns(
                image, pending, spec, typography, direction, regions, table_style, rng
            )

            if len(regions) == furniture:
                # Nothing could be placed anywhere on this page, so the leading block does
                # not fit *any* column of this page shape. Emitting the page would produce
                # a blank image with a confident label, and keeping the block would loop
                # until max_pages; dropping it is the only honest option.
                if pending:
                    pending.pop(0)
                continue

            page_number += 1
            page = Page(spec.width, spec.height, tuple(regions), direction).clipped()
            yield RenderedPage(image, page, page_number)

    # -- page composition --------------------------------------------------

    @staticmethod
    def _new_canvas(spec: PageSpec, background: BackgroundFactory | None) -> Image.Image:
        if background is None:
            return Image.new("RGBA", (spec.width, spec.height), (255, 255, 255, 255))
        canvas = background(spec.width, spec.height)
        if canvas.size != (spec.width, spec.height):
            canvas = canvas.resize((spec.width, spec.height), Image.Resampling.LANCZOS)
        return canvas.convert("RGBA")

    def _fill_columns(
        self, image, blocks, spec, typography, direction, regions, table_style, rng
    ) -> list[ContentBlock]:
        """Pour `blocks` into this page's columns; return whatever did not fit."""
        pending = list(blocks)
        order = len(regions)
        for column in spec.column_boxes():
            cursor = column.y0
            first_in_column = True
            while pending:
                block = pending[0]
                placement = self._place(
                    image,
                    block,
                    column,
                    cursor,
                    spec,
                    typography,
                    direction,
                    table_style,
                    rng,
                    at_column_start=first_in_column,
                )
                if placement is None:
                    break  # does not fit here; try the next column or page
                region, consumed_height, leftover = placement
                if region is not None:
                    regions.append(
                        Region(
                            region.type,
                            region.bbox,
                            region.lines,
                            region.table,
                            order,
                            region.attributes,
                        )
                    )
                    order += 1
                cursor += consumed_height
                first_in_column = False
                pending.pop(0)
                if leftover is not None:
                    pending.insert(0, leftover)
                    break  # the rest of this block continues in the next column
        return pending

    def _draw_band(self, image, block, box: BBox, typography, direction, rng) -> list[Region]:
        """Draw a running header or footer inside its reserved band."""
        role = typography.for_(block.type)
        rendered = self.text_renderer.render(
            block.text,
            role.font,
            role.style.with_(align=Alignment.CENTER),
            max_width=box.width,
            max_height=box.height,
            direction=direction,
            rng=rng,
        )
        if not rendered.lines:
            return []
        lines = rendered.place(image, box.x0, box.y0)
        return [
            Region(
                block.type,
                BBox.union_all(line.bbox for line in lines),
                lines,
                None,
                0,
                dict(block.attributes),
            )
        ]

    # -- block placement ---------------------------------------------------

    def _place(
        self,
        image,
        block: ContentBlock,
        column: BBox,
        cursor: float,
        spec: PageSpec,
        typography: Typography,
        direction: Direction,
        table_style: TableStyle | None,
        rng: random.Random,
        *,
        at_column_start: bool,
    ) -> tuple[Region | None, float, ContentBlock | None] | None:
        """Try to draw `block` at `cursor`.

        Returns `(region, consumed_height, leftover)` or None when the block does not fit
        and should be retried in the next column. `leftover` carries the remainder of a
        split paragraph. `consumed_height` includes the space reserved above and below.
        """
        role = typography.for_(block.type)
        space_before = 0.0 if at_column_start else role.space_before
        top = cursor + space_before
        available = column.y1 - top
        if available <= 0:
            return None

        if block.type is RegionType.SEPARATOR:
            placement = self._place_rule(image, column, top, role, available)
        elif block.type is RegionType.CHART and block.attributes.get("chart") is not None:
            placement = self._place_chart(image, block, column, top, available, role, direction, rng)
        elif block.type is RegionType.FORMULA and block.attributes.get("node") is not None:
            placement = self._place_formula(image, block, column, top, available, role)
        elif block.type is RegionType.FIGURE:
            placement = self._place_figure(image, block, column, top, available, role)
        elif block.type is RegionType.TABLE and block.table is not None:
            placement = self._place_table(
                image, block, column, top, available, role, table_style, direction, rng
            )
        else:
            placement = self._place_text(image, block, column, top, available, role, direction, rng)

        if placement is None:
            return None
        region, height, leftover = placement
        return (region, space_before + height, leftover)

    def _place_text(
        self, image, block, column, top, available, role, direction, rng
    ) -> tuple[Region | None, float, ContentBlock | None] | None:
        text = self._block_text(block)
        if not text.strip():
            return (None, role.space_after, None)

        metrics = metrics_for(role.font, self.text_renderer.shaper)
        line_height = metrics.line_height(role.style.line_spacing)
        capacity = int(available // line_height)
        if capacity < 1:
            return None

        # Each source line is wrapped on its own so list bullets and pre-broken text keep
        # their structure instead of collapsing into one paragraph.
        lines = [
            wrapped
            for source in text.split("\n")
            for wrapped in wrap_paragraph(source, metrics.line_advance, column.width)
        ]
        if not lines:
            return (None, role.space_after, None)

        if len(lines) > capacity:
            # Orphan control: one line of a paragraph stranded at the foot of a column
            # reads as a layout error, so the whole block moves instead of splitting.
            if not block.is_splittable or capacity < 2:
                return None
            head, leftover = lines[:capacity], block.with_text("\n".join(lines[capacity:]))
        else:
            head, leftover = lines, None

        rendered = self.text_renderer.render(
            "\n".join(head),
            role.font,
            role.style,
            max_width=column.width,
            direction=direction,
            rng=rng,
        )
        if not rendered.lines:
            return (None, role.space_after, leftover)

        placed = rendered.place(image, column.x0, top)
        region = Region(
            block.type,
            BBox.union_all(line.bbox for line in placed),
            placed,
            None,
            0,
            dict(block.attributes),
        )
        return (region, rendered.layout_size[1] + role.space_after, leftover)

    def _place_table(
        self, image, block, column, top, available, role, table_style, direction, rng
    ) -> tuple[Region | None, float, ContentBlock | None] | None:
        rendered: RenderedTable = self.table_renderer.render(
            block.table,
            role.font,
            max_width=column.width,
            style=table_style or TableStyle(),
            text_style=role.style.with_(align=Alignment.NATURAL),
            direction=direction,
            rng=rng,
        )
        if rendered.size[1] > available:
            return None
        x, y = int(round(column.x0)), int(round(top))
        image.alpha_composite(rendered.image, (x, y))
        placed = rendered.translated(x, y)
        box = BBox(x, y, x + rendered.size[0], y + rendered.size[1])
        region = Region(RegionType.TABLE, box, placed.lines, placed.table, 0, dict(block.attributes))
        return (region, rendered.size[1] + role.space_after, None)

    def _place_figure(
        self, image, block, column, top, available, role
    ) -> tuple[Region | None, float, ContentBlock | None] | None:
        width = min(float(block.attributes.get("width", column.width)), column.width)
        height = float(block.attributes.get("height", width * 0.6))
        if height > available:
            return None
        x = column.x0 + (column.width - width) / 2
        box = BBox(x, top, x + width, top + height)
        draw = ImageDraw.Draw(image)
        # A neutral placeholder: layout models are trained on where figures are, not on
        # what they depict, and inventing imagery would be pretending to data we lack.
        draw.rectangle(box.as_int(), outline=(120, 120, 120, 255), fill=(238, 238, 240, 255), width=2)
        region = Region(RegionType.FIGURE, box, (), None, 0, dict(block.attributes))
        return (region, height + role.space_after, None)

    def _place_chart(
        self, image, block, column, top, available, role, direction, rng
    ) -> tuple[Region | None, float, ContentBlock | None] | None:
        """Draw a chart. Its labels are real text, annotated like any other text."""
        width = min(float(block.attributes.get("width", column.width)), column.width)
        height = float(block.attributes.get("height", width * 0.7))
        if height > available:
            return None

        rendered = self.chart_renderer.render(
            block.attributes["chart"],
            role.font,
            width=int(width),
            height=int(height),
            direction=direction,
            rng=rng,
        )
        x = column.x0 + (column.width - width) / 2
        image.alpha_composite(rendered.image, (int(round(x)), int(round(top))))
        placed = rendered.translated(x, top)
        box = BBox(x, top, x + width, top + height)
        region = Region(
            RegionType.CHART,
            box,
            placed.lines,
            None,
            0,
            {**{k: v for k, v in block.attributes.items() if k != "chart"}, "chart": placed.chart.to_dict()},
        )
        return (region, height + role.space_after, None)

    def _place_formula(
        self, image, block, column, top, available, role
    ) -> tuple[Region | None, float, ContentBlock | None] | None:
        """Typeset a formula. The LaTeX travels in the region attributes."""
        renderer = FormulaRenderer(
            choose_math_font([role.font.path], fallback=str(role.font.path)),
            size=max(12, int(getattr(role.font, "size", 24) * 1.15)),
            ink=role.style.color,
        )
        rendered = renderer.render(block.attributes["node"])
        width, height = rendered.size
        if height > available or width > column.width * 1.6:
            return None

        x = column.x0 + max(0.0, (column.width - width) / 2)
        image.alpha_composite(rendered.image, (int(round(x)), int(round(top))))
        box = BBox(x, top, x + width, top + height)
        region = Region(
            RegionType.FORMULA,
            box,
            (),
            None,
            0,
            {**block.attributes, "latex": rendered.latex, "node": None},
        )
        return (region, height + role.space_after, None)

    def _place_rule(
        self, image, column, top, role, available
    ) -> tuple[Region | None, float, ContentBlock | None] | None:
        thickness = 2.0
        if thickness > available:
            return None
        draw = ImageDraw.Draw(image)
        y = top + thickness
        draw.line((column.x0, y, column.x1, y), fill=(120, 120, 120, 255), width=int(thickness))
        box = BBox(column.x0, top, column.x1, top + thickness * 2)
        region = Region(RegionType.SEPARATOR, box, (), None, 0, {})
        return (region, thickness * 2 + role.space_after, None)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _block_text(block: ContentBlock) -> str:
        if block.type is RegionType.LIST and block.items:
            ordered = bool(block.attributes.get("ordered"))
            bullets = [
                f"{index + 1}. {item}" if ordered else f"• {item}" for index, item in enumerate(block.items)
            ]
            return "\n".join(bullets)
        return block.text
