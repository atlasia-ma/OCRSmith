"""The annotation model: what a generated page actually contains.

The hierarchy is `Page -> Region -> Line -> Word`, plus `Table` with its own cell grid.
It is deliberately richer than "one image, one string", because a single rendered page can
supervise very different objectives:

* recognition — line and word crops with their text;
* detection — word/line boxes and polygons;
* layout analysis — typed regions in reading order;
* structure / markup — the page serialised back to HTML or Markdown, which is what
  document-to-markup models are trained against.

Every object is a frozen dataclass and knows how to serialise itself, so exporters stay
thin and no annotation can be mutated halfway down a pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from html import escape

from ..text.script import Direction
from .geometry import BBox, Polygon

__all__ = [
    "Line",
    "Page",
    "Region",
    "RegionType",
    "Table",
    "TableCell",
    "Word",
    "line_from_dict",
    "page_from_dict",
    "region_from_dict",
    "table_from_dict",
    "word_from_dict",
]


class RegionType(str, Enum):
    """Layout categories, aligned with common document-layout benchmarks."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FORMULA = "formula"
    CODE = "code"
    KEY_VALUE = "key_value"
    QUOTE = "quote"
    SEPARATOR = "separator"
    STAMP = "stamp"
    SIGNATURE = "signature"
    HANDWRITING = "handwriting"
    MARGINALIA = "marginalia"


@dataclass(frozen=True, slots=True)
class Word:
    """The smallest addressable unit: one whitespace-delimited token and where it sits.

    `text` is always in logical order, so a right-to-left word reads the same in the
    label as it does to a human, regardless of where its pixels landed.
    """

    text: str
    bbox: BBox
    polygon: Polygon | None = None

    def translate(self, dx: float, dy: float) -> Word:
        return Word(
            self.text,
            self.bbox.translate(dx, dy),
            self.polygon.translate(dx, dy) if self.polygon else None,
        )

    def to_dict(self) -> dict:
        data: dict = {"text": self.text, "bbox": list(self.bbox.as_tuple())}
        if self.polygon:
            data["polygon"] = list(self.polygon.as_flat())
        return data


@dataclass(frozen=True, slots=True)
class Line:
    """One rendered line of text."""

    text: str
    bbox: BBox
    words: tuple[Word, ...] = ()
    direction: Direction = Direction.LTR
    baseline: float | None = None
    polygon: Polygon | None = None

    def translate(self, dx: float, dy: float) -> Line:
        return Line(
            self.text,
            self.bbox.translate(dx, dy),
            tuple(word.translate(dx, dy) for word in self.words),
            self.direction,
            None if self.baseline is None else self.baseline + dy,
            self.polygon.translate(dx, dy) if self.polygon else None,
        )

    def to_dict(self) -> dict:
        data: dict = {
            "text": self.text,
            "bbox": list(self.bbox.as_tuple()),
            "direction": self.direction.value,
        }
        if self.words:
            data["words"] = [word.to_dict() for word in self.words]
        if self.baseline is not None:
            data["baseline"] = self.baseline
        if self.polygon:
            data["polygon"] = list(self.polygon.as_flat())
        return data


@dataclass(frozen=True, slots=True)
class TableCell:
    """A single cell, addressed by its top-left grid position plus spans."""

    row: int
    col: int
    text: str = ""
    bbox: BBox | None = None
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    lines: tuple[Line, ...] = ()

    def translate(self, dx: float, dy: float) -> TableCell:
        return TableCell(
            self.row,
            self.col,
            self.text,
            self.bbox.translate(dx, dy) if self.bbox else None,
            self.row_span,
            self.col_span,
            self.is_header,
            tuple(line.translate(dx, dy) for line in self.lines),
        )

    def to_dict(self) -> dict:
        data: dict = {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "is_header": self.is_header,
        }
        if self.bbox:
            data["bbox"] = list(self.bbox.as_tuple())
        return data


@dataclass(frozen=True, slots=True)
class Table:
    """A cell grid with the two serialisations table models are trained on.

    HTML is the usual target for TEDS-scored models; OTSL is the compact tag sequence
    used by table-structure transformers, where the grid is described cell by cell in
    row-major order.
    """

    rows: int
    cols: int
    cells: tuple[TableCell, ...] = ()
    has_header_row: bool = False

    def cell_at(self, row: int, col: int) -> TableCell | None:
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None

    def translate(self, dx: float, dy: float) -> Table:
        return Table(
            self.rows,
            self.cols,
            tuple(cell.translate(dx, dy) for cell in self.cells),
            self.has_header_row,
        )

    def to_html(self) -> str:
        out: list[str] = ["<table>"]
        for row in range(self.rows):
            out.append("<tr>")
            for col in range(self.cols):
                cell = self.cell_at(row, col)
                if cell is None:
                    continue  # covered by a span from an earlier cell
                tag = "th" if cell.is_header else "td"
                attrs = ""
                if cell.row_span > 1:
                    attrs += f' rowspan="{cell.row_span}"'
                if cell.col_span > 1:
                    attrs += f' colspan="{cell.col_span}"'
                out.append(f"<{tag}{attrs}>{escape(cell.text)}</{tag}>")
            out.append("</tr>")
        out.append("</table>")
        return "".join(out)

    def to_otsl(self) -> str:
        """Optimised Table Structure Language tag sequence.

        ``fcel`` is a filled cell, ``ecel`` an empty one, ``lcel``/``ucel`` mark
        continuations of a horizontal/vertical span, ``xcel`` a cross-span
        continuation, and ``nl`` ends a row.
        """
        occupied: dict[tuple[int, int], str] = {}
        for cell in self.cells:
            for dr in range(cell.row_span):
                for dc in range(cell.col_span):
                    if dr == 0 and dc == 0:
                        occupied[(cell.row, cell.col)] = "fcel" if cell.text else "ecel"
                    elif dr == 0:
                        occupied[(cell.row + dr, cell.col + dc)] = "lcel"
                    elif dc == 0:
                        occupied[(cell.row + dr, cell.col + dc)] = "ucel"
                    else:
                        occupied[(cell.row + dr, cell.col + dc)] = "xcel"

        tokens: list[str] = []
        for row in range(self.rows):
            for col in range(self.cols):
                tokens.append(occupied.get((row, col), "ecel"))
            tokens.append("nl")
        return " ".join(tokens)

    def to_markdown(self) -> str:
        lines: list[str] = []
        for row in range(self.rows):
            values = [(self.cell_at(row, col) or TableCell(row, col)).text for col in range(self.cols)]
            lines.append("| " + " | ".join(values) + " |")
            if row == 0 and self.has_header_row:
                lines.append("| " + " | ".join(["---"] * self.cols) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "has_header_row": self.has_header_row,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class Region:
    """A typed layout block: a paragraph, a heading, a table, a figure."""

    type: RegionType
    bbox: BBox
    lines: tuple[Line, ...] = ()
    table: Table | None = None
    reading_order: int = 0
    attributes: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.table is not None and not self.lines:
            return "\n".join(
                " ".join((self.table.cell_at(r, c) or TableCell(r, c)).text for c in range(self.table.cols))
                for r in range(self.table.rows)
            )
        return "\n".join(line.text for line in self.lines)

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(word for line in self.lines for word in line.words)

    def translate(self, dx: float, dy: float) -> Region:
        return Region(
            self.type,
            self.bbox.translate(dx, dy),
            tuple(line.translate(dx, dy) for line in self.lines),
            self.table.translate(dx, dy) if self.table else None,
            self.reading_order,
            dict(self.attributes),
        )

    def to_dict(self) -> dict:
        data: dict = {
            "type": self.type.value,
            "bbox": list(self.bbox.as_tuple()),
            "reading_order": self.reading_order,
        }
        if self.lines:
            data["lines"] = [line.to_dict() for line in self.lines]
        if self.table is not None:
            data["table"] = self.table.to_dict()
        if self.attributes:
            data["attributes"] = dict(self.attributes)
        return data


_BLOCK_HTML_TAGS: dict[RegionType, str] = {
    RegionType.TITLE: "h1",
    RegionType.HEADING: "h2",
    RegionType.PARAGRAPH: "p",
    RegionType.CAPTION: "figcaption",
    RegionType.HEADER: "header",
    RegionType.FOOTER: "footer",
    RegionType.QUOTE: "blockquote",
    RegionType.CODE: "pre",
    RegionType.PAGE_NUMBER: "footer",
    RegionType.MARGINALIA: "aside",
}

_BLOCK_MARKDOWN_PREFIX: dict[RegionType, str] = {
    RegionType.TITLE: "# ",
    RegionType.HEADING: "## ",
    RegionType.CAPTION: "*",
    RegionType.QUOTE: "> ",
}


@dataclass(frozen=True, slots=True)
class Page:
    """One rendered page and everything known about it."""

    width: int
    height: int
    regions: tuple[Region, ...] = ()
    direction: Direction = Direction.LTR
    attributes: dict = field(default_factory=dict)

    # -- views -------------------------------------------------------------

    @property
    def bbox(self) -> BBox:
        return BBox(0, 0, self.width, self.height)

    def ordered_regions(self) -> tuple[Region, ...]:
        return tuple(sorted(self.regions, key=lambda region: region.reading_order))

    def iter_lines(self) -> Iterator[Line]:
        for region in self.ordered_regions():
            yield from region.lines

    def iter_words(self) -> Iterator[Word]:
        for line in self.iter_lines():
            yield from line.words

    @property
    def text(self) -> str:
        """Plain-text transcription in reading order."""
        return "\n".join(region.text for region in self.ordered_regions() if region.text)

    def regions_of(self, *types: RegionType) -> tuple[Region, ...]:
        wanted = set(types)
        return tuple(r for r in self.ordered_regions() if r.type in wanted)

    # -- markup ------------------------------------------------------------

    def to_html(self) -> str:
        """Structural HTML ground truth for document-to-markup training."""
        parts: list[str] = []
        for region in self.ordered_regions():
            if region.type is RegionType.TABLE and region.table is not None:
                parts.append(region.table.to_html())
            elif region.type is RegionType.LIST:
                items = "".join(f"<li>{escape(line.text)}</li>" for line in region.lines)
                parts.append(f"<ul>{items}</ul>")
            elif region.type is RegionType.FIGURE:
                alt = escape(str(region.attributes.get("alt", "")))
                parts.append(f'<figure><img alt="{alt}"></figure>')
            elif region.type is RegionType.FORMULA:
                parts.append(f"<math>{escape(region.text)}</math>")
            elif region.type is RegionType.SEPARATOR:
                parts.append("<hr>")
            else:
                tag = _BLOCK_HTML_TAGS.get(region.type, "p")
                parts.append(f"<{tag}>{escape(region.text)}</{tag}>")
        return "\n".join(parts)

    def to_markdown(self) -> str:
        """Markdown ground truth, the target format of most modern OCR-to-text models."""
        parts: list[str] = []
        for region in self.ordered_regions():
            if region.type is RegionType.TABLE and region.table is not None:
                parts.append(region.table.to_markdown())
            elif region.type is RegionType.LIST:
                parts.append("\n".join(f"- {line.text}" for line in region.lines))
            elif region.type is RegionType.SEPARATOR:
                parts.append("---")
            elif region.type is RegionType.FORMULA:
                parts.append(f"$$\n{region.text}\n$$")
            elif region.type is RegionType.CODE:
                parts.append(f"```\n{region.text}\n```")
            elif region.type is RegionType.CAPTION:
                parts.append(f"*{region.text}*")
            else:
                prefix = _BLOCK_MARKDOWN_PREFIX.get(region.type, "")
                parts.append(f"{prefix}{region.text}")
        return "\n\n".join(part for part in parts if part.strip())

    # -- transforms --------------------------------------------------------

    def with_regions(self, regions: Iterable[Region]) -> Page:
        return Page(self.width, self.height, tuple(regions), self.direction, dict(self.attributes))

    def clipped(self) -> Page:
        """Trim every box to the page rectangle.

        Rotation and warping can push a box a pixel or two outside the canvas; detection
        formats reject out-of-bounds coordinates, so the page is the final authority.
        """
        page_box = self.bbox

        def clip_line(line: Line) -> Line:
            return Line(
                line.text,
                line.bbox.clip(0, 0, self.width, self.height),
                tuple(
                    Word(w.text, w.bbox.clip(0, 0, self.width, self.height), w.polygon) for w in line.words
                ),
                line.direction,
                line.baseline,
                line.polygon,
            )

        regions = tuple(
            Region(
                region.type,
                region.bbox.clip(0, 0, self.width, self.height),
                tuple(clip_line(line) for line in region.lines),
                region.table,
                region.reading_order,
                dict(region.attributes),
            )
            for region in self.regions
            if region.bbox.intersects(page_box)
        )
        return self.with_regions(regions)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        data: dict = {
            "width": self.width,
            "height": self.height,
            "direction": self.direction.value,
            "regions": [region.to_dict() for region in self.ordered_regions()],
        }
        if self.attributes:
            data["attributes"] = dict(self.attributes)
        return data


def assign_reading_order(regions: Sequence[Region], direction: Direction) -> tuple[Region, ...]:
    """Order regions top-to-bottom, then by the direction the script reads in.

    Regions whose vertical extents overlap are treated as being on the same visual row,
    which keeps a two-column header from being read as two separate rows.
    """

    def sort_key(region: Region) -> tuple[float, float]:
        horizontal = -region.bbox.x1 if direction.is_rtl else region.bbox.x0
        return (round(region.bbox.y0, 1), horizontal)

    ordered = sorted(regions, key=sort_key)
    return tuple(
        Region(
            region.type,
            region.bbox,
            region.lines,
            region.table,
            index,
            dict(region.attributes),
        )
        for index, region in enumerate(ordered)
    )


# -- deserialisation ------------------------------------------------------
#
# Annotations are read back far more often than they are written: validation, statistics,
# evaluation and any downstream loader all start from a serialised record. Keeping the
# decoder next to the encoder is what stops the two drifting apart.


def word_from_dict(data: dict) -> Word:
    polygon = data.get("polygon")
    return Word(
        text=data.get("text", ""),
        bbox=BBox.from_tuple(data["bbox"]),
        polygon=Polygon.from_flat(polygon) if polygon else None,
    )


def line_from_dict(data: dict) -> Line:
    polygon = data.get("polygon")
    return Line(
        text=data.get("text", ""),
        bbox=BBox.from_tuple(data["bbox"]),
        words=tuple(word_from_dict(word) for word in data.get("words", ())),
        direction=Direction(data.get("direction", "ltr")),
        baseline=data.get("baseline"),
        polygon=Polygon.from_flat(polygon) if polygon else None,
    )


def table_from_dict(data: dict) -> Table:
    return Table(
        rows=data["rows"],
        cols=data["cols"],
        cells=tuple(
            TableCell(
                row=cell["row"],
                col=cell["col"],
                text=cell.get("text", ""),
                bbox=BBox.from_tuple(cell["bbox"]) if cell.get("bbox") else None,
                row_span=cell.get("row_span", 1),
                col_span=cell.get("col_span", 1),
                is_header=cell.get("is_header", False),
            )
            for cell in data.get("cells", ())
        ),
        has_header_row=data.get("has_header_row", False),
    )


def region_from_dict(data: dict) -> Region:
    table = data.get("table")
    return Region(
        type=RegionType(data.get("type", "paragraph")),
        bbox=BBox.from_tuple(data["bbox"]),
        lines=tuple(line_from_dict(line) for line in data.get("lines", ())),
        table=table_from_dict(table) if table else None,
        reading_order=data.get("reading_order", 0),
        attributes=dict(data.get("attributes", {})),
    )


def page_from_dict(data: dict) -> Page:
    """Rebuild a `Page` from the dict produced by `Page.to_dict`."""
    return Page(
        width=data["width"],
        height=data["height"],
        regions=tuple(region_from_dict(region) for region in data.get("regions", ())),
        direction=Direction(data.get("direction", "ltr")),
        attributes=dict(data.get("attributes", {})),
    )
