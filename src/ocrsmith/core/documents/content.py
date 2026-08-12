"""The content model: what a document says, before anything decides how it looks.

Templates produce `DocumentContent`; the layout engine turns it into pixels. Keeping the
two apart means the same content can be laid out as a one-column article, a two-column
newspaper page or a narrow mobile scan, and the markup ground truth is identical in all
three — which is exactly the invariance a document-understanding model should learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.annotations import RegionType, Table, TableCell
from ...text.script import Direction, detect_direction

__all__ = ["ContentBlock", "DocumentBuilder", "DocumentContent"]


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """One logical block of content, not yet placed anywhere."""

    type: RegionType
    text: str = ""
    #: Entries of a list block, in order.
    items: tuple[str, ...] = ()
    #: Cell grid of a table block; boxes are filled in during layout.
    table: Table | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def is_splittable(self) -> bool:
        """Whether the block may be broken across columns or pages.

        Prose flows; a heading detached from its section, or half a table, is a layout
        bug rather than a realistic document.
        """
        return self.type in (RegionType.PARAGRAPH, RegionType.QUOTE)

    def with_text(self, text: str) -> ContentBlock:
        return ContentBlock(self.type, text, self.items, self.table, dict(self.attributes))


@dataclass(frozen=True, slots=True)
class DocumentContent:
    """A complete document's content and the direction it reads in."""

    blocks: tuple[ContentBlock, ...] = ()
    direction: Direction = Direction.LTR
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text)

    def __len__(self) -> int:
        return len(self.blocks)


class DocumentBuilder:
    """Fluent builder for `DocumentContent`.

    Templates read far better as a sequence of intentions (`.title(...).paragraph(...)`)
    than as dictionary literals, and the builder is the single place that knows how each
    intention maps onto a region type.
    """

    def __init__(self, direction: Direction | None = None, **metadata):
        self._blocks: list[ContentBlock] = []
        self._direction = direction
        self._metadata = dict(metadata)

    # -- text blocks -------------------------------------------------------

    def title(self, text: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.TITLE, text, attributes)

    def heading(self, text: str, level: int = 2, **attributes) -> DocumentBuilder:
        return self._add(RegionType.HEADING, text, {"level": level, **attributes})

    def paragraph(self, text: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.PARAGRAPH, text, attributes)

    def quote(self, text: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.QUOTE, text, attributes)

    def caption(self, text: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.CAPTION, text, attributes)

    def code(self, text: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.CODE, text, attributes)

    def formula(self, latex: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.FORMULA, latex, attributes)

    def header(self, text: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.HEADER, text, attributes)

    def footer(self, text: str, **attributes) -> DocumentBuilder:
        return self._add(RegionType.FOOTER, text, attributes)

    def separator(self) -> DocumentBuilder:
        return self._add(RegionType.SEPARATOR, "", {})

    # -- structured blocks -------------------------------------------------

    def list(self, items, ordered: bool = False, **attributes) -> DocumentBuilder:
        items = tuple(str(item) for item in items)
        self._blocks.append(
            ContentBlock(
                RegionType.LIST,
                text="\n".join(items),
                items=items,
                attributes={"ordered": ordered, **attributes},
            )
        )
        return self

    def table(self, rows, has_header_row: bool = True, **attributes) -> DocumentBuilder:
        """Add a table from a sequence of row sequences."""
        rows = [list(row) for row in rows]
        n_rows = len(rows)
        n_cols = max((len(row) for row in rows), default=0)
        cells = tuple(
            TableCell(
                row=r,
                col=c,
                text=str(rows[r][c]) if c < len(rows[r]) else "",
                is_header=has_header_row and r == 0,
            )
            for r in range(n_rows)
            for c in range(n_cols)
        )
        self._blocks.append(
            ContentBlock(
                RegionType.TABLE,
                table=Table(n_rows, n_cols, cells, has_header_row),
                attributes=dict(attributes),
            )
        )
        return self

    def key_values(self, pairs, **attributes) -> DocumentBuilder:
        """Add form-style ``label: value`` rows, as found on invoices and receipts."""
        pairs = [(str(k), str(v)) for k, v in pairs]
        for key, value in pairs:
            self._blocks.append(
                ContentBlock(
                    RegionType.KEY_VALUE,
                    text=f"{key}: {value}",
                    attributes={"key": key, "value": value, **attributes},
                )
            )
        return self

    def figure(self, width: int, height: int, caption: str | None = None, **attributes) -> DocumentBuilder:
        self._blocks.append(
            ContentBlock(
                RegionType.FIGURE,
                attributes={"width": width, "height": height, "alt": caption or "", **attributes},
            )
        )
        if caption:
            self.caption(caption)
        return self

    # -- assembly ----------------------------------------------------------

    def extend(self, blocks) -> DocumentBuilder:
        self._blocks.extend(blocks)
        return self

    def build(self) -> DocumentContent:
        direction = self._direction or detect_direction(" ".join(block.text for block in self._blocks[:5]))
        return DocumentContent(tuple(self._blocks), direction, dict(self._metadata))

    def _add(self, region_type: RegionType, text: str, attributes: dict) -> DocumentBuilder:
        self._blocks.append(ContentBlock(region_type, text=text, attributes=dict(attributes)))
        return self
