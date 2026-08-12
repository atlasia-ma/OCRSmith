"""Page geometry: paper size, margins and columns.

Real documents are described in millimetres at a scanning resolution, not in arbitrary
pixel counts. Deriving the canvas from (paper, DPI) keeps font sizes, margins and stroke
widths in a believable relationship to each other at every resolution, and makes "render
this dataset at 200 DPI instead of 96" a one-line change rather than a re-tuning exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.geometry import BBox
from ...text.script import Direction

__all__ = ["PAPER_SIZES_MM", "PageSpec"]

#: Width and height in millimetres.
PAPER_SIZES_MM: dict[str, tuple[float, float]] = {
    "a3": (297.0, 420.0),
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "a6": (105.0, 148.0),
    "letter": (215.9, 279.4),
    "legal": (215.9, 355.6),
    "receipt": (80.0, 200.0),
    "id_card": (85.6, 54.0),
}

_MM_PER_INCH = 25.4


@dataclass(frozen=True, slots=True)
class PageSpec:
    """The canvas a document is laid out on."""

    width: int
    height: int
    margin_top: int = 60
    margin_right: int = 60
    margin_bottom: int = 60
    margin_left: int = 60
    columns: int = 1
    column_gap: int = 30
    direction: Direction = Direction.LTR
    #: Vertical band reserved at the top of the content area for a running header.
    header_height: int = 0
    #: Vertical band reserved at the bottom for a running footer or page number.
    footer_height: int = 0
    dpi: int = 150

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Page dimensions must be positive")
        if self.columns < 1:
            raise ValueError("A page needs at least one column")
        if self.content_width <= 0 or self.content_height <= 0:
            raise ValueError("Margins leave no room for content")

    @classmethod
    def from_paper(
        cls,
        paper: str = "a4",
        dpi: int = 150,
        *,
        margin_mm: float = 20.0,
        landscape: bool = False,
        **kwargs,
    ) -> PageSpec:
        """Build a spec from a named paper size at a scanning resolution."""
        try:
            width_mm, height_mm = PAPER_SIZES_MM[paper.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown paper size {paper!r}. Available: {', '.join(sorted(PAPER_SIZES_MM))}"
            ) from None
        if landscape:
            width_mm, height_mm = height_mm, width_mm

        def px(mm: float) -> int:
            return max(1, int(round(mm / _MM_PER_INCH * dpi)))

        margin = px(margin_mm)
        kwargs.setdefault("margin_top", margin)
        kwargs.setdefault("margin_right", margin)
        kwargs.setdefault("margin_bottom", margin)
        kwargs.setdefault("margin_left", margin)
        return cls(width=px(width_mm), height=px(height_mm), dpi=dpi, **kwargs)

    # -- derived geometry --------------------------------------------------

    @property
    def content_width(self) -> int:
        return self.width - self.margin_left - self.margin_right

    @property
    def content_height(self) -> int:
        return self.height - self.margin_top - self.margin_bottom

    @property
    def content_box(self) -> BBox:
        """The area available to body content, excluding header and footer bands."""
        return BBox(
            self.margin_left,
            self.margin_top + self.header_height,
            self.width - self.margin_right,
            self.height - self.margin_bottom - self.footer_height,
        )

    @property
    def header_box(self) -> BBox:
        return BBox(
            self.margin_left,
            self.margin_top,
            self.width - self.margin_right,
            self.margin_top + self.header_height,
        )

    @property
    def footer_box(self) -> BBox:
        return BBox(
            self.margin_left,
            self.height - self.margin_bottom - self.footer_height,
            self.width - self.margin_right,
            self.height - self.margin_bottom,
        )

    @property
    def column_width(self) -> float:
        total_gap = self.column_gap * (self.columns - 1)
        return (self.content_box.width - total_gap) / self.columns

    def column_boxes(self) -> tuple[BBox, ...]:
        """Columns in *reading* order, so right-to-left pages start on the right."""
        box = self.content_box
        width = self.column_width
        boxes = [
            BBox(
                box.x0 + index * (width + self.column_gap),
                box.y0,
                box.x0 + index * (width + self.column_gap) + width,
                box.y1,
            )
            for index in range(self.columns)
        ]
        return tuple(reversed(boxes)) if self.direction.is_rtl else tuple(boxes)

    def scaled(self, factor: float) -> PageSpec:
        """A proportionally resized copy, e.g. to render the same layout at another DPI."""

        def s(value: int) -> int:
            return max(1, int(round(value * factor)))

        return PageSpec(
            width=s(self.width),
            height=s(self.height),
            margin_top=s(self.margin_top),
            margin_right=s(self.margin_right),
            margin_bottom=s(self.margin_bottom),
            margin_left=s(self.margin_left),
            columns=self.columns,
            column_gap=s(self.column_gap),
            direction=self.direction,
            header_height=s(self.header_height) if self.header_height else 0,
            footer_height=s(self.footer_height) if self.footer_height else 0,
            dpi=int(round(self.dpi * factor)),
        )
