"""Geometry primitives shared by every annotation.

Boxes are stored in absolute pixel coordinates with the origin at the top-left, matching
Pillow. Normalisation to the unit square is available for exporters that want it, but the
canonical form stays in pixels so that a crop or a resize is a single explicit transform.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["BBox", "Point", "Polygon"]

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class BBox:
    """An axis-aligned rectangle, normalised so that ``x0 <= x1`` and ``y0 <= y1``."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # Callers pass corners in whatever order they happen to have them; ordering here
        # means no downstream code has to defend against inverted boxes.
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        if x0 > x1:
            object.__setattr__(self, "x0", x1)
            object.__setattr__(self, "x1", x0)
        if y0 > y1:
            object.__setattr__(self, "y0", y1)
            object.__setattr__(self, "y1", y0)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> BBox:
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_tuple(cls, values: Sequence[float]) -> BBox:
        x0, y0, x1, y1 = values
        return cls(x0, y0, x1, y1)

    @classmethod
    def from_points(cls, points: Iterable[Point]) -> BBox:
        points = list(points)
        if not points:
            raise ValueError("A bounding box needs at least one point")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def union_all(cls, boxes: Iterable[BBox]) -> BBox:
        boxes = list(boxes)
        if not boxes:
            raise ValueError("A union needs at least one box")
        return cls(
            min(b.x0 for b in boxes),
            min(b.y0 for b in boxes),
            max(b.x1 for b in boxes),
            max(b.y1 for b in boxes),
        )

    # -- measurements ------------------------------------------------------

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    # -- transforms --------------------------------------------------------

    def translate(self, dx: float, dy: float) -> BBox:
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def scale(self, sx: float, sy: float | None = None) -> BBox:
        sy = sx if sy is None else sy
        return BBox(self.x0 * sx, self.y0 * sy, self.x1 * sx, self.y1 * sy)

    def pad(self, amount: float) -> BBox:
        return BBox(self.x0 - amount, self.y0 - amount, self.x1 + amount, self.y1 + amount)

    def clip(self, x0: float, y0: float, x1: float, y1: float) -> BBox:
        """Trim the box to the given bounds, e.g. the page rectangle."""
        return BBox(
            min(max(self.x0, x0), x1),
            min(max(self.y0, y0), y1),
            min(max(self.x1, x0), x1),
            min(max(self.y1, y0), y1),
        )

    def normalized(self, page_width: float, page_height: float) -> BBox:
        """Express the box in `[0, 1]` coordinates relative to a page."""
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Page dimensions must be positive to normalise a box")
        return BBox(
            self.x0 / page_width,
            self.y0 / page_height,
            self.x1 / page_width,
            self.y1 / page_height,
        )

    # -- relations ---------------------------------------------------------

    def union(self, other: BBox) -> BBox:
        return BBox.union_all((self, other))

    def intersection(self, other: BBox) -> BBox | None:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        return BBox(x0, y0, x1, y1)

    def intersects(self, other: BBox) -> bool:
        return self.intersection(other) is not None

    def iou(self, other: BBox) -> float:
        """Intersection over union; 0.0 when the boxes are disjoint or degenerate."""
        overlap = self.intersection(other)
        if overlap is None:
            return 0.0
        denominator = self.area + other.area - overlap.area
        return 0.0 if denominator <= 0 else overlap.area / denominator

    def contains(self, other: BBox) -> bool:
        return self.x0 <= other.x0 and self.y0 <= other.y0 and self.x1 >= other.x1 and self.y1 >= other.y1

    # -- serialisation -----------------------------------------------------

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def as_xywh(self) -> tuple[float, float, float, float]:
        """COCO-style ``[x, y, width, height]``."""
        return (self.x0, self.y0, self.width, self.height)

    def as_int(self) -> tuple[int, int, int, int]:
        """Pixel-aligned box that fully encloses this one."""
        return (
            math.floor(self.x0),
            math.floor(self.y0),
            math.ceil(self.x1),
            math.ceil(self.y1),
        )


@dataclass(frozen=True, slots=True)
class Polygon:
    """An ordered ring of points, used for rotated or warped text regions.

    Degradations such as perspective warp move a box's corners independently, at which
    point an axis-aligned rectangle can no longer describe the text tightly.
    """

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        points = tuple((float(x), float(y)) for x, y in self.points)
        if len(points) < 3:
            raise ValueError("A polygon needs at least three points")
        object.__setattr__(self, "points", points)

    @classmethod
    def from_bbox(cls, box: BBox) -> Polygon:
        """Clockwise quad starting at the top-left corner."""
        return cls(
            (
                (box.x0, box.y0),
                (box.x1, box.y0),
                (box.x1, box.y1),
                (box.x0, box.y1),
            )
        )

    @classmethod
    def from_flat(cls, values: Sequence[float]) -> Polygon:
        if len(values) % 2:
            raise ValueError("A flat polygon needs an even number of coordinates")
        return cls(tuple((values[i], values[i + 1]) for i in range(0, len(values), 2)))

    @property
    def bbox(self) -> BBox:
        return BBox.from_points(self.points)

    def translate(self, dx: float, dy: float) -> Polygon:
        return Polygon(tuple((x + dx, y + dy) for x, y in self.points))

    def scale(self, sx: float, sy: float | None = None) -> Polygon:
        sy = sx if sy is None else sy
        return Polygon(tuple((x * sx, y * sy) for x, y in self.points))

    def as_flat(self) -> tuple[float, ...]:
        """``[x0, y0, x1, y1, ...]``, the layout used by COCO and PaddleOCR."""
        return tuple(value for point in self.points for value in point)
