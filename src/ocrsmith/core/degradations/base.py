"""Degradation protocol and pipeline.

A degradation is anything that stands between clean rendered text and what a camera or a
scanner actually captures. The critical design point is that a degradation takes the
*annotation* as well as the image and returns both: a rotation that moves ink 8 degrees
but leaves the boxes where they were silently destroys the ground truth, and that failure
is invisible until a detector trained on the data underperforms for no obvious reason.

Photometric degradations leave geometry alone and inherit `PhotometricDegradation`;
anything that moves pixels must map the annotation through the same transform.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ...domain.annotations import Line, Page, Region, Word

__all__ = [
    "Degradation",
    "DegradationPipeline",
    "DegradationRecord",
    "PhotometricDegradation",
]


@dataclass(frozen=True, slots=True)
class DegradationRecord:
    """What a pipeline actually did, for provenance and for debugging a failure mode."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, **self.params}


class Degradation(ABC):
    """Transforms an image and, when it moves pixels, its annotation with it."""

    #: Probability the pipeline applies this degradation to any given sample.
    probability: float = 1.0

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def apply(
        self, image: Image.Image, page: Page, rng: random.Random
    ) -> tuple[Image.Image, Page, dict[str, Any]]:
        """Return the degraded image, the updated annotation, and the sampled parameters."""


class PhotometricDegradation(Degradation):
    """A degradation that changes colour or texture but not position.

    Subclasses implement `transform`; the annotation is passed through untouched, which is
    correct precisely because no pixel moves.
    """

    @abstractmethod
    def transform(self, image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]: ...

    def apply(
        self, image: Image.Image, page: Page, rng: random.Random
    ) -> tuple[Image.Image, Page, dict[str, Any]]:
        result, params = self.transform(image, rng)
        return result, page, params


def map_page(page: Page, mapper, width: int, height: int) -> Page:
    """Rebuild a page with every coordinate passed through `mapper`.

    Boxes are re-derived from the mapped corners of the original box, so a rotated word is
    described by the axis-aligned box that encloses its rotated quad — the convention every
    detection format expects.
    """
    from ...domain.geometry import BBox, Polygon

    def map_box(box: BBox) -> tuple[BBox, Polygon]:
        corners = [mapper(x, y) for x, y in Polygon.from_bbox(box).points]
        return BBox.from_points(corners), Polygon(tuple(corners))

    def map_word(word: Word) -> Word:
        box, quad = map_box(word.bbox)
        return Word(word.text, box, quad)

    def map_line(line: Line) -> Line:
        box, quad = map_box(line.bbox)
        return Line(
            line.text,
            box,
            tuple(map_word(word) for word in line.words),
            line.direction,
            None,
            quad,
        )

    def map_region(region: Region) -> Region:
        box, _ = map_box(region.bbox)
        table = region.table
        if table is not None:
            from ...domain.annotations import Table, TableCell

            table = Table(
                table.rows,
                table.cols,
                tuple(
                    TableCell(
                        cell.row,
                        cell.col,
                        cell.text,
                        map_box(cell.bbox)[0] if cell.bbox else None,
                        cell.row_span,
                        cell.col_span,
                        cell.is_header,
                        tuple(map_line(line) for line in cell.lines),
                    )
                    for cell in table.cells
                ),
                table.has_header_row,
            )
        return Region(
            region.type,
            box,
            tuple(map_line(line) for line in region.lines),
            table,
            region.reading_order,
            dict(region.attributes),
        )

    return Page(
        width,
        height,
        tuple(map_region(region) for region in page.regions),
        page.direction,
        dict(page.attributes),
    ).clipped()


class DegradationPipeline:
    """Applies a sequence of degradations, each with its own probability.

    Order matters and is deliberately *not* shuffled by default: a scan is geometrically
    distorted by the glass and then compressed by the software, never the other way round.
    """

    def __init__(self, degradations: list[Degradation] | None = None, *, shuffle: bool = False):
        self.degradations = list(degradations or [])
        self.shuffle = shuffle

    def add(self, degradation: Degradation, probability: float | None = None) -> DegradationPipeline:
        if probability is not None:
            degradation.probability = probability
        self.degradations.append(degradation)
        return self

    def __len__(self) -> int:
        return len(self.degradations)

    def apply(
        self, image: Image.Image, page: Page, rng: random.Random | None = None
    ) -> tuple[Image.Image, Page, tuple[DegradationRecord, ...]]:
        """Run the pipeline, returning the result and a record of what was applied."""
        rng = rng or random.Random()
        order = list(self.degradations)
        if self.shuffle:
            rng.shuffle(order)

        records: list[DegradationRecord] = []
        for degradation in order:
            if rng.random() > degradation.probability:
                continue
            image, page, params = degradation.apply(image, page, rng)
            records.append(DegradationRecord(degradation.name, params))
        return image, page, tuple(records)
