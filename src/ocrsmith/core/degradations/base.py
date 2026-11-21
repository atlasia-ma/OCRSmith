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

import numpy as np
from PIL import Image

from ...domain.annotations import Line, Page, Region, Word

__all__ = [
    "ArrayDegradation",
    "Degradation",
    "FieldDegradation",
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


class ArrayDegradation(PhotometricDegradation):
    """A photometric degradation whose real work is arithmetic on the pixel array.

    Subclasses implement `shade`, which receives a writable `(H, W, 3)` float32 array and
    returns one — the same array, modified in place, for anything that can be done that
    way. `transform` is still available for callers holding a PIL image, and wraps `shade`.

    The point is that the pipeline can run a whole run of these against one buffer. See
    `buffer.Surface`: the per-stage decode/clip/encode round trip these avoid was two
    thirds of the cost of the `photo` preset.

    Values handed to `shade` are **not** guaranteed to sit within 0–255, and neither is
    the result. Clipping is the pipeline's job, done once where the pixels become bytes,
    so that a shadow followed by a glare does not lose everything the shadow pushed below
    zero.
    """

    @abstractmethod
    def shade(self, array: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, Any]]: ...

    def transform(self, image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
        from .buffer import Surface

        surface = Surface.of(image)
        array, params = self.shade(surface.as_array(), rng)
        return Surface.from_array(array).as_image(), params


class FieldDegradation(ArrayDegradation):
    """A degradation that is a smooth multiply-and-add over the whole image.

    Shadows, vignetting, glare, stains, creases and uneven lighting are all the same
    operation — `pixel * gain + bias`, where `gain` and `bias` vary slowly across the
    page. Run as separate stages, each one walks every pixel of a multi-megapixel array
    to apply a field that is, by construction, almost constant over any given hundred
    pixels. Six of them in a row is six full passes over the page to express something
    that a single pass could carry.

    Subclasses implement `field`, which builds `(gain, bias)` at whatever resolution is
    offered rather than at the page's. `DegradationPipeline` then composes every
    consecutive field stage into one `(gain, bias)` pair, upsamples that once, and
    applies it once. The composition is exact: applying `(g1, b1)` and then `(g2, b2)`
    gives `x*g1*g2 + b1*g2 + b2`, which is the fold in `fuse_fields`.

    Building the fields small is the other half. Every one of these is smooth at a scale
    far larger than a glyph — that is what makes them a *field* rather than a texture —
    so they are built at a fraction of the page's resolution and upsampled. Anything
    whose detail matters at the pixel, such as paper grain or sensor noise, is an
    `ArrayDegradation` instead and keeps its own pass.
    """

    #: The field is built at `1 / FIELD_DIVISOR` of the page's resolution. 4 keeps the
    #: sharpest thing any of these draws — a crease, whose Gaussian sigma is at least
    #: 0.4% of the page — several field pixels wide, so the upsample cannot alias it.
    FIELD_DIVISOR = 4

    @abstractmethod
    def field(
        self, width: int, height: int, rng: random.Random
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """`(gain, bias, params)` for an image of `width` x `height` *field* pixels.

        Either may be None, meaning "identity": a pure shadow has no bias and a pure
        glare has no gain. Each must broadcast against `(height, width, 1)`.
        """

    def shade(self, array: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply this stage alone, for a caller using it outside a pipeline."""
        height, width = array.shape[:2]
        small_h, small_w = field_size(width, height, self.FIELD_DIVISOR)
        gain, bias, params = self.field(small_w, small_h, rng)
        return apply_field(array, gain, bias, (width, height)), params


def field_size(width: int, height: int, divisor: int) -> tuple[int, int]:
    """`(height, width)` of the reduced grid a field is built on."""
    return (max(2, height // divisor), max(2, width // divisor))


def _upsample(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a single-channel field to `(width, height)`, preserving its range.

    PIL rather than a numpy interpolation because the resample is the expensive half and
    PIL's is in C. The field is carried through an 8-bit image, so it is first mapped
    onto a known interval and mapped back: the quantisation that costs is 1/255 of the
    field's own range, which for a shading field is far below what survives being
    written out as an 8-bit page.
    """
    width, height = size
    if plane.shape[0] == height and plane.shape[1] == width:
        return plane
    low = float(plane.min())
    high = float(plane.max())
    if high - low < 1e-9:
        return np.full((height, width), low, dtype=np.float32)
    scaled = (plane - low) * (255.0 / (high - low))
    resized = np.asarray(
        Image.fromarray(scaled.astype(np.uint8), "L").resize((width, height), Image.BILINEAR),
        dtype=np.float32,
    )
    return resized * np.float32((high - low) / 255.0) + np.float32(low)


def _broadcast(plane: np.ndarray | None, small: tuple[int, int]) -> np.ndarray | None:
    """A field given as a row or column vector, grown to the full reduced grid."""
    if plane is None:
        return None
    return np.broadcast_to(plane, small).astype(np.float32, copy=False)


def fuse_fields(
    stages: list[tuple[FieldDegradation, np.ndarray | None, np.ndarray | None]],
    small: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compose several `(gain, bias)` pairs into the one pair that has the same effect."""
    gain: np.ndarray | None = None
    bias: np.ndarray | None = None
    for _stage, stage_gain, stage_bias in stages:
        stage_gain = _broadcast(stage_gain, small)
        stage_bias = _broadcast(stage_bias, small)
        if stage_gain is not None:
            gain = stage_gain if gain is None else gain * stage_gain
            if bias is not None:
                bias = bias * stage_gain
        if stage_bias is not None:
            bias = stage_bias if bias is None else bias + stage_bias
    return gain, bias


def apply_field(
    array: np.ndarray,
    gain: np.ndarray | None,
    bias: np.ndarray | None,
    size: tuple[int, int],
) -> np.ndarray:
    """Apply `array * gain + bias` in place, upsampling the fields to the image first."""
    if gain is not None:
        array *= _upsample(gain, size)[..., None]
    if bias is not None:
        array += _upsample(bias, size)[..., None]
    return array


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
        """Run the pipeline, returning the result and a record of what was applied.

        The page is carried as a `Surface` rather than as a PIL image, so that a run of
        arithmetic stages shares one float32 buffer instead of decoding and re-encoding
        the whole page between each pair of them.
        """
        from .buffer import Surface

        rng = rng or random.Random()
        order = list(self.degradations)
        if self.shuffle:
            rng.shuffle(order)

        surface = Surface.of(image)
        records: list[DegradationRecord] = []
        # Consecutive smooth-field stages are collected and applied as one. See
        # `FieldDegradation`: six separate passes over a multi-megapixel page, each to
        # apply something that barely changes from pixel to pixel, is most of what a
        # heavily degraded page used to cost.
        pending: list[tuple[FieldDegradation, np.ndarray | None, np.ndarray | None]] = []
        small: tuple[int, int] = (0, 0)

        def flush() -> None:
            nonlocal surface, pending
            if not pending:
                return
            array = surface.as_array()
            gain, bias = fuse_fields(pending, small)
            surface = Surface.from_array(apply_field(array, gain, bias, surface.size))
            pending = []

        for degradation in order:
            if rng.random() > degradation.probability:
                continue
            if isinstance(degradation, FieldDegradation):
                width, height = surface.size
                small = field_size(width, height, degradation.FIELD_DIVISOR)
                gain, bias, params = degradation.field(small[1], small[0], rng)
                pending.append((degradation, gain, bias))
                records.append(DegradationRecord(degradation.name, params))
                continue
            flush()
            if isinstance(degradation, ArrayDegradation):
                array, params = degradation.shade(surface.as_array(), rng)
                surface = Surface.from_array(array)
            elif (
                isinstance(degradation, PhotometricDegradation)
                # A photometric degradation that overrides `apply` wants the annotation
                # for something — the morphological filters read the stroke width off it
                # to decide whether their kernel would erase the text. Calling `transform`
                # on those would silently skip that guard.
                and type(degradation).apply is PhotometricDegradation.apply
            ):
                result, params = degradation.transform(surface.as_image(), rng)
                surface = Surface.of(result)
            else:
                result, page, params = degradation.apply(surface.as_image(), page, rng)
                surface = Surface.of(result)
            records.append(DegradationRecord(degradation.name, params))
        flush()
        return surface.as_image(), page, tuple(records)
