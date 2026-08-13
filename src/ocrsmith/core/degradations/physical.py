"""Physical degradations: paper that is not flat, and light that is not even.

Rendering-based synthesis produces a page lying perfectly flat under perfectly uniform
light. Real captures never do, and the gap is measurable — modelling deformation and
illumination is worth several points on documents photographed in the wild.

Three effects, each with a physical cause:

* **wrinkles and creases** — the sheet is displaced locally, and the ridges catch light;
* **page curl** — a bound or rolled page bends away from the sensor, compressing text
  towards one edge;
* **illumination fields** — a lamp or window lights one part of the page more than another,
  smoothly, at a scale much larger than the text.

Deformations move pixels, so they map the annotation through the same displacement. That
is done by sampling the displacement field at each box corner rather than by inverting it
analytically: the field is smooth, so corner sampling tracks the ink closely, and it works
for any field without needing a closed form.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image

from ...domain.annotations import Page
from .base import Degradation, PhotometricDegradation, map_page
from .photometric import _as_rgb, _sample

__all__ = ["IlluminationField", "PageCurl", "Wrinkles"]


def _smooth_noise(shape: tuple[int, int], scale: int, generator: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in [-1, 1], generated small and upsampled.

    Paper deformation and lighting are both smooth at a scale far larger than a glyph, so
    generating them per-pixel would be both wrong and slow.
    """
    height, width = shape
    small = generator.normal(0.0, 1.0, (max(2, height // scale), max(2, width // scale))).astype(np.float32)
    small = np.clip(small, -3.0, 3.0) / 3.0
    image = Image.fromarray(((small + 1.0) * 127.5).astype(np.uint8))
    resized = np.asarray(image.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
    return resized / 127.5 - 1.0


def _remap(image: Image.Image, dx: np.ndarray, dy: np.ndarray, fill) -> Image.Image:
    """Sample `image` at (x + dx, y + dy) with nearest-neighbour lookup.

    Nearest-neighbour rather than bilinear: the displacement is sub-pixel-smooth and the
    text is already antialiased, so interpolating twice visibly softens strokes — which
    would be indistinguishable from a blur degradation and confound any ablation.
    """
    source = np.asarray(_as_rgb(image))
    height, width = source.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    sample_x = np.clip(np.rint(xs + dx).astype(np.int32), 0, width - 1)
    sample_y = np.clip(np.rint(ys + dy).astype(np.int32), 0, height - 1)

    inside = (xs + dx >= 0) & (xs + dx <= width - 1) & (ys + dy >= 0) & (ys + dy <= height - 1)
    out = source[sample_y, sample_x]
    out = np.where(inside[..., None], out, np.array(fill, dtype=source.dtype))
    return Image.fromarray(out.astype(np.uint8), "RGB")


class _DisplacementDegradation(Degradation):
    """Shared machinery for anything that displaces pixels by a smooth field.

    Subclasses build the field; this maps both the pixels and the annotation through it.
    """

    fill: tuple[int, int, int] = (255, 255, 255)

    def _field(
        self, width: int, height: int, rng: random.Random
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def apply(self, image: Image.Image, page: Page, rng: random.Random):
        width, height = image.size
        dx, dy = self._field(width, height, rng)[:2]
        params = self._field_params

        warped = _remap(image, dx, dy, self.fill)

        def mapper(x: float, y: float) -> tuple[float, float]:
            # The field says where a *destination* pixel reads from, so a point in the
            # original lands at minus the displacement sampled there.
            column = min(max(int(round(x)), 0), width - 1)
            row = min(max(int(round(y)), 0), height - 1)
            return (x - float(dx[row, column]), y - float(dy[row, column]))

        return warped, map_page(page, mapper, width, height), params


class Wrinkles(_DisplacementDegradation):
    """Creases and crumpling: local displacement plus the shading its ridges catch."""

    def __init__(
        self,
        strength: float | tuple[float, float] = (2.0, 7.0),
        scale: int = 28,
        shading: float | tuple[float, float] = (0.05, 0.22),
        probability: float = 1.0,
    ):
        self.strength = strength
        self.scale = scale
        self.shading = shading
        self.probability = probability
        self._field_params: dict[str, Any] = {}
        self._shade: np.ndarray | None = None

    def _field(self, width: int, height: int, rng: random.Random):
        strength = _sample(rng, self.strength)
        shading = _sample(rng, self.shading)
        generator = np.random.default_rng(rng.getrandbits(32))

        dx = _smooth_noise((height, width), self.scale, generator) * strength
        dy = _smooth_noise((height, width), self.scale, generator) * strength
        # Ridges are where the sheet bends, i.e. where the displacement changes fastest.
        gradient = np.abs(np.gradient(dx)[1]) + np.abs(np.gradient(dy)[0])
        gradient /= max(1e-6, float(gradient.max()))
        self._shade = 1.0 - shading * gradient
        self._field_params = {"strength": strength, "shading": shading, "scale": self.scale}
        return dx, dy, self._field_params

    def apply(self, image: Image.Image, page: Page, rng: random.Random):
        warped, mapped_page, params = super().apply(image, page, rng)
        if self._shade is not None:
            array = np.asarray(warped, dtype=np.float32) * self._shade[..., None]
            warped = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
        return warped, mapped_page, params


class PageCurl(_DisplacementDegradation):
    """A page bending away from the sensor, as when photographing a bound book.

    Text compresses towards the curled edge and the shading darkens into the gutter —
    the single most recognisable artefact of a photographed book page, and one that flat
    perspective warping cannot produce.
    """

    def __init__(
        self,
        strength: float | tuple[float, float] = (0.03, 0.12),
        edge: str = "random",
        probability: float = 1.0,
    ):
        self.strength = strength
        self.edge = edge
        self.probability = probability
        self._field_params: dict[str, Any] = {}
        self._shade: np.ndarray | None = None

    def _field(self, width: int, height: int, rng: random.Random):
        strength = _sample(rng, self.strength)
        edge = self.edge if self.edge != "random" else rng.choice(["left", "right", "top", "bottom"])

        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        if edge in ("left", "right"):
            axis = xs / max(1, width - 1)
            if edge == "right":
                axis = 1.0 - axis
            profile = np.power(1.0 - axis, 2.5)
            dx = profile * strength * width
            dy = np.zeros_like(dx)
            if edge == "right":
                dx = -dx
        else:
            axis = ys / max(1, height - 1)
            if edge == "bottom":
                axis = 1.0 - axis
            profile = np.power(1.0 - axis, 2.5)
            dy = profile * strength * height
            dx = np.zeros_like(dy)
            if edge == "bottom":
                dy = -dy

        self._shade = 1.0 - 0.45 * strength / 0.12 * profile
        self._field_params = {"strength": strength, "edge": edge}
        return dx, dy, self._field_params

    def apply(self, image: Image.Image, page: Page, rng: random.Random):
        warped, mapped_page, params = super().apply(image, page, rng)
        if self._shade is not None:
            array = np.asarray(warped, dtype=np.float32) * self._shade[..., None]
            warped = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
        return warped, mapped_page, params


class IlluminationField(PhotometricDegradation):
    """Uneven lighting: a smooth multiplicative field across the page.

    Distinct from `Shadow`, which is a linear ramp, and from `Vignette`, which is radial
    and centred. Real lighting is neither — it is a lamp somewhere off to one side, a
    window, a hand, and the resulting field is smooth but arbitrary.
    """

    def __init__(
        self,
        strength: float | tuple[float, float] = (0.12, 0.4),
        scale: int = 6,
        probability: float = 1.0,
    ):
        self.strength = strength
        self.scale = scale
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        strength = _sample(rng, self.strength)
        image = _as_rgb(image)
        width, height = image.size
        generator = np.random.default_rng(rng.getrandbits(32))

        field = _smooth_noise((height, width), max(2, min(width, height) // self.scale), generator)
        multiplier = 1.0 + strength * field
        array = np.asarray(image, dtype=np.float32) * multiplier[..., None]
        return (
            Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"),
            {"strength": strength, "scale": self.scale},
        )
