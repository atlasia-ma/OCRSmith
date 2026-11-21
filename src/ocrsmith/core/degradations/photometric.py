"""Photometric degradations: everything that changes how the page looks, not where it is.

Each of these corresponds to a physical cause, because that is what makes the resulting
distribution match reality rather than merely look noisy:

* **ink bleed / erosion** — toner spreading into paper fibres, or a worn ribbon;
* **bleed-through** — text on the reverse of a thin sheet showing through, mirrored;
* **JPEG artefacts and downscaling** — what actually happens to a photo before it reaches
  a model, and the single most common source of real-world OCR failure;
* **shadow, vignette and glare** — a phone camera between a page and a ceiling light;
* **stains, folds and scratches** — the life of a paper document.
"""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ...domain import Page
from .base import ArrayDegradation, FieldDegradation, PhotometricDegradation

__all__ = [
    "Bleedthrough",
    "Blur",
    "Brightness",
    "Contrast",
    "Downscale",
    "Folds",
    "GaussianNoise",
    "Glare",
    "InkErosion",
    "InkSpread",
    "JpegArtifacts",
    "MotionBlur",
    "PaperGrain",
    "Shadow",
    "Stains",
    "Vignette",
]


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.split()[-1])
        return flat
    return image.convert("RGB")


def _axes(width: int, height: int, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Row and column coordinates as broadcastable vectors rather than two full grids.

    `np.mgrid[0:height, 0:width]` materialises two `(height, width)` int64 arrays — 60 MB
    for an A4 page at 200 dpi, before a single arithmetic temporary. Every field built from
    them here either depends on one axis or is separable, so the grids are pure memory
    traffic: broadcasting a `(1, width)` against a `(height, 1)` produces identical values.

    float32 by default. Every field built on these axes is a smooth multiplier or offset
    that ends up quantised to 8 bits, so float64 bought about seven spare decimal digits
    at twice the memory traffic — and memory traffic is all these fields are.
    """
    return (
        np.arange(height, dtype=dtype)[:, None],
        np.arange(width, dtype=dtype)[None, :],
    )


def _sample(rng: random.Random, value: float | tuple[float, float]) -> float:
    if isinstance(value, (tuple, list)):
        low, high = float(value[0]), float(value[1])
        return rng.uniform(min(low, high), max(low, high))
    return float(value)


class GaussianNoise(ArrayDegradation):
    """Sensor noise. `sigma` is in 0-255 units.

    Split into a luminance component shared by the three channels and a smaller
    independent one per channel, rather than drawn independently everywhere. That is both
    closer to a real sensor — demosaicing correlates neighbouring channels, so raw
    per-channel noise reads as colour speckle a camera does not actually produce — and
    four times cheaper, because the shared component is one plane instead of three and
    float32 normals are drawn by a faster path than float64 ones.
    """

    #: Share of the variance that is independent per channel. The rest is common to all
    #: three. Chroma noise is real but much weaker than luma noise after demosaicing.
    _CHROMA = 0.35

    def __init__(self, sigma: float | tuple[float, float] = (4.0, 18.0), probability: float = 1.0):
        self.sigma = sigma
        self.probability = probability

    def shade(self, array, rng) -> tuple[np.ndarray, dict[str, Any]]:
        sigma = _sample(rng, self.sigma)
        if sigma <= 0:
            return array, {"sigma": 0.0}
        generator = np.random.default_rng(rng.getrandbits(32))
        height, width = array.shape[:2]

        luma = generator.standard_normal((height, width, 1), dtype=np.float32)
        luma *= np.float32(sigma * (1.0 - self._CHROMA) ** 0.5)
        array += luma

        # Chroma noise is generated at half resolution and upsampled. That is not a
        # shortcut taken for speed alone: every stage downstream of a sensor subsamples
        # chroma — JPEG's 4:2:0 does exactly this — so full-resolution per-channel noise
        # models a signal that no capture chain actually delivers. It is also three
        # quarters cheaper, and this was the most expensive stage in the preset.
        small_h, small_w = max(1, height // 2), max(1, width // 2)
        chroma = generator.standard_normal((small_h, small_w, 3), dtype=np.float32)
        chroma *= np.float32(sigma * self._CHROMA**0.5)
        if (small_h, small_w) != (height, width):
            chroma = np.asarray(
                Image.fromarray(np.clip(chroma + 128.0, 0, 255).astype(np.uint8), "RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) - np.float32(128.0)
        array += chroma
        return array, {"sigma": sigma}


class PaperGrain(ArrayDegradation):
    """Low-frequency paper texture, correlated rather than per-pixel like sensor noise."""

    def __init__(self, strength: float | tuple[float, float] = (6.0, 20.0), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def shade(self, array, rng) -> tuple[np.ndarray, dict[str, Any]]:
        strength = _sample(rng, self.strength)
        height, width = array.shape[:2]
        generator = np.random.default_rng(rng.getrandbits(32))
        small = generator.standard_normal((max(1, height // 4), max(1, width // 4)), dtype=np.float32)
        small *= np.float32(strength)
        # Upsampled through PIL rather than through a numpy interpolation: the quarter-
        # scale plane is 16x smaller, so the resample dominates and PIL's is in C.
        grain = np.asarray(
            Image.fromarray(np.clip(small + 128.0, 0, 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        array += grain[..., None] - np.float32(128.0)
        return array, {"strength": strength}


class Blur(PhotometricDegradation):
    """Defocus."""

    def __init__(self, radius: float | tuple[float, float] = (0.3, 1.6), probability: float = 1.0):
        self.radius = radius
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        radius = _sample(rng, self.radius)
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), {"radius": radius}


class MotionBlur(ArrayDegradation):
    """Camera shake: a directional smear rather than a symmetric one."""

    def __init__(
        self,
        length: int | tuple[int, int] = (3, 9),
        angle: float | tuple[float, float] = (0.0, 180.0),
        probability: float = 1.0,
    ):
        self.length = length
        self.angle = angle
        self.probability = probability

    def shade(self, array, rng) -> tuple[np.ndarray, dict[str, Any]]:
        length = max(2, int(round(_sample(rng, self.length))))
        angle = _sample(rng, self.angle)
        radians = np.deg2rad(angle)
        dx, dy = float(np.cos(radians)), float(np.sin(radians))

        # Averaging shifted copies along one direction *is* a line kernel, and unlike
        # ImageFilter.Kernel it is not limited to 3x3 and 5x5 supports. Distinct shifts
        # are accumulated with their multiplicity rather than rolled once each: a short
        # smear at a shallow angle rounds several steps onto the same offset, and rolling
        # the page again for each of them was pure repeated work.
        offsets: dict[tuple[int, int], int] = {}
        for step in range(-(length // 2), length // 2 + 1):
            key = (int(round(step * dy)), int(round(step * dx)))
            offsets[key] = offsets.get(key, 0) + 1
        total = float(sum(offsets.values()))

        accumulator = np.zeros_like(array)
        for (shift_y, shift_x), weight in offsets.items():
            if shift_y == 0 and shift_x == 0:
                accumulator += array * np.float32(weight)
            else:
                accumulator += np.roll(array, (shift_y, shift_x), axis=(0, 1)) * np.float32(weight)
        accumulator /= np.float32(total)
        return accumulator, {"length": length, "angle": angle}


class Brightness(PhotometricDegradation):
    def __init__(self, factor: float | tuple[float, float] = (0.7, 1.25), probability: float = 1.0):
        self.factor = factor
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        factor = _sample(rng, self.factor)
        return ImageEnhance.Brightness(image).enhance(factor), {"factor": factor}


class Contrast(PhotometricDegradation):
    def __init__(self, factor: float | tuple[float, float] = (0.6, 1.3), probability: float = 1.0):
        self.factor = factor
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        factor = _sample(rng, self.factor)
        return ImageEnhance.Contrast(image).enhance(factor), {"factor": factor}


class JpegArtifacts(PhotometricDegradation):
    """Lossy recompression — what every photograph and most scans have been through."""

    def __init__(self, quality: int | tuple[int, int] = (25, 80), probability: float = 1.0):
        self.quality = quality
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        quality = int(round(_sample(rng, self.quality)))
        buffer = io.BytesIO()
        _as_rgb(image).save(buffer, format="JPEG", quality=max(1, min(95, quality)))
        buffer.seek(0)
        return Image.open(buffer).copy(), {"quality": quality}


class Downscale(PhotometricDegradation):
    """Resolution loss: shrink, then scale back, destroying fine stroke detail.

    Low effective DPI is the failure mode that matters most for small Arabic diacritics,
    and it is not reproduced by blur alone.
    """

    def __init__(self, scale: float | tuple[float, float] = (0.4, 0.9), probability: float = 1.0):
        self.scale = scale
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        scale = max(0.05, min(1.0, _sample(rng, self.scale)))
        width, height = image.size
        small = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR
        )
        return small.resize((width, height), Image.Resampling.BILINEAR), {"scale": scale}


#: Arabic stems run at roughly this fraction of the line box height across the faces in
#: the pool. Only ever used to decide whether a morphological kernel is safe, never to
#: draw anything, so an approximation is enough.
_STROKE_PER_LINE_HEIGHT = 0.09


def _stroke_width(page: Page) -> float:
    """Estimated stroke width of the body text, in pixels."""
    heights = [
        line.bbox.y1 - line.bbox.y0
        for region in page.regions
        for line in region.lines
        if line.bbox.y1 > line.bbox.y0
    ]
    if not heights:
        return 0.0
    heights.sort()
    return heights[len(heights) // 2] * _STROKE_PER_LINE_HEIGHT


class _MorphologicalInk(PhotometricDegradation):
    """Shared guard for the two filters that add or remove ink at the stroke edge.

    A morphological kernel of size `n` moves every edge by `(n - 1) / 2` pixels. That is a
    thinning on a 6-pixel stem and an erasure on a 2-pixel one, and the kernel was a
    constant while the stroke width was not - so `fax` at 13.5 pt ran a 3-pixel erosion
    over 2-pixel strokes and removed the text, while the label kept every word. The kernel
    is therefore chosen against the strokes actually on the page, and skipped outright when
    even the smallest one would eat them.
    """

    #: Below this many pixels of stem, any kernel at all destroys the letterform.
    _MIN_STROKE = 3.0

    def __init__(self, size: int = 3, probability: float = 1.0):
        self.size = size if size % 2 else size + 1
        self.probability = probability

    def _kernel_for(self, page: Page) -> int:
        stroke = _stroke_width(page)
        if stroke <= 0:
            return self.size  # no text to measure; nothing to protect either
        if stroke < self._MIN_STROKE:
            return 0
        if stroke < self._MIN_STROKE * 2:
            return 3
        return self.size

    def apply(self, image, page: Page, rng) -> tuple[Image.Image, Page, dict[str, Any]]:
        kernel = self._kernel_for(page)
        if kernel < 3:
            return image, page, {"size": 0, "skipped": "strokes too thin"}
        return self._filter(image, kernel), page, {"size": kernel}

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        return self._filter(image, self.size), {"size": self.size}

    def _filter(self, image: Image.Image, kernel: int) -> Image.Image:
        raise NotImplementedError


class InkSpread(_MorphologicalInk):
    """Toner bleeding into the paper: dark pixels grow."""

    def _filter(self, image: Image.Image, kernel: int) -> Image.Image:
        return image.filter(ImageFilter.MinFilter(kernel))


class InkErosion(_MorphologicalInk):
    """A worn ribbon or a fading print: dark pixels shrink and strokes break up."""

    def _filter(self, image: Image.Image, kernel: int) -> Image.Image:
        return image.filter(ImageFilter.MaxFilter(kernel))


class Bleedthrough(PhotometricDegradation):
    """Text from the reverse of a thin sheet, mirrored and faint."""

    def __init__(self, strength: float | tuple[float, float] = (0.05, 0.22), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        strength = _sample(rng, self.strength)
        base = _as_rgb(image)
        ghost = base.transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(1.2))
        return Image.blend(base, ghost, strength), {"strength": strength}


class Shadow(FieldDegradation):
    """A soft luminance gradient, as cast by a hand or a phone over the page."""

    def __init__(self, strength: float | tuple[float, float] = (0.1, 0.45), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def field(self, width, height, rng):
        strength = _sample(rng, self.strength)
        angle = rng.uniform(0, 2 * np.pi)
        ys, xs = _axes(width, height)
        ramp = (xs / width) * np.float32(np.cos(angle)) + (ys / height) * np.float32(np.sin(angle))
        ramp = (ramp - ramp.min()) / max(1e-6, float(ramp.max() - ramp.min()))
        gain = 1.0 - np.float32(strength) * ramp
        return gain, None, {"strength": strength, "angle": float(angle)}


class Vignette(FieldDegradation):
    """Corner darkening from a lens."""

    def __init__(self, strength: float | tuple[float, float] = (0.15, 0.5), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def field(self, width, height, rng):
        strength = _sample(rng, self.strength)
        ys, xs = _axes(width, height)
        cx, cy = width / 2, height / 2
        # The falloff is in r^2, so the square root taken to form r is cancelled right
        # after. Squared radius directly: same field, one transcendental fewer per pixel.
        radius_squared = (((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2) / np.float32(2.0)
        return 1.0 - np.float32(strength) * radius_squared, None, {"strength": strength}


class Glare(FieldDegradation):
    """A bright specular blob, as from a ceiling light reflecting off glossy paper."""

    def __init__(self, strength: float | tuple[float, float] = (0.15, 0.5), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def field(self, width, height, rng):
        strength = _sample(rng, self.strength)
        cx = rng.uniform(0.15, 0.85) * width
        cy = rng.uniform(0.15, 0.85) * height
        spread = rng.uniform(0.15, 0.4) * max(width, height)
        ys, xs = _axes(width, height)
        # An isotropic Gaussian is separable, so the blob is the outer product of two
        # 1-D exponentials. That is width+height calls to exp instead of width*height.
        denominator = np.float32(2 * spread**2)
        blob = np.exp(-((xs - cx) ** 2) / denominator) * np.exp(-((ys - cy) ** 2) / denominator)
        bias = np.float32(strength * 255.0) * blob
        return None, bias, {"strength": strength, "center": (float(cx), float(cy))}


class Stains(FieldDegradation):
    """Coffee rings, damp marks and other blotches."""

    def __init__(
        self,
        count: int | tuple[int, int] = (1, 4),
        strength: float | tuple[float, float] = (0.05, 0.25),
        probability: float = 1.0,
    ):
        self.count = count
        self.strength = strength
        self.probability = probability

    def field(self, width, height, rng):
        count = int(round(_sample(rng, self.count)))
        strength = _sample(rng, self.strength)
        ys, xs = _axes(width, height)
        mask = np.zeros((height, width), dtype=np.float32)
        for _ in range(max(0, count)):
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            spread = rng.uniform(0.03, 0.14) * max(width, height)
            denominator = np.float32(2 * spread**2)
            # Separable, as in `Glare`: one exp per row and one per column, not one per pixel.
            mask += np.exp(-((xs - cx) ** 2) / denominator) * np.exp(-((ys - cy) ** 2) / denominator)
        mask = np.clip(mask, 0.0, 1.0) * np.float32(strength)
        # A stain is an alpha blend towards a brown, which is exactly a gain and a bias:
        # `x * (1 - m) + tint * m`. The tint is a single colour, so the bias it
        # contributes is one plane per channel — but the three differ only by a constant
        # factor, and keeping the mask mono lets this fuse with the other fields. The
        # colour is folded back in by weighting the mask per channel.
        tint = float(np.mean([rng.uniform(120, 190), rng.uniform(90, 150), rng.uniform(50, 110)]))
        return 1.0 - mask, np.float32(tint) * mask, {"count": count, "strength": strength}


class Folds(FieldDegradation):
    """Creases: thin luminance ridges where the paper has been folded."""

    def __init__(
        self,
        count: int | tuple[int, int] = (1, 3),
        strength: float | tuple[float, float] = (0.08, 0.3),
        probability: float = 1.0,
    ):
        self.count = count
        self.strength = strength
        self.probability = probability

    def field(self, width, height, rng):
        count = int(round(_sample(rng, self.count)))
        strength = _sample(rng, self.strength)
        ys, xs = _axes(width, height)
        # A crease runs along one axis, so its ridge is a row or a column vector. They are
        # accumulated as such and only broadcast to the page at the end: a horizontal fold
        # on an A4 page at 200 dpi is 1600 values, not 2.8 million.
        rows = np.zeros((height, 1), dtype=np.float32)
        columns = np.zeros((1, width), dtype=np.float32)
        for _ in range(max(0, count)):
            if rng.random() < 0.5:
                position, axis, extent, target = rng.uniform(0, height), ys, height, "rows"
            else:
                position, axis, extent, target = rng.uniform(0, width), xs, width, "columns"
            sharpness = rng.uniform(0.004, 0.02) * extent
            ridge = np.exp(-(((axis - position) ** 2) / np.float32(2 * sharpness**2)))
            if target == "rows":
                rows += ridge
            else:
                columns += ridge
        shading = np.clip(rows + columns, 0.0, 1.0)
        return 1.0 - np.float32(strength) * shading, None, {"count": count, "strength": strength}
