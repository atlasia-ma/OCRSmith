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

from .base import PhotometricDegradation

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


def _sample(rng: random.Random, value: float | tuple[float, float]) -> float:
    if isinstance(value, (tuple, list)):
        low, high = float(value[0]), float(value[1])
        return rng.uniform(min(low, high), max(low, high))
    return float(value)


class GaussianNoise(PhotometricDegradation):
    """Sensor noise. `sigma` is in 0-255 units."""

    def __init__(self, sigma: float | tuple[float, float] = (4.0, 18.0), probability: float = 1.0):
        self.sigma = sigma
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        sigma = _sample(rng, self.sigma)
        array = np.asarray(_as_rgb(image), dtype=np.float32)
        generator = np.random.default_rng(rng.getrandbits(32))
        noisy = array + generator.normal(0.0, sigma, array.shape).astype(np.float32)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB"), {"sigma": sigma}


class PaperGrain(PhotometricDegradation):
    """Low-frequency paper texture, correlated rather than per-pixel like sensor noise."""

    def __init__(self, strength: float | tuple[float, float] = (6.0, 20.0), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        strength = _sample(rng, self.strength)
        image = _as_rgb(image)
        width, height = image.size
        generator = np.random.default_rng(rng.getrandbits(32))
        small = generator.normal(0.0, strength, (max(1, height // 4), max(1, width // 4), 1))
        grain = (
            np.asarray(
                Image.fromarray(np.clip(small[..., 0] + 128, 0, 255).astype(np.uint8)).resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )[..., None]
            - 128.0
        )
        array = np.asarray(image, dtype=np.float32) + grain
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), {"strength": strength}


class Blur(PhotometricDegradation):
    """Defocus."""

    def __init__(self, radius: float | tuple[float, float] = (0.3, 1.6), probability: float = 1.0):
        self.radius = radius
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        radius = _sample(rng, self.radius)
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), {"radius": radius}


class MotionBlur(PhotometricDegradation):
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

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        length = max(2, int(round(_sample(rng, self.length))))
        angle = _sample(rng, self.angle)
        radians = np.deg2rad(angle)
        dx, dy = float(np.cos(radians)), float(np.sin(radians))

        # Averaging shifted copies along one direction *is* a line kernel, and unlike
        # ImageFilter.Kernel it is not limited to 3x3 and 5x5 supports.
        array = np.asarray(_as_rgb(image), dtype=np.float32)
        offsets = range(-(length // 2), length // 2 + 1)
        accumulator = np.zeros_like(array)
        for step in offsets:
            shifted = np.roll(array, (int(round(step * dy)), int(round(step * dx))), axis=(0, 1))
            accumulator += shifted
        accumulator /= len(list(offsets))
        smeared = Image.fromarray(np.clip(accumulator, 0, 255).astype(np.uint8), "RGB")
        return smeared, {"length": length, "angle": angle}


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


class InkSpread(PhotometricDegradation):
    """Toner bleeding into the paper: dark pixels grow."""

    def __init__(self, size: int = 3, probability: float = 1.0):
        self.size = size if size % 2 else size + 1
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        return image.filter(ImageFilter.MinFilter(self.size)), {"size": self.size}


class InkErosion(PhotometricDegradation):
    """A worn ribbon or a fading print: dark pixels shrink and strokes break up."""

    def __init__(self, size: int = 3, probability: float = 1.0):
        self.size = size if size % 2 else size + 1
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        return image.filter(ImageFilter.MaxFilter(self.size)), {"size": self.size}


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


class Shadow(PhotometricDegradation):
    """A soft luminance gradient, as cast by a hand or a phone over the page."""

    def __init__(self, strength: float | tuple[float, float] = (0.1, 0.45), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        strength = _sample(rng, self.strength)
        image = _as_rgb(image)
        width, height = image.size
        angle = rng.uniform(0, 2 * np.pi)
        ys, xs = np.mgrid[0:height, 0:width]
        ramp = (xs / width) * np.cos(angle) + (ys / height) * np.sin(angle)
        ramp = (ramp - ramp.min()) / max(1e-6, ramp.max() - ramp.min())
        mask = (1.0 - strength * ramp)[..., None]
        array = np.asarray(image, dtype=np.float32) * mask
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), {
            "strength": strength,
            "angle": float(angle),
        }


class Vignette(PhotometricDegradation):
    """Corner darkening from a lens."""

    def __init__(self, strength: float | tuple[float, float] = (0.15, 0.5), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        strength = _sample(rng, self.strength)
        image = _as_rgb(image)
        width, height = image.size
        ys, xs = np.mgrid[0:height, 0:width]
        cx, cy = width / 2, height / 2
        radius = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2) / np.sqrt(2)
        mask = (1.0 - strength * radius**2)[..., None]
        array = np.asarray(image, dtype=np.float32) * mask
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), {"strength": strength}


class Glare(PhotometricDegradation):
    """A bright specular blob, as from a ceiling light reflecting off glossy paper."""

    def __init__(self, strength: float | tuple[float, float] = (0.15, 0.5), probability: float = 1.0):
        self.strength = strength
        self.probability = probability

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        strength = _sample(rng, self.strength)
        image = _as_rgb(image)
        width, height = image.size
        cx = rng.uniform(0.15, 0.85) * width
        cy = rng.uniform(0.15, 0.85) * height
        spread = rng.uniform(0.15, 0.4) * max(width, height)
        ys, xs = np.mgrid[0:height, 0:width]
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * spread**2)))
        array = np.asarray(image, dtype=np.float32) + (strength * 255.0 * blob)[..., None]
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), {
            "strength": strength,
            "center": (float(cx), float(cy)),
        }


class Stains(PhotometricDegradation):
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

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        count = int(round(_sample(rng, self.count)))
        strength = _sample(rng, self.strength)
        image = _as_rgb(image)
        width, height = image.size
        ys, xs = np.mgrid[0:height, 0:width]
        mask = np.zeros((height, width), dtype=np.float32)
        for _ in range(max(0, count)):
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            spread = rng.uniform(0.03, 0.14) * max(width, height)
            mask += np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * spread**2)))
        mask = np.clip(mask, 0.0, 1.0) * strength
        tint = np.array([rng.uniform(120, 190), rng.uniform(90, 150), rng.uniform(50, 110)], dtype=np.float32)
        array = np.asarray(image, dtype=np.float32)
        array = array * (1 - mask[..., None]) + tint * mask[..., None]
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), {
            "count": count,
            "strength": strength,
        }


class Folds(PhotometricDegradation):
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

    def transform(self, image, rng) -> tuple[Image.Image, dict[str, Any]]:
        count = int(round(_sample(rng, self.count)))
        strength = _sample(rng, self.strength)
        image = _as_rgb(image)
        width, height = image.size
        ys, xs = np.mgrid[0:height, 0:width]
        shading = np.zeros((height, width), dtype=np.float32)
        for _ in range(max(0, count)):
            if rng.random() < 0.5:
                position, axis, extent = rng.uniform(0, height), ys, height
            else:
                position, axis, extent = rng.uniform(0, width), xs, width
            sharpness = rng.uniform(0.004, 0.02) * extent
            shading += np.exp(-(((axis - position) ** 2) / (2 * sharpness**2)))
        mask = (1.0 - strength * np.clip(shading, 0.0, 1.0))[..., None]
        array = np.asarray(image, dtype=np.float32) * mask
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB"), {
            "count": count,
            "strength": strength,
        }
