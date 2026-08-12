"""Page backgrounds.

Backgrounds are produced by a factory taking `(width, height)`, so the layout engine never
needs to know whether a page is plain white, tinted stock, a gradient or a photograph of
real paper. Textures are generated at a quarter scale and upsampled: paper grain is
low-frequency, and generating it per-pixel at full page size is pure waste.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = [
    "BackgroundFactory",
    "BackgroundSampler",
    "gradient_background",
    "image_background",
    "paper_background",
    "solid_background",
]

BackgroundFactory = Callable[[int, int], Image.Image]
Color = tuple[int, int, int]

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def solid_background(color: Color) -> BackgroundFactory:
    def factory(width: int, height: int) -> Image.Image:
        return Image.new("RGB", (width, height), color)

    return factory


def paper_background(color: Color, grain: float, rng: random.Random) -> BackgroundFactory:
    """Tinted stock with low-frequency fibre texture."""
    seed = rng.getrandbits(32)

    def factory(width: int, height: int) -> Image.Image:
        generator = np.random.default_rng(seed)
        small_h, small_w = max(1, height // 4), max(1, width // 4)
        noise = generator.normal(0.0, grain, (small_h, small_w)).astype(np.float32)
        upsampled = (
            np.asarray(
                Image.fromarray(np.clip(noise + 128.0, 0, 255).astype(np.uint8)).resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )[..., None]
            - 128.0
        )
        base = np.full((height, width, 3), color, dtype=np.float32) + upsampled
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")

    return factory


def gradient_background(start: Color, end: Color, direction: str) -> BackgroundFactory:
    def factory(width: int, height: int) -> Image.Image:
        ys, xs = np.mgrid[0:height, 0:width]
        if direction == "horizontal":
            ratio = xs / max(1, width - 1)
        elif direction == "vertical":
            ratio = ys / max(1, height - 1)
        else:
            ratio = (xs + ys) / max(1, width + height - 2)
        ratio = ratio[..., None].astype(np.float32)
        array = (
            np.array(start, dtype=np.float32)
            + (np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32)) * ratio
        )
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")

    return factory


def image_background(path: str | Path) -> BackgroundFactory:
    """Centre-crop a photograph to the page aspect ratio, then scale it to fit."""

    def factory(width: int, height: int) -> Image.Image:
        source = Image.open(path).convert("RGB")
        target_ratio = width / height
        source_ratio = source.width / source.height
        if source_ratio > target_ratio:
            crop_width = int(source.height * target_ratio)
            left = (source.width - crop_width) // 2
            source = source.crop((left, 0, left + crop_width, source.height))
        else:
            crop_height = int(source.width / target_ratio)
            top = (source.height - crop_height) // 2
            source = source.crop((0, top, source.width, top + crop_height))
        return source.resize((width, height), Image.Resampling.LANCZOS)

    return factory


class BackgroundSampler:
    """Samples one background factory per document from weighted kinds."""

    def __init__(
        self,
        kinds: dict[str, float],
        *,
        image_paths: Sequence[str | Path] = (),
        tint_range: tuple[Color, Color] = ((238, 234, 226), (255, 255, 255)),
    ):
        self.kinds = {name: weight for name, weight in kinds.items() if weight > 0}
        if not self.kinds:
            raise ValueError("BackgroundSampler needs at least one kind with a positive weight")
        self.images = self._collect_images(image_paths)
        if "image" in self.kinds and not self.images:
            raise ValueError("background kind 'image' was requested but no images were found")
        self.tint_range = tint_range

    @staticmethod
    def _collect_images(paths: Sequence[str | Path]) -> tuple[Path, ...]:
        found: list[Path] = []
        for raw in paths:
            root = Path(raw)
            if root.is_file() and root.suffix.lower() in _IMAGE_EXTENSIONS:
                found.append(root)
            elif root.is_dir():
                found.extend(
                    path
                    for path in sorted(root.rglob("*"))
                    if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
                )
        return tuple(found)

    def _tint(self, rng: random.Random) -> Color:
        """Interpolate between the two endpoint colours with a single factor.

        Sampling each channel independently would produce hues no paper stock has — a
        pink or green page — because the channels of real paper move together.
        """
        low, high = self.tint_range
        factor = rng.random()
        return tuple(  # type: ignore[return-value]
            int(round(a + (b - a) * factor)) for a, b in zip(low, high, strict=True)
        )

    def sample(self, rng: random.Random) -> tuple[BackgroundFactory, str]:
        """Return a factory and the name of the kind it came from, for provenance."""
        names = list(self.kinds)
        kind = rng.choices(names, weights=[self.kinds[name] for name in names], k=1)[0]
        if kind == "solid":
            return solid_background(self._tint(rng)), kind
        if kind == "gradient":
            return (
                gradient_background(
                    self._tint(rng),
                    self._tint(rng),
                    rng.choice(["horizontal", "vertical", "diagonal"]),
                ),
                kind,
            )
        if kind == "image":
            return image_background(rng.choice(self.images)), kind
        return paper_background(self._tint(rng), rng.uniform(2.0, 9.0), rng), "paper"
