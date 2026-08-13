"""Measurable properties of a document image.

The standing criticism of synthetic OCR data is the sim-to-real gap, and it is usually
answered with an assertion. It can be measured instead: extract the same features from a
synthetic corpus and a real one, and compare the distributions.

The features here are chosen to be *causally meaningful* rather than merely numerous. Each
one corresponds to a knob the generator actually has, so a divergence points at something
you can change — if synthetic stroke width is systematically thinner than real, that is a
font-weight and ink-spread problem, not an unexplained "domain gap".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["FEATURE_NAMES", "ImageFeatures", "extract_features", "iter_image_features"]

#: Order is fixed so a feature vector is comparable across runs.
FEATURE_NAMES = (
    "ink_fraction",
    "ink_darkness",
    "paper_lightness",
    "contrast",
    "stroke_width",
    "edge_density",
    "high_frequency_energy",
    "block_noise",
    "illumination_range",
    "aspect_ratio",
)


@dataclass(frozen=True, slots=True)
class ImageFeatures:
    """One image reduced to features that map onto generator knobs."""

    ink_fraction: float
    ink_darkness: float
    paper_lightness: float
    contrast: float
    stroke_width: float
    edge_density: float
    high_frequency_energy: float
    block_noise: float
    illumination_range: float
    aspect_ratio: float

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in FEATURE_NAMES], dtype=np.float64)

    def to_dict(self) -> dict:
        return {name: round(float(getattr(self, name)), 5) for name in FEATURE_NAMES}


def _downscale(grey: np.ndarray, target: int = 1200) -> np.ndarray:
    """Cap the long edge so features do not depend on capture resolution."""
    height, width = grey.shape
    longest = max(height, width)
    if longest <= target:
        return grey
    scale = target / longest
    image = Image.fromarray(grey.astype(np.uint8))
    return np.asarray(
        image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _stroke_width(binary: np.ndarray) -> float:
    """Mean horizontal run length of ink, a cheap proxy for stroke thickness.

    Real scans thicken strokes through ink spread and thin them through erosion, and a
    generator that never varies weight shows up here immediately.
    """
    runs: list[int] = []
    for row in binary[:: max(1, binary.shape[0] // 120)]:
        length = 0
        for value in row:
            if value:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
    if not runs:
        return 0.0
    trimmed = [run for run in runs if run <= 40]  # ignore rules and filled blocks
    return float(np.mean(trimmed)) if trimmed else 0.0


def extract_features(image: Image.Image) -> ImageFeatures:
    """Reduce one page to its feature vector."""
    grey = _downscale(np.asarray(image.convert("L"), dtype=np.float32))
    height, width = grey.shape

    #: Otsu-free split: ink is what sits well below the page's own median.
    median = float(np.median(grey))
    threshold = median - 40
    ink_mask = grey < threshold
    ink_fraction = float(ink_mask.mean())

    ink_values = grey[ink_mask]
    paper_values = grey[~ink_mask]
    ink_darkness = float(ink_values.mean()) if ink_values.size else median
    paper_lightness = float(paper_values.mean()) if paper_values.size else median

    # Local contrast between ink and its own paper, not against a global constant.
    contrast = paper_lightness - ink_darkness

    gradient_x = np.abs(np.diff(grey, axis=1))
    gradient_y = np.abs(np.diff(grey, axis=0))
    edge_density = float((gradient_x > 20).mean() + (gradient_y > 20).mean()) / 2

    # High-frequency energy separates a crisp render from a blurred or downscaled capture.
    high_frequency_energy = float(gradient_x.std() + gradient_y.std()) / 2

    # JPEG works in 8x8 blocks, so compression artefacts show as energy at that period.
    columns = grey.mean(axis=0)
    block_noise = float(np.abs(np.diff(columns)[7::8]).mean()) if columns.size > 16 else 0.0

    # Lighting varies at a scale far larger than a glyph; measure it on a coarse grid.
    coarse = grey[:: max(1, height // 8), :: max(1, width // 8)]
    illumination_range = float(coarse.max() - coarse.min()) if coarse.size else 0.0

    return ImageFeatures(
        ink_fraction=ink_fraction,
        ink_darkness=ink_darkness,
        paper_lightness=paper_lightness,
        contrast=contrast,
        stroke_width=_stroke_width(ink_mask),
        edge_density=edge_density,
        high_frequency_energy=high_frequency_energy,
        block_noise=block_noise,
        illumination_range=illumination_range,
        aspect_ratio=float(width / height) if height else 0.0,
    )


def iter_image_features(directory: str | Path, limit: int = 0):
    """Extract features from every image under `directory`."""
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    count = 0
    for path in sorted(Path(directory).rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        try:
            with Image.open(path) as image:
                yield path, extract_features(image)
        except Exception:
            continue  # an unreadable file is not worth ending a comparison over
        count += 1
        if limit and count >= limit:
            return
