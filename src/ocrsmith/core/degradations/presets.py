"""Named degradation pipelines.

A corpus is only as good as its coverage of *capture conditions*. These presets describe
recognisable ones, so a dataset can be composed as "40% flatbed scan, 40% phone photo, 20%
clean" rather than as an undifferentiated cloud of noise. Ordering within each preset
follows the physical chain: the page is distorted by the optics first, then shaded, then
sampled and compressed.
"""

from __future__ import annotations

from .base import DegradationPipeline
from .geometric import PerspectiveWarp, Rotation
from .photometric import (
    Bleedthrough,
    Blur,
    Brightness,
    Contrast,
    Downscale,
    Folds,
    GaussianNoise,
    Glare,
    InkErosion,
    InkSpread,
    JpegArtifacts,
    MotionBlur,
    PaperGrain,
    Shadow,
    Stains,
    Vignette,
)
from .physical import IlluminationField, PageCurl, Wrinkles

__all__ = ["PRESETS", "build_preset", "preset_names"]


def _clean() -> DegradationPipeline:
    """A born-digital render: no capture stage at all."""
    return DegradationPipeline()


def _scan() -> DegradationPipeline:
    """Flatbed scan: slight skew, paper texture, toner spread, mild compression."""
    return DegradationPipeline(
        [
            Rotation(max_angle=1.5, probability=0.7),
            PaperGrain(strength=(4.0, 12.0), probability=0.8),
            Bleedthrough(strength=(0.03, 0.12), probability=0.25),
            InkSpread(size=3, probability=0.2),
            Blur(radius=(0.2, 0.7), probability=0.5),
            GaussianNoise(sigma=(2.0, 8.0), probability=0.6),
            Brightness(factor=(0.92, 1.08), probability=0.5),
            JpegArtifacts(quality=(60, 92), probability=0.6),
        ]
    )


def _photo() -> DegradationPipeline:
    """Phone photograph: perspective, uneven light, glare, motion, heavy compression."""
    return DegradationPipeline(
        [
            # A photographed page sits on a surface, not on more paper, so the area the
            # warp exposes is filled with a neutral desk grey rather than white.
            PerspectiveWarp(magnitude=(0.01, 0.06), fill=(118, 116, 112), probability=0.85),
            Rotation(max_angle=4.0, fill=(118, 116, 112), probability=0.6),
            # The sheet is not flat and the light is not even; both are what separates a
            # photographed page from a scanned one.
            PageCurl(strength=(0.02, 0.09), probability=0.35),
            Wrinkles(strength=(1.5, 5.0), probability=0.4),
            IlluminationField(strength=(0.12, 0.4), probability=0.7),
            Shadow(strength=(0.1, 0.45), probability=0.7),
            Glare(strength=(0.1, 0.4), probability=0.35),
            Vignette(strength=(0.1, 0.4), probability=0.5),
            MotionBlur(length=(3, 9), probability=0.3),
            Blur(radius=(0.3, 1.4), probability=0.5),
            Downscale(scale=(0.45, 0.9), probability=0.5),
            GaussianNoise(sigma=(4.0, 16.0), probability=0.7),
            JpegArtifacts(quality=(25, 70), probability=0.9),
        ]
    )


def _fax() -> DegradationPipeline:
    """Low-resolution, high-contrast transmission with broken strokes."""
    return DegradationPipeline(
        [
            Rotation(max_angle=2.0, probability=0.5),
            Downscale(scale=(0.3, 0.55), probability=0.9),
            Contrast(factor=(1.4, 2.2), probability=0.9),
            InkErosion(size=3, probability=0.5),
            GaussianNoise(sigma=(6.0, 20.0), probability=0.7),
            JpegArtifacts(quality=(20, 55), probability=0.7),
        ]
    )


def _archive() -> DegradationPipeline:
    """An aged document: stains, folds, bleed-through and a faded, low-contrast image."""
    return DegradationPipeline(
        [
            Rotation(max_angle=2.5, probability=0.6),
            PaperGrain(strength=(8.0, 22.0), probability=0.9),
            Stains(count=(1, 4), strength=(0.05, 0.25), probability=0.8),
            Folds(count=(1, 3), strength=(0.08, 0.3), probability=0.6),
            Wrinkles(strength=(2.0, 7.0), shading=(0.08, 0.25), probability=0.7),
            Bleedthrough(strength=(0.08, 0.22), probability=0.6),
            InkErosion(size=3, probability=0.4),
            Contrast(factor=(0.6, 0.95), probability=0.7),
            Brightness(factor=(0.8, 1.05), probability=0.6),
            GaussianNoise(sigma=(3.0, 12.0), probability=0.6),
            JpegArtifacts(quality=(45, 85), probability=0.5),
        ]
    )


#: Preset name to factory. Factories rather than instances, so each caller gets a
#: pipeline it can mutate without affecting anyone else.
PRESETS = {
    "clean": _clean,
    "scan": _scan,
    "photo": _photo,
    "fax": _fax,
    "archive": _archive,
}


def preset_names() -> tuple[str, ...]:
    return tuple(sorted(PRESETS))


def build_preset(name: str) -> DegradationPipeline:
    """Instantiate the named preset pipeline."""
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown degradation preset {name!r}. Available: {', '.join(preset_names())}"
        ) from None
