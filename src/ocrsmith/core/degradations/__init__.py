"""Degradations: the gap between a clean render and a real capture."""

from .base import (
    Degradation,
    DegradationPipeline,
    DegradationRecord,
    PhotometricDegradation,
    map_page,
)
from .geometric import PerspectiveWarp, Rotation, perspective_coefficients
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
from .presets import PRESETS, build_preset, preset_names

__all__ = [
    "PRESETS",
    "Bleedthrough",
    "Blur",
    "Brightness",
    "Contrast",
    "Degradation",
    "DegradationPipeline",
    "DegradationRecord",
    "Downscale",
    "Folds",
    "GaussianNoise",
    "Glare",
    "InkErosion",
    "InkSpread",
    "JpegArtifacts",
    "MotionBlur",
    "PaperGrain",
    "PerspectiveWarp",
    "PhotometricDegradation",
    "Rotation",
    "Shadow",
    "Stains",
    "Vignette",
    "build_preset",
    "map_page",
    "perspective_coefficients",
    "preset_names",
]
