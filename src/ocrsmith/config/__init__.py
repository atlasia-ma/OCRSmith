"""Configuration: one validated object describing a whole corpus."""

from .loader import DEFAULT_CONFIG_PATH, apply_overrides, load_config
from .schema import (
    BackgroundConfig,
    DegradationConfig,
    FontConfig,
    GenerationConfig,
    NormalizationConfig,
    OutputConfig,
    PageConfig,
    RunConfig,
    TemplateConfig,
    TextConfig,
    TextSourceConfig,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "BackgroundConfig",
    "DegradationConfig",
    "FontConfig",
    "GenerationConfig",
    "NormalizationConfig",
    "OutputConfig",
    "PageConfig",
    "RunConfig",
    "TemplateConfig",
    "TextConfig",
    "TextSourceConfig",
    "apply_overrides",
    "load_config",
]
