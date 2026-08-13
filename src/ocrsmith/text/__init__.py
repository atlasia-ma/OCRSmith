"""Text handling: script detection, normalisation, shaping and font coverage.

This package owns everything that happens to a string *before* it becomes pixels.
"""

from .coverage import (
    CoverageReport,
    fonts_supporting,
    has_glyph,
    missing_glyphs,
    supports_text,
)
from .diacritics import (
    DiacriticsMode,
    DiacriticsPolicy,
    apply_diacritics,
    count_diacritics,
    diacritic_ratio,
    strip_partial,
)
from .normalization import (
    NormalizationPolicy,
    NumeralSystem,
    normalize_text,
    strip_diacritics,
    strip_tatweel,
    to_numeral_system,
)
from .script import Direction, Script, detect_direction, detect_script, is_arabic_char
from .shaping import (
    ReshaperBidiShaper,
    ShapedText,
    TextShaper,
    TransparentShaper,
    raqm_available,
    resolve_shaper,
)

__all__ = [
    # script
    "Direction",
    "Script",
    "detect_direction",
    "detect_script",
    "is_arabic_char",
    # normalisation
    "NormalizationPolicy",
    "NumeralSystem",
    "normalize_text",
    "strip_diacritics",
    "strip_tatweel",
    "to_numeral_system",
    # shaping
    "ReshaperBidiShaper",
    "ShapedText",
    "TextShaper",
    "TransparentShaper",
    "raqm_available",
    "resolve_shaper",
    # diacritics
    "DiacriticsMode",
    "DiacriticsPolicy",
    "apply_diacritics",
    "count_diacritics",
    "diacritic_ratio",
    "strip_partial",
    # coverage
    "CoverageReport",
    "fonts_supporting",
    "has_glyph",
    "missing_glyphs",
    "supports_text",
]
