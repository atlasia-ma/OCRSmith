"""Script and direction detection.

Everything downstream — shaping backend, line alignment, numeral rendering, even which
fonts are eligible — depends on knowing whether a string is Arabic, Latin or a mix, so
detection lives in one place with one definition.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Direction", "Script", "detect_direction", "detect_script", "is_arabic_char"]

# Arabic (0600-06FF), Arabic Supplement (0750-077F), Extended-A (08A0-08FF),
# Presentation Forms-A (FB50-FDFF) and Forms-B (FE70-FEFF).
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)

# Other right-to-left scripts we should not mistake for Latin.
_RTL_RANGES = _ARABIC_RANGES + (
    (0x0590, 0x05FF),  # Hebrew
    (0x0700, 0x074F),  # Syriac
    (0x07C0, 0x08FF),  # NKo, Thaana neighbours
)

_LATIN_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),  # Latin-1 Supplement through Latin Extended-B
)


class Script(str, Enum):
    """Dominant writing system of a piece of text."""

    ARABIC = "arabic"
    LATIN = "latin"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class Direction(str, Enum):
    """Base paragraph direction."""

    LTR = "ltr"
    RTL = "rtl"

    @property
    def is_rtl(self) -> bool:
        return self is Direction.RTL


def _in_ranges(code_point: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= code_point <= high for low, high in ranges)


def is_arabic_char(char: str) -> bool:
    """Whether a single character belongs to the Arabic script blocks."""
    return bool(char) and _in_ranges(ord(char[0]), _ARABIC_RANGES)


def detect_script(text: str) -> Script:
    """Classify `text` as Arabic, Latin, a mix of both, or script-neutral.

    Digits, punctuation and whitespace are neutral: a string made only of them
    carries no script signal and is reported as `Script.NEUTRAL`.
    """
    has_arabic = False
    has_latin = False
    for char in text:
        code_point = ord(char)
        if not has_arabic and _in_ranges(code_point, _ARABIC_RANGES):
            has_arabic = True
        elif not has_latin and _in_ranges(code_point, _LATIN_RANGES):
            has_latin = True
        if has_arabic and has_latin:
            return Script.MIXED
    if has_arabic:
        return Script.ARABIC
    if has_latin:
        return Script.LATIN
    return Script.NEUTRAL


def detect_direction(text: str) -> Direction:
    """Base direction from the first strong character, as the Unicode bidi algorithm does.

    Neutral leading characters (digits, punctuation, whitespace) are skipped, so
    ``"2024 مرحبا"`` is right-to-left while ``"2024"`` alone falls back to left-to-right.
    """
    for char in text:
        code_point = ord(char)
        if _in_ranges(code_point, _RTL_RANGES):
            return Direction.RTL
        if _in_ranges(code_point, _LATIN_RANGES):
            return Direction.LTR
    return Direction.LTR
