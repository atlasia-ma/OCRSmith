"""Text normalisation policies.

Every transform here changes the ground-truth label, not just the pixels, so each one is
opt-in and recorded on the sample. A dataset that silently strips diacritics teaches a
model to drop them; that has to be a deliberate choice.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "NormalizationPolicy",
    "NumeralSystem",
    "normalize_text",
    "strip_diacritics",
    "strip_tatweel",
    "to_numeral_system",
]

TATWEEL = "ـ"

# Tashkeel, Quranic annotation marks and the superscript alef.
_DIACRITICS = re.compile(
    "["
    "ؐ-ؚ"
    "ً-ٟ"
    "ٰ"
    "ۖ-ۜ"
    "۟-ۨ"
    "۪-ۭ"
    "࣓-ࣿ"
    "]"
)

_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_YA_VARIANTS = str.maketrans({"ى": "ي"})
_TA_MARBUTA = str.maketrans({"ة": "ه"})

_WESTERN_DIGITS = "0123456789"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_EASTERN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

_HORIZONTAL_WS = re.compile(r"[^\S\n]+")
_ALL_WS = re.compile(r"\s+")


class NumeralSystem(str, Enum):
    """Which digit shapes appear in the rendered text and in the label."""

    KEEP = "keep"
    WESTERN = "western"
    ARABIC_INDIC = "arabic_indic"
    EASTERN_ARABIC_INDIC = "eastern_arabic_indic"


_DIGIT_TABLES: dict[NumeralSystem, str] = {
    NumeralSystem.WESTERN: _WESTERN_DIGITS,
    NumeralSystem.ARABIC_INDIC: _ARABIC_INDIC_DIGITS,
    NumeralSystem.EASTERN_ARABIC_INDIC: _EASTERN_DIGITS,
}

_ALL_DIGITS = _WESTERN_DIGITS + _ARABIC_INDIC_DIGITS + _EASTERN_DIGITS


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel and Quranic annotation marks, leaving the skeleton."""
    return _DIACRITICS.sub("", text)


def strip_tatweel(text: str) -> str:
    """Remove kashida (tatweel) elongation characters."""
    return text.replace(TATWEEL, "")


def to_numeral_system(text: str, system: NumeralSystem) -> str:
    """Rewrite every digit in `text` using `system`'s digit shapes."""
    if system is NumeralSystem.KEEP:
        return text
    target = _DIGIT_TABLES[system]
    table = {ord(digit): target[index % 10] for index, digit in enumerate(_ALL_DIGITS)}
    return text.translate(table)


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    """Declarative description of how raw source text becomes a label.

    Defaults are deliberately conservative: only runs of whitespace are collapsed, which
    no OCR label format preserves anyway.
    """

    collapse_whitespace: bool = True
    preserve_line_breaks: bool = False
    strip_diacritics: bool = False
    strip_tatweel: bool = False
    unify_alef: bool = False
    unify_ya: bool = False
    unify_ta_marbuta: bool = False
    numerals: NumeralSystem = NumeralSystem.KEEP
    unicode_form: str | None = "NFC"

    def apply(self, text: str) -> str:
        return normalize_text(text, self)


def normalize_text(text: str, policy: NormalizationPolicy | None = None) -> str:
    """Apply `policy` to `text`.

    The order is fixed and the result is idempotent: composing characters are folded
    first so that later character-level rules see a canonical form.
    """
    policy = policy or NormalizationPolicy()

    if policy.unicode_form:
        text = unicodedata.normalize(policy.unicode_form, text)
    if policy.strip_diacritics:
        text = strip_diacritics(text)
    if policy.strip_tatweel:
        text = strip_tatweel(text)
    if policy.unify_alef:
        text = text.translate(_ALEF_VARIANTS)
    if policy.unify_ya:
        text = text.translate(_YA_VARIANTS)
    if policy.unify_ta_marbuta:
        text = text.translate(_TA_MARBUTA)
    text = to_numeral_system(text, policy.numerals)

    if policy.collapse_whitespace:
        if policy.preserve_line_breaks:
            text = _HORIZONTAL_WS.sub(" ", text)
            # Keep one break between non-empty lines and no leading/trailing padding,
            # so a paragraph's line structure survives without ragged indentation.
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        else:
            text = _ALL_WS.sub(" ", text)
        text = text.strip()
    return text
