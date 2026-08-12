"""Font glyph coverage.

A font that lacks a glyph does not fail — it draws a blank or a tofu box while the label
still claims the character is there. At scale that quietly teaches a model to hallucinate.
Coverage is therefore checked up front and unsupported (font, text) pairs are rejected
before they can reach the dataset.

Coverage is read from the font's `cmap` via fontTools, which is exact and needs no
rendering. Results are cached per font file because a generation run asks the same
question millions of times.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "CoverageReport",
    "fonts_supporting",
    "has_glyph",
    "missing_glyphs",
    "supports_text",
]

# Characters no font is expected to carry a glyph for and which never need one.
_IGNORED = set(" \t\n\r​‌‍‎‏﻿")


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Which characters of a text a font can actually draw."""

    font_path: str
    total: int
    missing: tuple[str, ...]

    @property
    def covered(self) -> int:
        return self.total - len(self.missing)

    @property
    def ratio(self) -> float:
        """Fraction of checkable characters the font can draw; 1.0 for empty text."""
        return 1.0 if self.total == 0 else self.covered / self.total

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def __bool__(self) -> bool:
        return self.is_complete


@lru_cache(maxsize=512)
def _codepoints(font_path: str) -> frozenset[int]:
    """Every code point mapped by the font's character map."""
    from fontTools.ttLib import TTFont

    covered: set[int] = set()
    with TTFont(font_path, fontNumber=0, lazy=True) as font:
        for table in font["cmap"].tables:
            covered.update(table.cmap.keys())
    return frozenset(covered)


def has_glyph(font_path: str | Path, char: str) -> bool:
    """Whether the font at `font_path` maps `char` to a glyph."""
    if not char:
        return True
    if char in _IGNORED:
        return True
    return ord(char[0]) in _codepoints(str(Path(font_path)))


def missing_glyphs(font_path: str | Path, text: str) -> tuple[str, ...]:
    """Unique characters of `text` the font cannot draw, in first-seen order."""
    covered = _codepoints(str(Path(font_path)))
    missing: list[str] = []
    seen: set[str] = set()
    for char in text:
        if char in _IGNORED or char in seen:
            continue
        seen.add(char)
        if ord(char) not in covered:
            missing.append(char)
    return tuple(missing)


def supports_text(font_path: str | Path, text: str) -> CoverageReport:
    """Full coverage report for rendering `text` with the font at `font_path`."""
    checkable = {char for char in text if char not in _IGNORED}
    return CoverageReport(
        font_path=str(font_path),
        total=len(checkable),
        missing=missing_glyphs(font_path, text),
    )


def fonts_supporting(
    font_paths: Iterable[str | Path],
    text: str,
    *,
    min_ratio: float = 1.0,
) -> Sequence[str]:
    """Subset of `font_paths` that can draw `text` to at least `min_ratio` coverage.

    Used to pick a font per sample instead of discovering mid-render that the chosen
    face has no Arabic in it.
    """
    eligible: list[str] = []
    for path in font_paths:
        try:
            report = supports_text(path, text)
        except Exception:
            continue  # unreadable or exotic font container: treat as ineligible
        if report.ratio >= min_ratio:
            eligible.append(str(path))
    return eligible
