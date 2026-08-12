"""Typographic style for a block of text.

Everything a renderer needs to know beyond "what does it say", collected in one
immutable object so a style can be sampled once, recorded in provenance, and replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Alignment", "TextStyle"]


class Alignment(str, Enum):
    """Horizontal alignment of lines inside a text block."""

    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    JUSTIFY = "justify"
    #: Left for left-to-right text, right for right-to-left text.
    NATURAL = "natural"


@dataclass(frozen=True, slots=True)
class TextStyle:
    """How a block of text is drawn.

    The jitter fields exist because perfectly regular typesetting is a distribution
    shift: scanned and photographed documents never have pixel-exact baselines or
    identical word gaps, and a model trained only on perfect spacing learns to rely on it.
    """

    color: tuple[int, int, int] = (0, 0, 0)
    opacity: int = 255
    stroke_width: int = 0
    stroke_color: tuple[int, int, int] = (0, 0, 0)
    align: Alignment = Alignment.NATURAL
    #: Multiplier on the font's natural line height.
    line_spacing: float = 1.2
    #: Multiplier on the width of a space between words.
    word_spacing: float = 1.0
    #: Maximum vertical wobble applied per line, in pixels.
    baseline_jitter: float = 0.0
    #: Maximum extra horizontal gap applied per word, in pixels.
    word_spacing_jitter: float = 0.0
    underline: bool = False
    strikethrough: bool = False
    #: Synthetic emboldening, in pixels of stroke. Real bold faces are preferred.
    synthetic_bold: float = 0.0
    #: Synthetic oblique shear factor. Real italic faces are preferred.
    synthetic_italic: float = 0.0

    @property
    def fill(self) -> tuple[int, int, int, int]:
        return (*self.color, self.opacity)

    @property
    def stroke_fill(self) -> tuple[int, int, int, int]:
        return (*self.stroke_color, self.opacity)

    def with_(self, **changes) -> TextStyle:
        """Copy with selected fields replaced."""
        from dataclasses import replace

        return replace(self, **changes)
