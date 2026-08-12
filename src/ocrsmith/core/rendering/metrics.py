"""Font measurement.

Wraps a Pillow font so that measurement always happens on the *visual* form of the text —
the same string that will be drawn. Measuring the logical form and drawing the shaped one
is a subtle way to get boxes that are consistently a few pixels wrong on Arabic.

Measurements are cached per (font, string): wrapping a paragraph asks for the width of the
same word many times over, and a generation run repeats that across millions of samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL.ImageFont import FreeTypeFont

from ...text.shaping import TextShaper, resolve_shaper

__all__ = ["FontMetrics", "TextExtent", "metrics_for"]


@dataclass(frozen=True, slots=True)
class TextExtent:
    """How much room a string takes and where its ink actually sits.

    `advance` is the pen movement (what positioning uses); the ink box is the visible
    extent (what an annotation should report). They differ by side bearings, which is why
    a tight box is never just "pen start to pen end".
    """

    advance: float
    ink_x0: float
    ink_y0: float
    ink_x1: float
    ink_y1: float

    @property
    def ink_width(self) -> float:
        return self.ink_x1 - self.ink_x0

    @property
    def ink_height(self) -> float:
        return self.ink_y1 - self.ink_y0

    @property
    def is_blank(self) -> bool:
        return self.ink_width <= 0 or self.ink_height <= 0


class FontMetrics:
    """Measures text for one font, in the form it will be drawn."""

    def __init__(self, font: FreeTypeFont, shaper: TextShaper | None = None):
        self.font = font
        self.shaper = shaper or resolve_shaper()
        ascent, descent = font.getmetrics()
        self.ascent = ascent
        self.descent = descent
        self._extent = lru_cache(maxsize=8192)(self._measure)

    # -- vertical metrics --------------------------------------------------

    @property
    def natural_line_height(self) -> float:
        """Ascender-to-descender height: the tightest spacing that never clips."""
        return float(self.ascent + self.descent)

    def line_height(self, spacing: float = 1.0) -> float:
        return self.natural_line_height * spacing

    # -- horizontal metrics ------------------------------------------------

    def visual(self, text: str) -> str:
        """The form of `text` that should be handed to the rasteriser."""
        return self.shaper.shape(text).visual

    def extent(self, text: str) -> TextExtent:
        return self._extent(text)

    def advance(self, text: str) -> float:
        return self._extent(text).advance

    @property
    def space_advance(self) -> float:
        return self._extent(" ").advance

    def _measure(self, text: str) -> TextExtent:
        visual = self.visual(text)
        if not visual:
            return TextExtent(0.0, 0.0, 0.0, 0.0, 0.0)
        advance = float(self.font.getlength(visual))
        x0, y0, x1, y1 = self.font.getbbox(visual)
        return TextExtent(advance, float(x0), float(y0), float(x1), float(y1))


#: Metrics are cached per (font file, size, shaper) rather than per call site. Building a
#: fresh `FontMetrics` for every block would throw away the measurement cache between
#: blocks, which is most of the cost of laying out a page.
_METRICS_CACHE: dict[tuple[str, float, str], FontMetrics] = {}
#: Bound so a long run over many fonts and sizes cannot grow without limit.
_MAX_CACHED_METRICS = 256


def metrics_for(font: FreeTypeFont, shaper: TextShaper | None = None) -> FontMetrics:
    """Shared `FontMetrics` for a font, reusing its measurement cache across blocks."""
    shaper = shaper or resolve_shaper()
    key = (
        str(getattr(font, "path", id(font))),
        float(getattr(font, "size", 0)),
        type(shaper).__name__,
    )
    cached = _METRICS_CACHE.get(key)
    if cached is None:
        if len(_METRICS_CACHE) >= _MAX_CACHED_METRICS:
            _METRICS_CACHE.clear()
        cached = FontMetrics(font, shaper)
        _METRICS_CACHE[key] = cached
    return cached
