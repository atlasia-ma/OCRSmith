"""Text rendering: pixels and their annotation, produced in the same pass."""

from .bidi_layout import DirectionalRun, visual_word_order
from .metrics import FontMetrics, TextExtent
from .style import Alignment, TextStyle
from .text_renderer import RenderedText, TextBlockRenderer
from .wrapping import break_long_word, fit_lines, wrap_paragraph, wrap_text

__all__ = [
    "Alignment",
    "DirectionalRun",
    "FontMetrics",
    "RenderedText",
    "TextBlockRenderer",
    "TextExtent",
    "TextStyle",
    "break_long_word",
    "fit_lines",
    "visual_word_order",
    "wrap_paragraph",
    "wrap_text",
]
