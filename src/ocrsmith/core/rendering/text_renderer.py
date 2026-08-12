"""Text block rendering with word-level ground truth.

This is the component the whole project exists to get right. It turns a string into
pixels *and* into the annotation describing those pixels, and the two are produced by the
same pass so they cannot disagree.

Words are drawn one at a time rather than a line at a time. That costs a little speed and
buys exact per-word boxes, and it is safe for Arabic because letters never join across a
space — the only place joining matters is inside a word, which is still shaped as a unit.

Coordinates are relative to the returned image; the caller translates them once when the
block is pasted onto a page.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont

from ...domain.annotations import Line, Word
from ...domain.geometry import BBox
from ...text.script import Direction, detect_direction
from ...text.shaping import TextShaper, resolve_shaper
from .bidi_layout import visual_word_order
from .metrics import FontMetrics, metrics_for
from .style import Alignment, TextStyle
from .wrapping import fit_lines, wrap_text

__all__ = ["RenderedText", "TextBlockRenderer"]

#: Transparent canvases are grown by this many pixels on each side before drawing, so
#: strokes, synthetic bold and descenders are never clipped by the tight layout box.
_BLEED = 4


@dataclass(frozen=True, slots=True)
class _WordPlan:
    """One word and the x offset it will be drawn at, relative to the line origin."""

    text: str
    offset: float
    advance: float


@dataclass(frozen=True, slots=True)
class _LinePlan:
    """A laid-out line: its logical text, its words, and the width it occupies."""

    text: str
    words: tuple[_WordPlan, ...]
    width: float


@dataclass(frozen=True, slots=True)
class RenderedText:
    """A rendered block of text and the annotation describing it."""

    image: Image.Image
    lines: tuple[Line, ...]
    #: Lines that did not fit the height budget and were therefore not drawn.
    dropped_lines: int = 0
    #: Transparent margin around the text, so strokes and descenders are not clipped.
    #: The text origin sits at (padding, padding) inside `image`.
    padding: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def layout_size(self) -> tuple[int, int]:
        """Size the block occupies in a layout, excluding the transparent bleed."""
        width, height = self.image.size
        return (max(0, width - 2 * self.padding), max(0, height - 2 * self.padding))

    @property
    def text(self) -> str:
        """Logical-order transcription of exactly what was drawn."""
        return "\n".join(line.text for line in self.lines)

    @property
    def bbox(self) -> BBox:
        if not self.lines:
            return BBox(0, 0, *self.image.size)
        return BBox.union_all(line.bbox for line in self.lines)

    def translated(self, dx: float, dy: float) -> tuple[Line, ...]:
        """The annotation as it would read once the block is pasted at (dx, dy)."""
        return tuple(line.translate(dx, dy) for line in self.lines)

    def place(self, canvas: Image.Image, x: float, y: float) -> tuple[Line, ...]:
        """Composite this block so its text origin lands exactly at (x, y).

        The bleed is compensated for here rather than by every caller, so a block placed
        at a column's left edge starts at that edge instead of `padding` pixels inside it.
        """
        left = int(round(x)) - self.padding
        top = int(round(y)) - self.padding
        canvas.alpha_composite(self.image, (max(0, left), max(0, top)))
        return self.translated(max(0, left), max(0, top))


class TextBlockRenderer:
    """Renders a block of text onto a transparent canvas, with word-level annotation."""

    def __init__(self, shaper: TextShaper | None = None):
        self.shaper = shaper or resolve_shaper()

    def render(
        self,
        text: str,
        font: FreeTypeFont,
        style: TextStyle | None = None,
        *,
        max_width: float | None = None,
        max_height: float | None = None,
        direction: Direction | None = None,
        rng: random.Random | None = None,
    ) -> RenderedText:
        """Draw `text` and return it together with its per-word annotation."""
        style = style or TextStyle()
        rng = rng or random.Random()
        metrics = metrics_for(font, self.shaper)
        base_direction = direction or detect_direction(text)

        line_texts = list(wrap_text(text.split("\n"), metrics.advance, max_width))
        line_height = metrics.line_height(style.line_spacing)
        line_texts, dropped = fit_lines(line_texts, line_height, max_height)

        if not line_texts:
            return RenderedText(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), (), dropped, _BLEED)

        plans = [self._plan_line(text_line, metrics, style, base_direction, rng) for text_line in line_texts]
        content_width = max((plan.width for plan in plans), default=0.0)
        block_width = max_width if max_width and max_width > 0 else content_width
        block_width = max(block_width, content_width)

        canvas_width = int(round(block_width)) + 2 * _BLEED
        canvas_height = int(round(line_height * len(plans))) + 2 * _BLEED
        image = Image.new("RGBA", (max(1, canvas_width), max(1, canvas_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        lines: list[Line] = []
        for index, plan in enumerate(plans):
            if not plan.words:
                continue
            origin_x = _BLEED + self._align_offset(plan.width, block_width, style, base_direction)
            origin_y = _BLEED + index * line_height
            if style.baseline_jitter:
                origin_y += rng.uniform(-style.baseline_jitter, style.baseline_jitter)
            lines.append(self._draw_line(draw, plan, metrics, style, origin_x, origin_y, base_direction))

        image = self._apply_synthetic_italic(image, style)
        return RenderedText(image, tuple(lines), dropped, _BLEED)

    # -- line planning -----------------------------------------------------

    def _plan_line(
        self,
        text: str,
        metrics: FontMetrics,
        style: TextStyle,
        base: Direction,
        rng: random.Random,
    ) -> _LinePlan:
        """Decide where each word of a line sits, in left-to-right screen order."""
        words = text.split()
        if not words:
            return _LinePlan(text="", words=(), width=0.0)

        order = visual_word_order(words, base)
        gap = metrics.space_advance * style.word_spacing

        placed: list[_WordPlan | None] = [None] * len(words)
        cursor = 0.0
        for position, index in enumerate(order):
            if position:
                extra = rng.uniform(0, style.word_spacing_jitter) if style.word_spacing_jitter else 0.0
                cursor += gap + extra
            advance = metrics.advance(words[index])
            placed[index] = _WordPlan(words[index], cursor, advance)
            cursor += advance

        # Words are stored in logical order so the annotation reads like the label,
        # while `offset` keeps the position they were actually drawn at.
        return _LinePlan(
            text=" ".join(words),
            words=tuple(plan for plan in placed if plan is not None),
            width=cursor,
        )

    # -- drawing -----------------------------------------------------------

    def _draw_line(
        self,
        draw: ImageDraw.ImageDraw,
        plan: _LinePlan,
        metrics: FontMetrics,
        style: TextStyle,
        origin_x: float,
        origin_y: float,
        base: Direction,
    ) -> Line:
        stroke_width = int(round(max(style.stroke_width, style.synthetic_bold)))
        stroke_fill = style.stroke_fill if style.stroke_width else style.fill

        words: list[Word] = []
        for word_plan in plan.words:
            x = origin_x + word_plan.offset
            visual = metrics.visual(word_plan.text)
            draw.text(
                (x, origin_y),
                visual,
                font=metrics.font,
                fill=style.fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            extent = metrics.extent(word_plan.text)
            box = BBox(
                x + extent.ink_x0 - stroke_width,
                origin_y + extent.ink_y0 - stroke_width,
                x + extent.ink_x1 + stroke_width,
                origin_y + extent.ink_y1 + stroke_width,
            )
            words.append(Word(word_plan.text, box))

        line_box = BBox.union_all(word.bbox for word in words)
        baseline = origin_y + metrics.ascent

        if style.underline:
            y = baseline + max(1.0, metrics.descent * 0.35)
            draw.line((line_box.x0, y, line_box.x1, y), fill=style.fill, width=max(1, stroke_width or 1))
        if style.strikethrough:
            y = origin_y + metrics.ascent * 0.6
            draw.line((line_box.x0, y, line_box.x1, y), fill=style.fill, width=max(1, stroke_width or 1))

        return Line(
            text=plan.text,
            bbox=line_box,
            words=tuple(words),
            direction=base,
            baseline=baseline,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _align_offset(
        line_width: float,
        block_width: float,
        style: TextStyle,
        base: Direction,
    ) -> float:
        slack = max(0.0, block_width - line_width)
        align = style.align
        if align is Alignment.NATURAL:
            align = Alignment.RIGHT if base.is_rtl else Alignment.LEFT
        if align is Alignment.RIGHT:
            return slack
        if align is Alignment.CENTER:
            return slack / 2
        return 0.0  # LEFT, and JUSTIFY which stretches gaps rather than shifting

    @staticmethod
    def _apply_synthetic_italic(image: Image.Image, style: TextStyle) -> Image.Image:
        """Shear the canvas to fake an oblique face.

        Boxes are intentionally *not* sheared with it: an oblique glyph still occupies
        roughly its upright cell, and reporting a sheared quad here would imply a
        precision the transform does not have.
        """
        if not style.synthetic_italic:
            return image
        shear = style.synthetic_italic
        width, height = image.size
        new_width = int(width + abs(shear) * height)
        return image.transform(
            (new_width, height),
            Image.AFFINE,
            (1, shear, -shear * height if shear > 0 else 0, 0, 1, 0),
            resample=Image.BICUBIC,
        )
