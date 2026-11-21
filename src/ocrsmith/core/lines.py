"""Single-line rendering, for recognition training data.

A recognition model is trained on strips: one line of text, a few tens of pixels tall.
Producing those by laying out a whole A4 page and then cutting it up is correct — the
crops are real crops of a real document, which is why `output.line_crops` exists — but it
costs a full page render per handful of lines, and most of that render is thrown away.

This module takes the other route. It draws the line directly onto a strip the size of the
line, which is roughly a thousand times fewer pixels than an A4 page at 200 dpi, and that
ratio is essentially the whole speed difference between this and the document pipeline.

What it deliberately keeps
--------------------------
The cheap way to write this is a white canvas, one font, `ImageDraw.text`, done — and that
is what most line generators are. It is also why their output is recognisable at a glance
and why models trained on it fall over on real pages. So this shares the document
pipeline's machinery rather than reimplementing a simpler version of it:

* the same `TextBlockRenderer`, so a line carries exact per-word boxes and the same
  logical-order label, shaped the same way;
* the same `BackgroundSampler`, so a strip sits on tinted stock with fibre texture, a
  gradient or a photograph, not on #FFFFFF;
* the same coverage-checked font pool, so no strip ever contains a tofu box;
* the same degradation presets, at line scale.

And one thing a cut-from-nothing generator cannot have: **neighbour bleed**. A real line
crop taken from a real page almost always clips a sliver of the line above or below,
because no cropper is perfect. A corpus whose strips are always perfectly isolated teaches
a model that the top and bottom rows of pixels are always background, and that assumption
breaks on the first tightly-set page it meets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image
from PIL.ImageFont import FreeTypeFont

from ..domain.annotations import Line, Page, Region, RegionType
from ..domain.geometry import BBox
from ..text.script import Direction
from .rendering.style import Alignment, TextStyle
from .rendering.text_renderer import TextBlockRenderer

__all__ = ["LineRenderer", "LineStrip", "RenderedLine"]


@dataclass(frozen=True, slots=True)
class LineStrip:
    """The shape of one strip: how tall the text is and how much room surrounds it."""

    #: Target height of the text's own line box, in pixels. The strip is taller.
    text_height: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int

    @property
    def horizontal_padding(self) -> int:
        return self.pad_left + self.pad_right

    @property
    def vertical_padding(self) -> int:
        return self.pad_top + self.pad_bottom


@dataclass(frozen=True, slots=True)
class RenderedLine:
    """A finished strip and the annotation describing it."""

    image: Image.Image
    line: Line

    @property
    def text(self) -> str:
        return self.line.text

    def as_page(self) -> Page:
        """The strip as a one-region, one-line `Page`.

        Everything downstream — the writers, the quality gate, the geometric
        degradations — speaks `Page`. A strip is a page with one line on it, so saying so
        here means none of them needs a second code path, and a recognition dataset can
        be written in any of the formats a document dataset can.
        """
        width, height = self.image.size
        region = Region(
            type=RegionType.PARAGRAPH,
            bbox=self.line.bbox,
            lines=(self.line,),
            reading_order=0,
        )
        return Page(width, height, (region,), self.line.direction)


class LineRenderer:
    """Draws one line of text onto a strip, with its per-word annotation."""

    def __init__(self, text_renderer: TextBlockRenderer | None = None):
        self.text_renderer = text_renderer or TextBlockRenderer()

    def render(
        self,
        text: str,
        font: FreeTypeFont,
        background: Image.Image | None,
        strip: LineStrip,
        *,
        style: TextStyle | None = None,
        direction: Direction | None = None,
        neighbours: tuple[str, str] | None = None,
        rng: random.Random | None = None,
    ) -> RenderedLine:
        """Draw `text` as a single strip.

        `background` is a factory-produced patch at least as large as the strip; it is
        cropped to fit, so the caller can generate one page-sized texture and cut many
        strips from it. `neighbours` is `(above, below)` — text to clip at the top and
        bottom edges, which is what a real crop of a real page contains.
        """
        rng = rng or random.Random()
        style = style or TextStyle()
        # NATURAL alignment would right-align an RTL line inside the strip, which is right
        # for a column and wrong here: a strip *is* the line, so any slack would read as a
        # stray margin rather than as alignment.
        style = style.with_(align=Alignment.LEFT, line_spacing=1.0)

        rendered = self.text_renderer.render(
            text.replace("\n", " ").strip(),
            font,
            style,
            direction=direction,
            rng=rng,
        )
        content_width, content_height = rendered.layout_size
        if content_width <= 0 or content_height <= 0:
            blank = Image.new("RGB", (max(1, strip.horizontal_padding), max(1, strip.text_height)))
            empty = Line(text="", bbox=BBox(0, 0, 0, 0), words=(), direction=direction or Direction.LTR)
            return RenderedLine(blank, empty)

        width = content_width + strip.horizontal_padding
        height = content_height + strip.vertical_padding
        canvas = self._canvas(background, width, height, rng)

        if neighbours:
            self._bleed(canvas, neighbours, font, style, strip, direction, rng)

        composite = canvas.convert("RGBA")
        lines = rendered.place(composite, strip.pad_left, strip.pad_top)
        flat = Image.new("RGB", composite.size, (255, 255, 255))
        flat.paste(composite, mask=composite.split()[-1])

        line = lines[0] if lines else Line("", BBox(0, 0, 0, 0), (), direction or Direction.LTR)
        return RenderedLine(flat, line)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _canvas(
        background: Image.Image | None, width: int, height: int, rng: random.Random
    ) -> Image.Image:
        """A strip-sized patch of `background`, taken from a random offset within it.

        A random offset rather than the top-left corner, so a thousand strips cut from one
        generated texture do not all share the same fibres. The texture is usually much
        larger than a strip, which is the whole point of generating it once. The offset
        comes from the sample's own rng, so a strip stays reproducible from its seed.
        """
        if background is None:
            return Image.new("RGB", (width, height), (255, 255, 255))
        # `convert` on an image already in RGB returns a copy, and the texture is far
        # larger than the strip — copying all of it to cut a sliver out was the single
        # largest memcpy in the line pipeline.
        source = background if background.mode == "RGB" else background.convert("RGB")
        if source.size[0] < width or source.size[1] < height:
            source = source.resize((max(width, source.size[0]), max(height, source.size[1])))
        left = rng.randrange(0, max(1, source.size[0] - width + 1))
        top = rng.randrange(0, max(1, source.size[1] - height + 1))
        return source.crop((left, top, left + width, top + height))

    def _bleed(
        self,
        canvas: Image.Image,
        neighbours: tuple[str, str],
        font: FreeTypeFont,
        style: TextStyle,
        strip: LineStrip,
        direction: Direction | None,
        rng: random.Random,
    ) -> None:
        """Clip a sliver of the lines above and below into the strip's margins.

        Drawn directly onto the background and *not* reported in the annotation, which is
        the honest description of what a crop contains: ink that belongs to another line.
        Teaching a model that those pixels are text it must read would be worse than not
        having them at all.
        """
        above, below = neighbours
        width, height = canvas.size
        layer = canvas.convert("RGBA")
        for text, edge in ((above, "top"), (below, "bottom")):
            if not text or rng.random() > 0.5:
                continue
            margin = strip.pad_top if edge == "top" else strip.pad_bottom
            # Strictly inside the margin. A neighbour that reached the text would be ink
            # the label does not mention sitting on top of ink it does, which is not
            # "a realistic crop", it is a wrong label.
            visible = rng.randint(1, max(1, margin - 2)) if margin > 3 else 1

            neighbour = self.text_renderer.render(
                text.replace("\n", " ").strip()[:120], font, style, direction=direction, rng=rng
            )
            source = neighbour.image
            if source.height <= visible or source.width < 2:
                continue
            # Sliced here rather than placed at a negative offset: compositing clamps a
            # negative origin to zero, which pasted the neighbour's *whole* line into the
            # strip instead of the few rows that were meant to show.
            sliver = (
                source.crop((0, source.height - visible, source.width, source.height))
                if edge == "top"
                else source.crop((0, 0, source.width, visible))
            )
            if sliver.width > width:
                sliver = sliver.crop((0, 0, width, sliver.height))
            x = rng.randint(0, max(0, width - sliver.width))
            layer.alpha_composite(sliver, (x, 0 if edge == "top" else height - visible))
        canvas.paste(layer.convert("RGB"))
