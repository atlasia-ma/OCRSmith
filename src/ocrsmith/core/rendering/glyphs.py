"""A text engine that shapes with HarfBuzz and rasterises with FreeType.

Why this exists
---------------
Pillow can only draw a string of *characters*. For Arabic that is not enough, because the
shape of a letter depends on its neighbours, and the mapping from letters to shapes lives
in the font's `GSUB` table — which Pillow applies only when it was built with Raqm.

Without Raqm the fallback is `arabic-reshaper`: rewrite the text into Unicode's Arabic
presentation forms (U+FE70–FEFF), which are pre-joined characters a plain renderer can
draw one at a time. It works, and it is what this project shipped. But it quietly costs
most of the font pool, because **the presentation-form block is deprecated and modern
faces do not implement it**. They join through `GSUB` and carry no FExx glyphs at all. On
the bundled Arabic pool that is 32 of 101 faces unusable, and — this is the part that
matters — *every single handwriting and calligraphic face*, because those are all modern
OpenType designs. Aref Ruqaa, the standard ruqʿah hand, has 110 Arabic glyphs and zero
presentation forms. Arabic handwriting synthesis was therefore not merely poor, it was
impossible: the coverage gate correctly refused every hand in the pool.

This engine removes the constraint rather than working around it. HarfBuzz applies the
font's own `GSUB`/`GPOS` to the base letters, exactly as a browser or a word processor
does, and returns positioned glyph *indices*; FreeType rasterises those indices; the
bitmaps are composited by PIL. No presentation forms are involved anywhere, so a face only
needs the letters it claims to support.

What it buys beyond handwriting:

* **Every modern face becomes usable** — the whole pool rather than the traditional half.
* **Correct mark placement.** `GPOS` anchors a shadda or a kasra to the letter it belongs
  to. The presentation-form path has no positioning information and stacks marks on the
  advance width, which is why vocalised text drifted.
* **Real ligatures**, lam-alef included, instead of the handful encoded as single
  characters.

Shaping is per *word*, which is the project's existing contract and the thing that makes
word boxes exact: letters never join across a space, so shaping a word in isolation gives
the same glyphs it would get in the full line.

Availability
------------
Optional. `pip install "ocrsmith[shaping]"` brings `uharfbuzz` and `freetype-py`. When
they are absent everything falls back to the reshaper path and nothing breaks — it is the
font pool that shrinks, which is what `ocrsmith doctor` reports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops

__all__ = [
    "GlyphEngine",
    "GlyphRun",
    "PlacedGlyph",
    "engine_available",
    "glyph_engine",
]


@dataclass(frozen=True, slots=True)
class PlacedGlyph:
    """One glyph, by index, at the position HarfBuzz put it."""

    index: int
    x: float
    y: float
    advance: float


@dataclass(frozen=True, slots=True)
class GlyphRun:
    """A shaped word: where every glyph goes, and where the ink ends up.

    Positions are relative to the pen origin on the baseline, with y increasing downward
    to match image coordinates (HarfBuzz and FreeType both use y-up).
    """

    glyphs: tuple[PlacedGlyph, ...]
    advance: float
    ink_x0: float
    ink_y0: float
    ink_x1: float
    ink_y1: float

    @property
    def is_blank(self) -> bool:
        return self.ink_x1 <= self.ink_x0 or self.ink_y1 <= self.ink_y0


def engine_available() -> bool:
    """Whether HarfBuzz shaping and FreeType rasterising can be used in this process."""
    if os.environ.get("OCRSMITH_DISABLE_HARFBUZZ"):
        return False
    try:
        import freetype  # noqa: F401
        import uharfbuzz  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=128)
def _hb_font(path: str, size: int):
    import uharfbuzz as hb

    face = hb.Face(hb.Blob.from_file_path(path))
    font = hb.Font(face)
    # HarfBuzz works in 26.6 fixed point, like FreeType, so a pixel is 64 units.
    font.scale = (size * 64, size * 64)
    hb.ot_font_set_funcs(font)
    return font


@lru_cache(maxsize=128)
def _ft_face(path: str, size: int):
    import freetype

    face = freetype.Face(path)
    face.set_char_size(size * 64)
    return face


class GlyphEngine:
    """Shapes and draws one font at one size.

    Instances are cached per `(path, size)` by `glyph_engine`, because both the HarfBuzz
    font and the FreeType face are expensive to build and a run reuses a handful of them
    for millions of words.
    """

    #: OpenType features to apply. Kerning and standard ligatures are what a text renderer
    #: turns on by default; required ligatures (`rlig`, lam-alef and friends) are on
    #: unconditionally in HarfBuzz and are not listed.
    FEATURES = {"kern": True, "liga": True}

    def __init__(self, path: str | Path, size: int):
        self.path = str(path)
        self.size = int(size)
        self._shaped: dict[str, GlyphRun] = {}
        self._bitmaps: dict[int, tuple[Image.Image, int, int] | None] = {}

    # -- metrics -----------------------------------------------------------

    @property
    def ascent(self) -> float:
        face = _ft_face(self.path, self.size)
        return face.size.ascender / 64.0

    @property
    def descent(self) -> float:
        # FreeType reports the descender as a negative number; the rest of the codebase
        # treats descent as a positive distance below the baseline.
        face = _ft_face(self.path, self.size)
        return abs(face.size.descender / 64.0)

    # -- shaping -----------------------------------------------------------

    def shape(self, text: str) -> GlyphRun:
        """Shape `text` as one run, measuring the ink it will actually put down.

        Cached per string: a page asks for the same words over and over, and shaping is
        the expensive half.
        """
        cached = self._shaped.get(text)
        if cached is not None:
            return cached

        import uharfbuzz as hb

        buffer = hb.Buffer()
        buffer.add_str(text)
        # Direction, script and language are inferred from the characters rather than
        # passed in. The caller already knows the paragraph direction, but a word is a
        # run of one script and HarfBuzz's guess for it is the correct one — passing a
        # paragraph's RTL down to a Latin word inside it would reverse the word.
        buffer.guess_segment_properties()
        hb.shape(_hb_font(self.path, self.size), buffer, self.FEATURES)

        glyphs: list[PlacedGlyph] = []
        pen = 0.0
        x0 = y0 = float("inf")
        x1 = y1 = float("-inf")
        for info, position in zip(buffer.glyph_infos, buffer.glyph_positions, strict=True):
            x = pen + position.x_offset / 64.0
            # y_offset is up-positive in HarfBuzz and the canvas is down-positive.
            y = -position.y_offset / 64.0
            glyphs.append(PlacedGlyph(info.codepoint, x, y, position.x_advance / 64.0))
            bitmap = self._bitmap(info.codepoint)
            if bitmap is not None:
                image, left, top = bitmap
                x0 = min(x0, x + left)
                y0 = min(y0, y - top)
                x1 = max(x1, x + left + image.width)
                y1 = max(y1, y - top + image.height)
            pen += position.x_advance / 64.0

        if x0 == float("inf"):  # whitespace, or a run with no ink
            run = GlyphRun(tuple(glyphs), pen, 0.0, 0.0, 0.0, 0.0)
        else:
            run = GlyphRun(tuple(glyphs), pen, x0, y0, x1, y1)
        self._shaped[text] = run
        return run

    # -- drawing -----------------------------------------------------------

    def _bitmap(self, index: int) -> tuple[Image.Image, int, int] | None:
        """The rendered glyph as an alpha mask, with its bearings. None when blank."""
        if index in self._bitmaps:
            return self._bitmaps[index]

        import freetype

        face = _ft_face(self.path, self.size)
        face.load_glyph(index, freetype.FT_LOAD_RENDER)
        slot = face.glyph
        bitmap = slot.bitmap
        result: tuple[Image.Image, int, int] | None = None
        if bitmap.width and bitmap.rows:
            # FreeType's buffer is 8-bit coverage, which is exactly an alpha channel.
            mask = Image.frombytes("L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer))
            result = (mask, slot.bitmap_left, slot.bitmap_top)
        self._bitmaps[index] = result
        return result

    def draw(
        self,
        canvas: Image.Image,
        text: str,
        x: float,
        baseline: float,
        fill: tuple[int, int, int, int],
    ) -> GlyphRun:
        """Draw `text` with its pen origin at `(x, baseline)`. Returns the shaped run.

        `canvas` must be RGBA; each glyph is composited as a solid colour through its
        coverage mask, which is how an antialiased glyph keeps its edges over whatever is
        already on the page.
        """
        run = self.shape(text)
        if run.is_blank:
            return run

        # The glyphs are gathered into one coverage mask for the whole word and
        # composited once, rather than one alpha_composite per glyph. A word is five to
        # eight glyphs and a page is thousands of words, so the per-glyph version spent
        # most of its time allocating small images; this is the same pixels in one pass.
        pad = 2
        width = int(run.ink_x1 - run.ink_x0) + 2 * pad
        height = int(run.ink_y1 - run.ink_y0) + 2 * pad
        if width <= 0 or height <= 0:
            return run
        mask = Image.new("L", (width, height), 0)
        for glyph in run.glyphs:
            bitmap = self._bitmap(glyph.index)
            if bitmap is None:
                continue
            coverage, left, top = bitmap
            box = (
                int(round(glyph.x + left - run.ink_x0)) + pad,
                int(round(glyph.y - top - run.ink_y0)) + pad,
            )
            # Lighter, not paste: two glyphs can overlap — a mark over its letter, or a
            # cursive join — and overwriting would punch a transparent hole in the first.
            region = mask.crop((box[0], box[1], box[0] + coverage.width, box[1] + coverage.height))
            mask.paste(ImageChops.lighter(region, coverage), box)

        patch = Image.new("RGBA", (width, height), fill)
        patch.putalpha(mask)
        canvas.alpha_composite(
            patch,
            (int(round(x + run.ink_x0)) - pad, int(round(baseline + run.ink_y0)) - pad),
        )
        return run


#: Engines are cached per (font file, size). Building one parses the font twice — once
#: for HarfBuzz and once for FreeType — and a run reuses a handful across millions of
#: words, so this is the difference between a usable backend and an unusable one.
_ENGINES: dict[tuple[str, int], GlyphEngine] = {}
_MAX_ENGINES = 192


def glyph_engine(path: str | Path, size: int) -> GlyphEngine:
    """The shared engine for a font at a size."""
    key = (str(path), int(size))
    engine = _ENGINES.get(key)
    if engine is None:
        if len(_ENGINES) >= _MAX_ENGINES:
            _ENGINES.clear()
        engine = GlyphEngine(path, size)
        _ENGINES[key] = engine
    return engine
