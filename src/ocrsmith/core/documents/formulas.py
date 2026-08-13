"""Mathematical formulas, typeset from a tree that also emits their LaTeX.

Formula conversion is the single largest gain category in the document-parsing benchmarks,
and OCRSmith previously rendered a `FORMULA` region as plain text — which taught a model
nothing about fractions, radicals or limits.

The design mirrors the chart module: an expression is a **tree**, and both the drawing and
the LaTeX are derived from it. Rendering LaTeX source with a parser would invert that and
let the two drift; here they cannot, because neither is the input.

This is a small typesetter, not TeX. It covers the constructs that actually appear in
document OCR — fractions, powers, indices, radicals, sums and integrals with limits — and
does so without a LaTeX toolchain, which would be an unreasonable dependency for a data
generator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont

from ...domain.geometry import BBox
from ..fonts import load_font

__all__ = [
    "MATH_GLYPHS",
    "BigOperator",
    "Fraction",
    "Node",
    "RenderedFormula",
    "Row",
    "Sqrt",
    "Subscript",
    "Superscript",
    "Symbol",
    "FormulaRenderer",
    "choose_math_font",
    "sample_formula",
]

#: Operators that carry limits above and below, with the glyph used to draw them.
_BIG = {"sum": ("∑", r"\sum"), "prod": ("∏", r"\prod"), "int": ("∫", r"\int")}

#: Glyphs a face must carry to typeset mathematics. Checked up front for the same reason
#: text coverage is: a font without these draws tofu while the LaTeX claims a summation.
MATH_GLYPHS = "∑∏∫√±×÷≤≥≠∞0123456789=+-()"


def choose_math_font(candidates, fallback: str | None = None) -> str:
    """Pick a face that can actually draw mathematics.

    Most text faces have no summation or integral sign. Choosing blindly produces a
    formula of empty boxes whose LaTeX confidently asserts a sum - the same silent
    corruption that glyph coverage exists to prevent, in a corner that is easy to forget.
    """
    from ...text.coverage import supports_text

    best, best_ratio = fallback, -1.0
    for candidate in candidates:
        try:
            ratio = supports_text(candidate, MATH_GLYPHS).ratio
        except Exception:
            continue
        if ratio > best_ratio:
            best, best_ratio = str(candidate), ratio
        if ratio >= 1.0:
            break
    if best is None:
        raise ValueError("No font available that can draw mathematics")
    return best


@runtime_checkable
class Node(Protocol):
    """A piece of an expression that can size itself, draw itself and name itself."""

    def latex(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Symbol:
    """A literal run of characters: a variable, a number, an operator."""

    text: str

    def latex(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Row:
    """A horizontal sequence."""

    items: tuple[Node, ...]

    def latex(self) -> str:
        return " ".join(item.latex() for item in self.items)


@dataclass(frozen=True, slots=True)
class Fraction:
    numerator: Node
    denominator: Node

    def latex(self) -> str:
        return rf"\frac{{{self.numerator.latex()}}}{{{self.denominator.latex()}}}"


@dataclass(frozen=True, slots=True)
class Superscript:
    base: Node
    exponent: Node

    def latex(self) -> str:
        return f"{self.base.latex()}^{{{self.exponent.latex()}}}"


@dataclass(frozen=True, slots=True)
class Subscript:
    base: Node
    index: Node

    def latex(self) -> str:
        return f"{self.base.latex()}_{{{self.index.latex()}}}"


@dataclass(frozen=True, slots=True)
class Sqrt:
    radicand: Node

    def latex(self) -> str:
        return rf"\sqrt{{{self.radicand.latex()}}}"


@dataclass(frozen=True, slots=True)
class BigOperator:
    """A sum, product or integral with optional limits."""

    kind: str
    body: Node
    lower: Node | None = None
    upper: Node | None = None

    def latex(self) -> str:
        command = _BIG[self.kind][1]
        limits = ""
        if self.lower is not None:
            limits += f"_{{{self.lower.latex()}}}"
        if self.upper is not None:
            limits += f"^{{{self.upper.latex()}}}"
        return f"{command}{limits} {self.body.latex()}"


@dataclass(frozen=True, slots=True)
class RenderedFormula:
    """A drawn formula and the LaTeX that describes it."""

    image: Image.Image
    latex: str

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


@dataclass(frozen=True, slots=True)
class _Box:
    """Measured extent of a node: width, plus height above and below the baseline."""

    width: float
    above: float
    below: float

    @property
    def height(self) -> float:
        return self.above + self.below


class FormulaRenderer:
    """Typesets an expression tree.

    Everything is positioned relative to a *baseline* rather than a bounding box, which is
    what makes `a^2 + \\frac{b}{c}` line up the way a reader expects.
    """

    def __init__(self, font_path: str, size: int = 28, ink: tuple[int, int, int] = (20, 20, 24)):
        self.font_path = str(font_path)
        self.size = size
        self.ink = ink

    def _font(self, size: float) -> FreeTypeFont:
        return load_font(self.font_path, max(7, int(round(size))))

    # -- measurement -------------------------------------------------------

    def measure(self, node: Node, size: float) -> _Box:
        font = self._font(size)
        ascent, descent = font.getmetrics()

        if isinstance(node, Symbol):
            return _Box(font.getlength(node.text), ascent * 0.78, descent * 0.9)

        if isinstance(node, Row):
            parts = [self.measure(item, size) for item in node.items]
            gap = size * 0.12 * max(0, len(parts) - 1)
            return _Box(
                sum(part.width for part in parts) + gap,
                max((part.above for part in parts), default=0.0),
                max((part.below for part in parts), default=0.0),
            )

        if isinstance(node, Fraction):
            top = self.measure(node.numerator, size * 0.92)
            bottom = self.measure(node.denominator, size * 0.92)
            return _Box(
                max(top.width, bottom.width) + size * 0.3,
                top.height + size * 0.25,
                bottom.height + size * 0.15,
            )

        if isinstance(node, (Superscript, Subscript)):
            base = self.measure(node.base, size)
            script_node = node.exponent if isinstance(node, Superscript) else node.index
            script = self.measure(script_node, size * 0.62)
            if isinstance(node, Superscript):
                return _Box(
                    base.width + script.width, max(base.above, base.above + script.height * 0.45), base.below
                )
            return _Box(
                base.width + script.width, base.above, max(base.below, base.below + script.height * 0.35)
            )

        if isinstance(node, Sqrt):
            inner = self.measure(node.radicand, size)
            return _Box(inner.width + size * 0.75, inner.above + size * 0.22, inner.below)

        if isinstance(node, BigOperator):
            glyph = self.measure(Symbol(_BIG[node.kind][0]), size * 1.7)
            body = self.measure(node.body, size)
            lower = self.measure(node.lower, size * 0.6) if node.lower else _Box(0, 0, 0)
            upper = self.measure(node.upper, size * 0.6) if node.upper else _Box(0, 0, 0)
            width = max(glyph.width, lower.width, upper.width) + body.width + size * 0.3
            return _Box(width, glyph.above + upper.height, glyph.below + lower.height)

        return _Box(0.0, 0.0, 0.0)

    # -- drawing -----------------------------------------------------------

    def draw(self, draw: ImageDraw.ImageDraw, node: Node, x: float, baseline: float, size: float) -> float:
        """Draw `node` with its baseline at `baseline`; return the x after it."""
        font = self._font(size)
        ascent, _ = font.getmetrics()
        fill = (*self.ink, 255)

        if isinstance(node, Symbol):
            draw.text((x, baseline - ascent), node.text, font=font, fill=fill)
            return x + font.getlength(node.text)

        if isinstance(node, Row):
            cursor = x
            for index, item in enumerate(node.items):
                if index:
                    cursor += size * 0.12
                cursor = self.draw(draw, item, cursor, baseline, size)
            return cursor

        if isinstance(node, Fraction):
            inner = size * 0.92
            top = self.measure(node.numerator, inner)
            bottom = self.measure(node.denominator, inner)
            width = max(top.width, bottom.width)
            rule_y = baseline - size * 0.28
            self.draw(
                draw,
                node.numerator,
                x + (width - top.width) / 2 + size * 0.15,
                rule_y - size * 0.18 - top.below,
                inner,
            )
            self.draw(
                draw,
                node.denominator,
                x + (width - bottom.width) / 2 + size * 0.15,
                rule_y + size * 0.2 + bottom.above,
                inner,
            )
            draw.line((x, rule_y, x + width + size * 0.3, rule_y), fill=fill, width=max(1, int(size * 0.06)))
            return x + width + size * 0.3

        if isinstance(node, Superscript):
            after = self.draw(draw, node.base, x, baseline, size)
            base = self.measure(node.base, size)
            self.draw(draw, node.exponent, after, baseline - base.above * 0.62, size * 0.62)
            return after + self.measure(node.exponent, size * 0.62).width

        if isinstance(node, Subscript):
            after = self.draw(draw, node.base, x, baseline, size)
            self.draw(draw, node.index, after, baseline + size * 0.22, size * 0.62)
            return after + self.measure(node.index, size * 0.62).width

        if isinstance(node, Sqrt):
            inner = self.measure(node.radicand, size)
            hook = size * 0.55
            top = baseline - inner.above - size * 0.16
            draw.line(
                (x, baseline - inner.above * 0.42, x + hook * 0.4, baseline + inner.below * 0.5),
                fill=fill,
                width=max(1, int(size * 0.06)),
            )
            draw.line(
                (x + hook * 0.4, baseline + inner.below * 0.5, x + hook * 0.62, top),
                fill=fill,
                width=max(1, int(size * 0.06)),
            )
            draw.line(
                (x + hook * 0.62, top, x + hook + inner.width + size * 0.1, top),
                fill=fill,
                width=max(1, int(size * 0.06)),
            )
            self.draw(draw, node.radicand, x + hook + size * 0.05, baseline, size)
            return x + hook + inner.width + size * 0.15

        if isinstance(node, BigOperator):
            glyph_size = size * 1.7
            glyph = Symbol(_BIG[node.kind][0])
            glyph_box = self.measure(glyph, glyph_size)
            column = max(
                glyph_box.width,
                self.measure(node.lower, size * 0.6).width if node.lower else 0,
                self.measure(node.upper, size * 0.6).width if node.upper else 0,
            )
            self.draw(
                draw, glyph, x + (column - glyph_box.width) / 2, baseline + glyph_box.below * 0.35, glyph_size
            )
            if node.upper is not None:
                upper = self.measure(node.upper, size * 0.6)
                self.draw(
                    draw,
                    node.upper,
                    x + (column - upper.width) / 2,
                    baseline - glyph_box.above * 0.8,
                    size * 0.6,
                )
            if node.lower is not None:
                lower = self.measure(node.lower, size * 0.6)
                self.draw(
                    draw,
                    node.lower,
                    x + (column - lower.width) / 2,
                    baseline + glyph_box.below + lower.above * 0.9,
                    size * 0.6,
                )
            return self.draw(draw, node.body, x + column + size * 0.2, baseline, size)

        return x

    def render(self, node: Node, *, padding: int = 8) -> RenderedFormula:
        box = self.measure(node, self.size)
        width = int(box.width + padding * 2) + 2
        height = int(box.height + padding * 2) + 2
        image = Image.new("RGBA", (max(4, width), max(4, height)), (0, 0, 0, 0))
        self.draw(ImageDraw.Draw(image), node, padding, padding + box.above, self.size)
        return RenderedFormula(image, node.latex())

    def bbox(self, node: Node, padding: int = 8) -> BBox:
        box = self.measure(node, self.size)
        return BBox(0, 0, box.width + padding * 2, box.height + padding * 2)


def sample_formula(rng: random.Random, *, depth: int = 0) -> Node:
    """Build a plausible expression.

    Weighted towards the constructs that actually appear in documents rather than towards
    exotic ones: most real formulas are a fraction, a power or a sum.
    """
    letters = "abcxyznmktpqr"
    if depth >= 2 or rng.random() < 0.35:
        if rng.random() < 0.45:
            return Symbol(str(rng.randint(1, 99)))
        return Symbol(rng.choice(letters))

    choice = rng.choices(
        ["frac", "sup", "sub", "sqrt", "big", "row"],
        weights=[3.0, 3.0, 1.5, 1.5, 1.5, 2.5],
        k=1,
    )[0]
    child = lambda: sample_formula(rng, depth=depth + 1)  # noqa: E731 - local shorthand

    if choice == "frac":
        return Fraction(child(), child())
    if choice == "sup":
        return Superscript(Symbol(rng.choice(letters)), Symbol(str(rng.randint(2, 9))))
    if choice == "sub":
        return Subscript(Symbol(rng.choice(letters)), Symbol(str(rng.randint(0, 9))))
    if choice == "sqrt":
        return Sqrt(child())
    if choice == "big":
        kind = rng.choice(list(_BIG))
        return BigOperator(
            kind,
            child(),
            lower=Symbol(f"{rng.choice('ijk')}=1"),
            upper=Symbol(rng.choice(["n", "m", "∞"])),
        )
    operator = rng.choice(["+", "-", "=", "×"])
    return Row((child(), Symbol(operator), child()))
