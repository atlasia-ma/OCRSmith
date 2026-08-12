"""Line breaking.

Greedy first-fit wrapping measured in pixels, with two properties the previous
implementation lacked and that matter for label integrity:

* **no silent loss** — every word of the input appears in the output. A wrapper that
  drops the tail of a paragraph when it runs out of vertical room produces an image whose
  label claims text that was never drawn.
* **oversized words are broken, not overflowed** — a single token wider than the column
  is split at character boundaries instead of running off the page.

Deciding *how much* text to draw is the caller's job (see `fit_lines`), and when text is
dropped the caller is told, so the label can be trimmed to match the pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

__all__ = ["break_long_word", "fit_lines", "wrap_paragraph", "wrap_text"]

Measure = Callable[[str], float]


def break_long_word(word: str, measure: Measure, max_width: float) -> list[str]:
    """Split a word too wide for the column into the longest fitting pieces."""
    if max_width <= 0 or measure(word) <= max_width:
        return [word]

    pieces: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if current and measure(candidate) > max_width:
            pieces.append(current)
            current = char
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def wrap_paragraph(text: str, measure: Measure, max_width: float | None) -> list[str]:
    """Break one paragraph into lines no wider than `max_width` pixels.

    `max_width` of None (or non-positive) means "do not wrap": the paragraph becomes a
    single line, which is what single-line recognition crops want.
    """
    words = text.split()
    if not words:
        return []
    if not max_width or max_width <= 0:
        return [" ".join(words)]

    lines: list[str] = []
    current: list[str] = []
    for word in words:
        for piece in break_long_word(word, measure, max_width):
            candidate = " ".join([*current, piece])
            if current and measure(candidate) > max_width:
                lines.append(" ".join(current))
                current = [piece]
            else:
                current.append(piece)
    if current:
        lines.append(" ".join(current))
    return lines


def wrap_text(
    paragraphs: Iterable[str],
    measure: Measure,
    max_width: float | None,
) -> Iterator[str]:
    """Wrap several paragraphs, yielding an empty string between them.

    The blank entries survive as paragraph separators so a caller can render vertical
    spacing without losing where one paragraph ended and the next began.
    """
    first = True
    for paragraph in paragraphs:
        if not first:
            yield ""
        first = False
        yield from wrap_paragraph(paragraph, measure, max_width)


def fit_lines(
    lines: Sequence[str],
    line_height: float,
    max_height: float | None,
) -> tuple[list[str], int]:
    """Keep as many leading lines as fit in `max_height`.

    Returns the kept lines and how many were dropped, so the caller can shorten the label
    to exactly what was drawn instead of asserting text that is not in the image.
    """
    if not max_height or max_height <= 0 or line_height <= 0:
        return list(lines), 0

    capacity = max(1, int(max_height // line_height))
    if len(lines) <= capacity:
        return list(lines), 0
    return list(lines[:capacity]), len(lines) - capacity
