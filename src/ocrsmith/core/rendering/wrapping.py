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


def _is_connected_script(word: str) -> bool:
    """Whether the word is written in a script whose letters join.

    Arabic, Syriac, Nko and the Arabic-derived blocks. Latin, digits and CJK all break
    at character boundaries without changing what the reader sees; these do not.
    """
    return any(
        0x0600 <= ord(ch) <= 0x08FF or 0xFB50 <= ord(ch) <= 0xFEFF for ch in word
    )


def break_long_word(word: str, measure: Measure, max_width: float) -> list[str]:
    """Split a word too wide for the column into the longest fitting pieces.

    A connected script is never broken. Cutting an Arabic word in half does not merely
    hyphenate it: the letters either side of the cut take their isolated and final forms,
    so the glyphs drawn stop being the glyphs that spell the word the label claims. The
    word is returned whole instead, over-wide, and the caller declines to place the block
    - which is the honest outcome, because a column too narrow for a single word is a
    broken page shape rather than a paragraph needing cleverer breaking.
    """
    if max_width <= 0 or measure(word) <= max_width:
        return [word]
    if _is_connected_script(word):
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
    """Wrap several source lines into one flat sequence of drawn lines.

    No blank separators are emitted. A newline in the input means "start a new line", not
    "leave a gap": vertical space between blocks belongs to the layout engine, which knows
    the surrounding context. Emitting blanks here would also make a line count returned to
    a caller disagree with the number of lines actually drawn.
    """
    for paragraph in paragraphs:
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
