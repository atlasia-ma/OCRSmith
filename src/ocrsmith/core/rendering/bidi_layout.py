"""Word-level bidirectional layout.

Bounding boxes are only trustworthy if we know where each *word* physically lands, and in
mixed Arabic/Latin text the physical order is not the logical order. Rather than draw a
whole line as one string and then guess at word positions, the renderer places words
individually — which is typographically safe for Arabic, since letters never join across
a space — and this module decides the order they are placed in.

The rule implemented here is the word-level reduction of the Unicode bidi algorithm:
group words into directional runs, order the runs by the paragraph direction, and reverse
word order inside right-to-left runs. That handles the cases synthetic documents actually
contain ("سنة 2024 OCR", a Latin acronym inside an Arabic sentence) without pulling in a
full bidi implementation for text we already control.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...text.script import Direction, strong_direction

__all__ = ["DirectionalRun", "visual_word_order"]


@dataclass(frozen=True, slots=True)
class DirectionalRun:
    """A maximal stretch of words sharing one direction."""

    direction: Direction
    #: Indices into the logical word sequence.
    indices: tuple[int, ...]


def _runs(words: Sequence[str], base: Direction) -> list[DirectionalRun]:
    """Split words into directional runs, neutrals joining whatever precedes them."""
    runs: list[DirectionalRun] = []
    current: list[int] = []
    current_direction = base

    for index, word in enumerate(words):
        direction = strong_direction(word) or (current_direction if current else base)
        if current and direction is current_direction:
            current.append(index)
            continue
        if current:
            runs.append(DirectionalRun(current_direction, tuple(current)))
        current = [index]
        current_direction = direction

    if current:
        runs.append(DirectionalRun(current_direction, tuple(current)))
    return runs


def visual_word_order(words: Sequence[str], base: Direction) -> tuple[int, ...]:
    """Indices of `words` in the left-to-right order they should be drawn.

    For a purely left-to-right line this is the identity; for a purely right-to-left line
    it is the reverse; mixed lines keep embedded runs internally ordered while the runs
    themselves follow the paragraph direction.
    """
    if not words:
        return ()

    runs = _runs(words, base)
    if base.is_rtl:
        runs = list(reversed(runs))

    order: list[int] = []
    for run in runs:
        indices = run.indices
        if run.direction.is_rtl:
            indices = tuple(reversed(indices))
        order.extend(indices)
    return tuple(order)
