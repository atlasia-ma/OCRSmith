"""Splitting a parsed document into surface-sized pieces.

A page holds what a page holds. Feed a book to a generator that draws one surface per
sample and the rest of the book is thrown away — so the caller chunks first, and every
chunk becomes its own sample. Only the caller knows whether the input is a book or a
receipt, which is why chunking lives here rather than inside an adapter.

Chunks are cut on structure, never mid-block: a heading opens the chunk that contains
its section rather than being stranded at the foot of the previous one.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from ..core.documents.content import ContentBlock, DocumentContent
from ..domain import RegionType

__all__ = ["chunk_content", "iter_chunks"]

#: Blocks that open a section. A chunk would rather start here than end here.
_OPENERS = (RegionType.TITLE, RegionType.HEADING)


def _weight(block: ContentBlock) -> int:
    """Roughly how much surface a block will claim, in characters."""
    size = len(block.text)
    size += sum(len(item) for item in block.items)
    if block.table is not None:
        size += sum(len(cell.text) for cell in block.table.cells)
        # A table costs more room than its characters suggest: rules, padding, and a row
        # of height regardless of how short its cells are.
        size += 40 * max(1, block.table.rows)
    return size


def chunk_content(
    content: DocumentContent,
    *,
    max_chars: int = 1600,
    max_blocks: int = 12,
) -> list[DocumentContent]:
    """Split `content` into pieces no larger than the given budget."""
    return list(iter_chunks(content, max_chars=max_chars, max_blocks=max_blocks))


def iter_chunks(
    content: DocumentContent,
    *,
    max_chars: int = 1600,
    max_blocks: int = 12,
) -> Iterator[DocumentContent]:
    """Stream chunks, so a large source document never sits in memory twice."""
    if not content.blocks:
        return

    current: list[ContentBlock] = []
    size = 0

    def emit() -> Iterator[DocumentContent]:
        if current:
            yield DocumentContent(tuple(current), content.direction, dict(content.metadata))

    for block in content.blocks:
        cost = _weight(block)
        opens_section = block.type in _OPENERS
        full = current and (size + cost > max_chars or len(current) >= max_blocks)
        # Break *before* a heading when the chunk already has substance, so sections stay
        # with their headings instead of being orphaned across the cut.
        if full or (opens_section and len(current) > 1 and size > max_chars // 3):
            yield from emit()
            current, size = [], 0
        current.append(block)
        size += cost

    yield from emit()


def sample_chunk(
    content: DocumentContent,
    rng: random.Random,
    *,
    max_chars: int = 1600,
    max_blocks: int = 12,
) -> DocumentContent:
    """One chunk of `content`, chosen at random.

    Taking a random chunk rather than always the first means a long source document
    contributes its middle and its end to the corpus too, instead of the same opening
    page every time it is drawn.
    """
    chunks = chunk_content(content, max_chars=max_chars, max_blocks=max_blocks)
    if not chunks:
        return content
    return rng.choice(chunks)
