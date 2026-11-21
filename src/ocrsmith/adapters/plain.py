"""Plain text: no markup to recover, so every paragraph is a paragraph."""

from __future__ import annotations

import re

from ..core.documents.content import DocumentBuilder, DocumentContent
from ..text.script import Direction

__all__ = ["PlainTextAdapter"]

_BLANK_LINE = re.compile(r"\n\s*\n")
#: A short opening line with no terminal punctuation reads as a title in almost every
#: plain-text document that has one. Longer than this and it is simply a first sentence.
_TITLE_MAX_WORDS = 12


class PlainTextAdapter:
    """Splits text into paragraphs on blank lines, promoting a leading short line."""

    name = "plain"

    def parse(self, text: str, *, direction: Direction | None = None) -> DocumentContent:
        builder = DocumentBuilder(direction, source_format="plain")
        chunks = [chunk.strip() for chunk in _BLANK_LINE.split(text) if chunk.strip()]
        if not chunks:
            return builder.build()

        first = " ".join(chunks[0].split())
        if self._reads_as_title(first):
            builder.title(first)
            chunks = chunks[1:]

        for chunk in chunks:
            # Single newlines inside a paragraph are wrapping in the source, not breaks.
            builder.paragraph(" ".join(chunk.split()))
        return builder.build()

    @staticmethod
    def _reads_as_title(line: str) -> bool:
        return bool(line) and len(line.split()) <= _TITLE_MAX_WORDS and line[-1] not in ".!?۔"
