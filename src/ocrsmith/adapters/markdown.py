"""Markdown: recover structure, discard presentation.

Only block-level structure is kept — what each run of text *is*. Inline markers are
stripped rather than styled, because the label is plain text: if `**مهم**` reached the
page as literal asterisks, the annotation would claim characters no writer put in the
document. The page shows emphasis the way print does, or not at all.
"""

from __future__ import annotations

import re

from ..core.documents.content import DocumentBuilder, DocumentContent
from ..text.script import Direction

__all__ = ["MarkdownAdapter", "strip_inline"]

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\([^)]*\)\s*$")

_INLINE = (
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # image -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # link -> label
    (re.compile(r"`{1,3}([^`]+)`{1,3}"), r"\1"),  # inline code
    (re.compile(r"\*\*\*([^*]+)\*\*\*"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)"), r"\1"),
    (re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)"), r"\1"),
    (re.compile(r"~~([^~]+)~~"), r"\1"),
)


def strip_inline(text: str) -> str:
    """Remove inline markers, keeping the words they wrapped."""
    for pattern, replacement in _INLINE:
        text = pattern.sub(replacement, text)
    return " ".join(text.replace("\\", "").split())


class MarkdownAdapter:
    """Maps Markdown block structure onto the internal document schema."""

    name = "markdown"

    def parse(self, text: str, *, direction: Direction | None = None) -> DocumentContent:
        builder = DocumentBuilder(direction, source_format="markdown")
        lines = text.replace("\r\n", "\n").split("\n")
        index = 0
        paragraph: list[str] = []
        has_title = False

        def flush() -> None:
            nonlocal paragraph
            if paragraph:
                body = strip_inline(" ".join(paragraph))
                if body:
                    builder.paragraph(body)
                paragraph = []

        while index < len(lines):
            line = lines[index]

            if not line.strip():
                flush()
                index += 1
                continue

            if _FENCE.match(line):
                flush()
                index += 1
                block: list[str] = []
                while index < len(lines) and not _FENCE.match(lines[index]):
                    block.append(lines[index])
                    index += 1
                index += 1  # closing fence
                body = "\n".join(block).strip()
                if body:
                    builder.code(body)
                continue

            if _RULE.match(line):
                flush()
                builder.separator()
                index += 1
                continue

            match = _IMAGE.match(line)
            if match:
                flush()
                caption = strip_inline(match.group(1))
                builder.figure(width=420, height=280, caption=caption or None)
                index += 1
                continue

            match = _HEADING.match(line)
            if match:
                flush()
                level = len(match.group(1))
                body = strip_inline(match.group(2))
                if body:
                    # One title per document: the first level-1 heading. Later ones are
                    # sections, because a page with two titles is not a document shape
                    # that occurs.
                    if level == 1 and not has_title:
                        builder.title(body)
                        has_title = True
                    else:
                        builder.heading(body, level=max(2, level))
                index += 1
                continue

            if _TABLE_ROW.match(line):
                flush()
                index = self._consume_table(builder, lines, index)
                continue

            if _QUOTE.match(line):
                flush()
                quoted: list[str] = []
                while index < len(lines) and (m := _QUOTE.match(lines[index])):
                    quoted.append(m.group(1).strip())
                    index += 1
                body = strip_inline(" ".join(part for part in quoted if part))
                if body:
                    builder.quote(body)
                continue

            if _BULLET.match(line) or _ORDERED.match(line):
                flush()
                index = self._consume_list(builder, lines, index)
                continue

            paragraph.append(line.strip())
            index += 1

        flush()
        return builder.build()

    @staticmethod
    def _consume_list(builder: DocumentBuilder, lines: list[str], index: int) -> int:
        items: list[str] = []
        ordered = bool(_ORDERED.match(lines[index]))
        while index < len(lines):
            match = _ORDERED.match(lines[index]) if ordered else _BULLET.match(lines[index])
            if not match:
                break
            item = strip_inline(match.group(1))
            if item:
                items.append(item)
            index += 1
        if items:
            builder.list(items, ordered=ordered)
        return index

    @staticmethod
    def _consume_table(builder: DocumentBuilder, lines: list[str], index: int) -> int:
        rows: list[list[str]] = []
        has_header = False
        while index < len(lines):
            line = lines[index]
            if _TABLE_DIVIDER.match(line) and rows:
                has_header = True
                index += 1
                continue
            match = _TABLE_ROW.match(line)
            if not match:
                break
            cells = [strip_inline(cell) for cell in match.group(1).split("|")]
            rows.append(cells)
            index += 1
        # A one-row "table" is a stray pipe in prose, not a table.
        if len(rows) >= 2:
            width = max(len(row) for row in rows)
            padded = [row + [""] * (width - len(row)) for row in rows]
            builder.table(padded, has_header_row=has_header)
        return index
