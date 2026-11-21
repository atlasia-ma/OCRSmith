"""The input port: turning supplied text into the one internal document schema.

An adapter's only job is to recover *structure* from text that already exists — which
run of characters is a heading, which is a list, which is a table — and hand back a
`DocumentContent`. It decides nothing about appearance.

That distinction matters. Markdown, LaTeX and HTML are convenient ways to *collect* text
with its structure intact; they are not rendering targets. We never run a Markdown
renderer or a browser and photograph the result. `#` means "this is a heading" and
nothing more; how a heading looks on the page is the renderer's decision, and it should
look like a heading in a printed document rather than one on a web page.

Everything downstream — layout, degradation, annotation, validation — consumes
`DocumentContent` and cannot tell which adapter produced it.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..core.documents.content import DocumentContent
from ..text.script import Direction

__all__ = [
    "TextAdapter",
    "adapter_for",
    "adapter_names",
    "detect_format",
    "parse_document",
]


@runtime_checkable
class TextAdapter(Protocol):
    """Parses supplied text into the internal document schema."""

    name: str

    def parse(self, text: str, *, direction: Direction | None = None) -> DocumentContent: ...


#: Markers that only appear deliberately. Emphasis (`*`, `_`) and brackets are excluded:
#: they occur in ordinary prose often enough that sniffing on them misreads plain text.
_MARKDOWN_SIGNALS = (
    re.compile(r"^#{1,6}\s+\S", re.MULTILINE),  # ATX heading
    re.compile(r"^\s*```", re.MULTILINE),  # fenced code
    re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE),  # table row
    re.compile(r"^\s*>\s+\S", re.MULTILINE),  # block quote
    re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE),  # bullet list
    re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE),  # ordered list
)


def detect_format(text: str) -> str:
    """Guess which adapter suits `text`.

    A convenience only. Format should normally be stated outright: a plain-text document
    whose lists happen to be written with dashes is indistinguishable from Markdown, and
    guessing wrong turns body text into headings.
    """
    hits = sum(1 for pattern in _MARKDOWN_SIGNALS if pattern.search(text))
    return "markdown" if hits >= 2 else "plain"


def adapter_names() -> tuple[str, ...]:
    return ("plain", "markdown")


def adapter_for(name: str) -> TextAdapter:
    """The adapter registered under `name`; "auto" is resolved by `parse_document`."""
    from .markdown import MarkdownAdapter
    from .plain import PlainTextAdapter

    adapters = {"plain": PlainTextAdapter(), "markdown": MarkdownAdapter()}
    try:
        return adapters[name]
    except KeyError:
        raise ValueError(
            f"Unknown input format {name!r}. Available: auto, {', '.join(adapter_names())}"
        ) from None


def parse_document(
    text: str,
    *,
    fmt: str = "auto",
    direction: Direction | None = None,
) -> DocumentContent:
    """Parse `text` with the named adapter, sniffing the format when asked to."""
    resolved = detect_format(text) if fmt == "auto" else fmt
    return adapter_for(resolved).parse(text, direction=direction)
