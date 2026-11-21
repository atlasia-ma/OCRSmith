"""Input adapters: supplied text in, one internal document schema out."""

from .base import TextAdapter, adapter_for, adapter_names, detect_format, parse_document
from .chunking import chunk_content, iter_chunks, sample_chunk
from .markdown import MarkdownAdapter, strip_inline
from .plain import PlainTextAdapter

__all__ = [
    "MarkdownAdapter",
    "PlainTextAdapter",
    "TextAdapter",
    "adapter_for",
    "adapter_names",
    "chunk_content",
    "detect_format",
    "iter_chunks",
    "parse_document",
    "sample_chunk",
    "strip_inline",
]
