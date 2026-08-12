"""Where templates get their words.

Templates need "a title", "a paragraph", "a short cell value" — not a specific dataset
row. Putting that behind a small protocol lets the same template be filled from a CSV, a
Hugging Face corpus or a fixed list in a test, and keeps the templates readable.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

__all__ = ["CorpusTextProvider", "TextProvider"]


@runtime_checkable
class TextProvider(Protocol):
    """Supplies text fragments of roughly the requested shape."""

    def title(self, rng: random.Random) -> str: ...

    def sentence(self, rng: random.Random) -> str: ...

    def paragraph(self, rng: random.Random, sentences: int = 3) -> str: ...

    def phrase(self, rng: random.Random, max_words: int = 3) -> str: ...


class CorpusTextProvider:
    """Serves fragments cut from a corpus of real text.

    Real corpora are the point: synthetic *layout* over synthetic *language* teaches a
    model a vocabulary that does not exist. Sentences are cut on punctuation so fragments
    end where a reader would expect.
    """

    _SENTENCE_ENDS = ".!?؟۔،\n"

    def __init__(self, texts: Iterable[str], *, min_words: int = 3):
        self._sentences: list[str] = []
        for text in texts:
            self._sentences.extend(self._split(str(text), min_words))
        if not self._sentences:
            raise ValueError("CorpusTextProvider needs at least one usable sentence")

    @classmethod
    def _split(cls, text: str, min_words: int) -> list[str]:
        sentences: list[str] = []
        current: list[str] = []
        for char in text:
            current.append(char)
            if char in cls._SENTENCE_ENDS:
                sentence = "".join(current).strip()
                if len(sentence.split()) >= min_words:
                    sentences.append(sentence)
                current = []
        tail = "".join(current).strip()
        if len(tail.split()) >= min_words:
            sentences.append(tail)
        return sentences

    def __len__(self) -> int:
        return len(self._sentences)

    def sentence(self, rng: random.Random) -> str:
        return rng.choice(self._sentences)

    def paragraph(self, rng: random.Random, sentences: int = 3) -> str:
        count = max(1, sentences)
        return " ".join(self.sentence(rng) for _ in range(count))

    def title(self, rng: random.Random) -> str:
        words = self.sentence(rng).split()
        length = min(len(words), rng.randint(2, 7))
        return " ".join(words[:length]).rstrip(".،,;:")

    def phrase(self, rng: random.Random, max_words: int = 3) -> str:
        words = self.sentence(rng).split()
        length = min(len(words), max(1, rng.randint(1, max_words)))
        start = rng.randint(0, max(0, len(words) - length))
        return " ".join(words[start : start + length]).strip(".،,;:")

    @classmethod
    def from_sentences(cls, sentences: Sequence[str]) -> CorpusTextProvider:
        provider = cls.__new__(cls)
        provider._sentences = list(sentences)
        if not provider._sentences:
            raise ValueError("CorpusTextProvider needs at least one sentence")
        return provider
