"""Where templates get their words.

Templates need "a title", "a paragraph", "a short cell value" — not a specific dataset
row. Putting that behind a small protocol lets the same template be filled from a CSV, a
Hugging Face corpus or a fixed list in a test, and keeps the templates readable.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

__all__ = ["CorpusTextProvider", "FieldGenerator", "TextProvider"]


class FieldGenerator:
    """Short strings that are not prose: dates, codes, amounts, references.

    Real documents are full of these and a sentence corpus contains none of them. Engines
    that added contextless and non-lexical strings report gains precisely on the cases
    where a model has no language model to lean on - a reference number is only readable
    glyph by glyph, which is the hardest thing a recogniser does.

    Numerals are emitted in whichever system the caller asks for, since Arabic documents
    mix Western and Arabic-Indic digits freely and a corpus of only one teaches the wrong
    prior.
    """

    _MONTHS_AR = (
        "يناير",
        "فبراير",
        "مارس",
        "أبريل",
        "ماي",
        "يونيو",
        "يوليوز",
        "غشت",
        "شتنبر",
        "أكتوبر",
        "نونبر",
        "دجنبر",
    )
    _CURRENCIES = ("MAD", "درهم", "DH", "€", "$")

    def __init__(self, numerals: str = "keep"):
        self.numerals = numerals

    def _digits(self, text: str) -> str:
        from ...text.normalization import NumeralSystem, to_numeral_system

        if self.numerals in ("keep", ""):
            return text
        return to_numeral_system(text, NumeralSystem(self.numerals))

    def date(self, rng: random.Random) -> str:
        day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2026)
        style = rng.random()
        if style < 0.4:
            return self._digits(f"{day:02d}/{month:02d}/{year}")
        if style < 0.7:
            return self._digits(f"{day} {self._MONTHS_AR[month - 1]} {year}")
        return self._digits(f"{year}-{month:02d}-{day:02d}")

    def amount(self, rng: random.Random, currency: str | None = None) -> str:
        """One money amount. `currency` fixes the unit for a whole document.

        A single invoice quoting one line in dirhams, the next in euros and its total in
        dollars is not a document anyone has ever been handed - real paper commits to a
        unit at the top and keeps it. The caller passes the unit it drew once, rather than
        each amount drawing its own.
        """
        value = f"{rng.uniform(1, 99999):,.2f}"
        return self._digits(f"{value} {currency or rng.choice(self._CURRENCIES)}")

    def currency(self, rng: random.Random) -> str:
        """The unit a document quotes its money in, drawn once per document."""
        return rng.choice(self._CURRENCIES)

    def money(self, rng: random.Random, value: float, currency: str) -> str:
        """A specific amount in a fixed unit, so a table's arithmetic can hold together."""
        return self._digits(f"{value:,.2f} {currency}")

    def digits(self, text: str) -> str:
        """Text passed through the configured numeral system, formatted by the caller."""
        return self._digits(text)

    def plain_number(self, value: int | float, decimals: int = 0) -> str:
        """A bare quantity - no unit, no currency."""
        return self._digits(f"{value:,.{decimals}f}")

    def code(self, rng: random.Random) -> str:
        """A reference or serial: the case with no language model to fall back on."""
        letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(rng.randint(2, 4)))
        digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(4, 9)))
        separator = rng.choice(["-", "/", "", " "])
        return self._digits(f"{letters}{separator}{digits}")

    def phone(self, rng: random.Random) -> str:
        return self._digits(f"+212 {rng.randint(5, 7)}{rng.randint(10, 99)} {rng.randint(100000, 999999)}")

    def percentage(self, rng: random.Random) -> str:
        return self._digits(f"{rng.uniform(0, 100):.1f}%")

    def any_field(self, rng: random.Random, currency: str | None = None) -> str:
        """One arbitrary non-prose value. `currency` keeps a document's money consistent."""
        pick = rng.choice(["date", "amount", "code", "phone", "percentage"])
        if pick == "amount":
            return self.amount(rng, currency)
        return getattr(self, pick)(rng)


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

    #: Markup that must never reach a rendered page. A corpus stored as Markdown is text
    #: *plus* instructions for setting it, and only the text is the document: drawing
    #: "## العنوان" puts two hashes on the paper and then claims them as ground truth, so a
    #: model learns that Arabic headings begin with punctuation they have never worn. The
    #: same corpus is parsed properly in "documents" mode, where an adapter turns the
    #: markers into structure. Here the sentences are borrowed only for their language, so
    #: the markers are noise.
    #: Anchored to whitespace rather than to line starts. The corpus reaches this class
    #: already whitespace-collapsed, so a heading that began its own line arrives mid-string
    #: as "... كاملة. ## جمع" - a `^` anchor matched none of it, and the markers sailed
    #: through onto the page. Nothing here is legitimate Arabic prose in these positions.
    _MARKUP = re.compile(
        r"(?:(?<=^)|(?<=\s))#{1,6}(?=\s)"  # ATX heading markers
        r"|(?:(?<=^)|(?<=\s))>(?=\s)"  # block quotes
        r"|(?:(?<=^)|(?<=\s))\d+[.)](?=\s)"  # ordered-list markers
        r"|(?:(?<=^)|(?<=\s))[-*+](?=\s)"  # bullets
        r"|-{3,}"  # horizontal and table rules
        r"|\|"  # table cell separators
        r"|\*{1,3}"  # bold / italic markers
        r"|`+"  # code spans
        r"|~~",  # strikethrough
        re.MULTILINE,
    )
    #: Collapses the gaps left behind once markers are removed.
    _SPACES = re.compile(r"[ \t]{2,}")
    #: A link becomes its visible text; the target is not on the page.
    _LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")

    def __init__(self, texts: Iterable[str], *, min_words: int = 3, strip_markup: bool = True):
        self._sentences: list[str] = []
        for text in texts:
            cleaned = self._plain(str(text)) if strip_markup else str(text)
            self._sentences.extend(self._split(cleaned, min_words))
        if not self._sentences:
            raise ValueError("CorpusTextProvider needs at least one usable sentence")

    @classmethod
    def _plain(cls, text: str) -> str:
        """The document's words, with the instructions for setting them removed."""
        return cls._SPACES.sub(" ", cls._MARKUP.sub(" ", cls._LINK.sub(r"\1", text)))

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

    def fragment(self, rng: random.Random) -> str:
        """A partial word, as produced by an occlusion or a crop boundary.

        A recogniser meets these constantly at the edge of a detected box, and a corpus of
        whole words teaches it to complete rather than to read what is there.
        """
        word = rng.choice(self.sentence(rng).split())
        if len(word) < 3:
            return word
        cut = rng.randint(1, len(word) - 1)
        return word[:cut] if rng.random() < 0.5 else word[cut:]

    def contextless(self, rng: random.Random, max_words: int = 4) -> str:
        """Unrelated words in sequence, with no sentence to predict from."""
        return " ".join(rng.choice(self.sentence(rng).split()) for _ in range(rng.randint(1, max_words)))

    @classmethod
    def from_sentences(cls, sentences: Sequence[str]) -> CorpusTextProvider:
        provider = cls.__new__(cls)
        provider._sentences = list(sentences)
        if not provider._sentences:
            raise ValueError("CorpusTextProvider needs at least one sentence")
        return provider
