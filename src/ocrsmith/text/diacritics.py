"""Arabic diacritics: measuring them, and varying how many survive.

Diacritics (tashkeel) are the marks that fix short vowels and gemination. Arabic OCR
systems handle them badly — it is called out as a systemic weakness in the Arabic
benchmark literature, and it is the first limitation AtlasOCR reports about itself.

The reason models struggle is distributional: real documents are *partially* diacritised.
Religious and pedagogical texts are fully marked, newspapers carry a handful of
disambiguating marks, and most prose has none. A corpus that is uniformly one or the other
teaches a model to expect that uniformity.

What this module deliberately does **not** do is invent diacritics. Adding marks to
undiacritised text requires a diacritiser model and would fabricate ground truth — the
label would assert vowels that no human wrote. Instead: start from a diacritised corpus
and *remove* a sampled fraction. Removal is always truthful, because the remaining text is
a form the source actually supports.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "DiacriticsMode",
    "DiacriticsPolicy",
    "apply_diacritics",
    "count_diacritics",
    "diacritic_ratio",
    "strip_partial",
]

#: Tashkeel, Quranic annotation marks and the superscript alef — the same set
#: `normalization.strip_diacritics` removes, kept here as a character class for sampling.
_MARK = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ࣓-ࣿ]")


class DiacriticsMode(str, Enum):
    """What happens to the marks a source text carries."""

    #: Leave the source exactly as it is.
    KEEP = "keep"
    #: Remove every mark.
    STRIP = "strip"
    #: Remove a fixed fraction, sampled once per document.
    PARTIAL = "partial"
    #: Sample a mode per document, which is what a mixed real corpus looks like.
    MIXED = "mixed"


def count_diacritics(text: str) -> int:
    """How many diacritic marks `text` carries."""
    return len(_MARK.findall(text))


def diacritic_ratio(text: str) -> float:
    """Marks per non-mark character, a cheap measure of how vocalised a text is.

    Fully vocalised Arabic sits near 0.5-0.7; newspaper prose is nearer 0.0-0.05. Useful
    for reporting what a corpus actually contains rather than what it was assumed to.
    """
    marks = count_diacritics(text)
    base = len(text) - marks
    return marks / base if base else 0.0


def strip_partial(text: str, keep_fraction: float, rng: random.Random) -> str:
    """Keep `keep_fraction` of the marks, chosen uniformly at random.

    Marks are dropped independently rather than by region, because partial vocalisation in
    real documents is driven by ambiguity — a writer marks the word that would otherwise be
    misread — and that is scattered, not clustered.
    """
    if keep_fraction >= 1.0:
        return text
    if keep_fraction <= 0.0:
        return _MARK.sub("", text)
    return _MARK.sub(lambda match: match.group(0) if rng.random() < keep_fraction else "", text)


@dataclass(frozen=True, slots=True)
class DiacriticsPolicy:
    """How diacritics vary across a corpus."""

    mode: DiacriticsMode = DiacriticsMode.KEEP
    #: Range the kept fraction is sampled from, for PARTIAL and the partial branch of MIXED.
    keep_range: tuple[float, float] = (0.1, 0.6)
    #: Weights for MIXED: fully marked, partially marked, unmarked. Defaults approximate a
    #: general-purpose corpus, where most prose is unmarked.
    mixed_weights: tuple[float, float, float] = (0.15, 0.25, 0.60)

    def apply(self, text: str, rng: random.Random) -> tuple[str, float]:
        return apply_diacritics(text, self, rng)


def apply_diacritics(text: str, policy: DiacriticsPolicy, rng: random.Random) -> tuple[str, float]:
    """Apply `policy` to `text`.

    Returns the text and the fraction of marks kept, so provenance can record how
    vocalised each sample was — which is what makes a diacritics ablation possible later.
    """
    if not text or not count_diacritics(text):
        return text, 1.0

    mode = policy.mode
    if mode is DiacriticsMode.MIXED:
        mode = rng.choices(
            [DiacriticsMode.KEEP, DiacriticsMode.PARTIAL, DiacriticsMode.STRIP],
            weights=list(policy.mixed_weights),
            k=1,
        )[0]

    if mode is DiacriticsMode.KEEP:
        return text, 1.0
    if mode is DiacriticsMode.STRIP:
        return _MARK.sub("", text), 0.0

    low, high = sorted(policy.keep_range)
    keep = rng.uniform(low, high)
    return strip_partial(text, keep, rng), keep
