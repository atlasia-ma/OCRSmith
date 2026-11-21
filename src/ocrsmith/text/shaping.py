"""Bidi and Arabic shaping.

A rendered Arabic line is stored twice: the *logical* string, which is the label a model
must predict, and the *visual* string, which is what gets handed to the rasteriser. They
differ whenever the text is right-to-left or needs joined letter forms, and mixing them
up is the single most common way to produce an unusable Arabic OCR dataset.

Three backends cover the environments we can find ourselves in:

* `HarfBuzzShaper` — `uharfbuzz` and `freetype-py` are installed, so OCRSmith shapes and
  rasterises the text itself through the font's own `GSUB`/`GPOS`. The best case, and the
  only one in which a modern face that joins through `GSUB` — which is every handwriting
  and calligraphic Arabic font — can be used at all. See `core.rendering.glyphs`.
* `TransparentShaper` — Pillow was built against Raqm, so the rasteriser applies shaping
  and the bidi algorithm itself. Passing it the logical string is both correct and
  measurable (`font.getlength` runs the same shaper).
* `ReshaperBidiShaper` — neither, so we substitute Arabic presentation forms and reorder
  to visual order ourselves before drawing. Correct for the faces that implement the
  presentation-form block, and those are a shrinking, traditional minority.

Either way `ShapedText.logical` is the label, so the dataset is identical across machines
even though the pixels are produced by different paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, runtime_checkable

from .script import Direction, detect_direction

__all__ = [
    "HarfBuzzShaper",
    "ShapedText",
    "TextShaper",
    "TransparentShaper",
    "ReshaperBidiShaper",
    "harfbuzz_available",
    "raqm_available",
    "resolve_shaper",
]


@dataclass(frozen=True, slots=True)
class ShapedText:
    """A string paired with the form in which it should be drawn."""

    logical: str
    visual: str
    direction: Direction

    @property
    def was_reshaped(self) -> bool:
        """True when drawing order differs from label order."""
        return self.logical != self.visual


@runtime_checkable
class TextShaper(Protocol):
    """Turns a logical string into something a rasteriser can draw correctly."""

    def shape(self, text: str) -> ShapedText: ...


class TransparentShaper:
    """Pass text through untouched, delegating shaping and bidi to the rasteriser."""

    name = "raqm"

    def shape(self, text: str) -> ShapedText:
        return ShapedText(logical=text, visual=text, direction=detect_direction(text))


class HarfBuzzShaper:
    """Leave the text alone; `core.rendering.glyphs` shapes it with HarfBuzz.

    Identical behaviour to `TransparentShaper` — the logical string is what gets drawn —
    but a distinct name, because which engine produced a sample is recorded in its
    provenance and the two are not interchangeable. It also keeps the measurement caches
    apart, since the two paths measure different things.
    """

    name = "harfbuzz"

    def shape(self, text: str) -> ShapedText:
        return ShapedText(logical=text, visual=text, direction=detect_direction(text))


#: Shaping is a pure function of the string, so results are cached. This is not a
#: micro-optimisation: arabic-reshaper 3.0.0 guards its ligature-regex cache with
#: `hasattr(self, '__ligatures_re')`, and because that string literal is not name-mangled
#: the guard never fires — so every single call rebuilds the regex, re-reading ~290
#: configparser entries. Laying out one page called it ~1_800 times, which measured as 68%
#: of total generation time. Caching here sidesteps it without patching their library.
_SHAPE_CACHE_SIZE = 200_000


@lru_cache(maxsize=_SHAPE_CACHE_SIZE)
def _visual_form(text: str) -> str:
    return _bidi_display(_reshape(text))


class ReshaperBidiShaper:
    """Substitute Arabic presentation forms and reorder runs to visual order."""

    name = "reshaper"

    def shape(self, text: str) -> ShapedText:
        direction = detect_direction(text)
        if not text:
            return ShapedText(logical=text, visual=text, direction=direction)
        return ShapedText(logical=text, visual=_visual_form(text), direction=direction)


_BACKENDS: dict[str, type] = {
    HarfBuzzShaper.name: HarfBuzzShaper,
    TransparentShaper.name: TransparentShaper,
    ReshaperBidiShaper.name: ReshaperBidiShaper,
}


@lru_cache(maxsize=1)
def harfbuzz_available() -> bool:
    """Whether OCRSmith can shape and rasterise complex scripts itself."""
    from ..core.rendering.glyphs import engine_available

    return engine_available()


@lru_cache(maxsize=1)
def raqm_available() -> bool:
    """Whether Pillow can shape complex scripts on its own."""
    try:
        from PIL import features
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        return False
    try:
        return bool(features.check("raqm"))
    except Exception:  # pragma: no cover - older Pillow builds
        return False


def resolve_shaper(backend: str = "auto") -> TextShaper:
    """Build the shaper named by `backend`.

    ``"auto"`` prefers HarfBuzz, then Raqm, then the reshaper. The order is by how much
    of the font pool each one can draw, not by convenience: the reshaper path needs
    Arabic presentation forms, and the faces that carry them are the traditional
    minority. Arabic renders correctly on all three; only the choice of faces differs.
    """
    if backend == "auto":
        if harfbuzz_available():
            return HarfBuzzShaper()
        return TransparentShaper() if raqm_available() else ReshaperBidiShaper()
    try:
        return _BACKENDS[backend]()
    except KeyError:
        raise ValueError(
            f"Unknown shaping backend {backend!r}. Available: auto, {', '.join(sorted(_BACKENDS))}"
        ) from None


# -- third-party adapters -------------------------------------------------


@lru_cache(maxsize=1)
def _reshaper():
    """A reshaper that keeps the diacritics, and whose ligature-regex cache works.

    **`delete_harakat` must be off.** arabic-reshaper defaults it to `True`, which quietly
    deletes every tashkeel mark from the string before reshaping it. On the no-Raqm path —
    the default on a Windows or conda Pillow — that means the marks never reach the
    rasteriser at all, while the label still carries every one of them. The whole
    `text.diacritics` section becomes inert: `keep`, `partial` and `strip` all render
    identically, and the first two assert vowels that are not in the pixels. A model
    trained on it learns to hallucinate tashkeel, and nothing anywhere raises.

    That is the exact failure this project's invariant exists to prevent, and it is worth
    one line of configuration plus this paragraph. Marks the chosen face cannot draw are a
    separate question, and `text.coverage` answers it: the probe is the shaped string, so
    a font without the combining marks is rejected rather than silently dropping them.

    The rest is a performance fix. arabic-reshaper 3.0.0 guards its ligature-regex cache
    with `hasattr(self, '__ligatures_re')`, but writes it to `self.__ligatures_re` — which,
    inside the class body, Python mangles to `_ArabicReshaper__ligatures_re`. The string
    passed to `hasattr` is *not* mangled, so the guard checks a name that is never set and
    the regex is rebuilt on every call, re-reading around 290 configparser entries each
    time. Warming the property once and then setting the unmangled name makes the guard
    fire from the second call onwards. Reaching into a third-party private attribute is not
    something to do lightly; it is contained to this adapter, and the alternative is paying
    that cost on every word of every page.
    """
    from arabic_reshaper import ArabicReshaper

    reshaper = ArabicReshaper(configuration={"delete_harakat": False})
    reshaper._ligatures_re  # noqa: B018 - builds and caches under the mangled name
    if not hasattr(reshaper, "__ligatures_re"):
        object.__setattr__(reshaper, "__ligatures_re", True)
    return reshaper.reshape


@lru_cache(maxsize=1)
def _bidi():
    """Return python-bidi's `get_display`, which moved package location in 0.5."""
    try:
        from bidi import get_display  # python-bidi >= 0.5
    except ImportError:  # pragma: no cover - older python-bidi
        from bidi.algorithm import get_display
    return get_display


def _reshape(text: str) -> str:
    return _reshaper()(text)


def _bidi_display(text: str) -> str:
    return _bidi()(text)
