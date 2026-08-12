"""Bidi and Arabic shaping.

A rendered Arabic line is stored twice: the *logical* string, which is the label a model
must predict, and the *visual* string, which is what gets handed to the rasteriser. They
differ whenever the text is right-to-left or needs joined letter forms, and mixing them
up is the single most common way to produce an unusable Arabic OCR dataset.

Two backends cover the two environments we can find ourselves in:

* `TransparentShaper` — Pillow was built against Raqm/HarfBuzz, so the rasteriser applies
  shaping and the bidi algorithm itself. Passing it the logical string is both correct
  and measurable (`font.getlength` runs the same shaper).
* `ReshaperBidiShaper` — no Raqm, so we substitute presentation forms and reorder to
  visual order ourselves before drawing.

Either way `ShapedText.logical` is the label, so the dataset is identical across machines
even though the pixels are produced by different paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, runtime_checkable

from .script import Direction, detect_direction

__all__ = [
    "ShapedText",
    "TextShaper",
    "TransparentShaper",
    "ReshaperBidiShaper",
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
    TransparentShaper.name: TransparentShaper,
    ReshaperBidiShaper.name: ReshaperBidiShaper,
}


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

    ``"auto"`` picks `TransparentShaper` when Pillow has Raqm and falls back to
    `ReshaperBidiShaper` otherwise, so Arabic renders correctly on either build.
    """
    if backend == "auto":
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
    """A reshaper whose ligature-regex cache actually works.

    arabic-reshaper 3.0.0 guards that cache with `hasattr(self, '__ligatures_re')`, but
    writes it to `self.__ligatures_re` — which, inside the class body, Python mangles to
    `_ArabicReshaper__ligatures_re`. The string passed to `hasattr` is *not* mangled, so
    the guard checks a name that is never set and the regex is rebuilt on every call,
    re-reading around 290 configparser entries each time.

    Warming the property once and then setting the unmangled name makes the guard fire
    from the second call onwards. Reaching into a third-party private attribute is not
    something to do lightly; it is contained to this adapter, and the alternative is
    paying that cost on every word of every page.
    """
    from arabic_reshaper import ArabicReshaper

    reshaper = ArabicReshaper()
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
