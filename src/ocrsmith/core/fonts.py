"""Font discovery, caching and coverage-aware selection.

A font pool answers one question well: *which faces can actually draw this text?* Asking
it up front is what prevents the commonest silent corruption in synthetic OCR data — a
label claiming characters the chosen face renders as empty boxes.

Loaded `FreeTypeFont` objects are cached per (path, size) because a generation run asks
for the same handful of faces millions of times and FreeType instantiation is not free.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont

from ..text.coverage import (
    drawable_codepoints,
    required_codepoints,
    supports_text,
    warm_coverage_cache,
)
from ..text.shaping import TextShaper, resolve_shaper

__all__ = ["FontPool", "clear_font_cache", "discover_fonts", "font_variations", "load_font"]

_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
_cache: dict[tuple[str, int, str], FreeTypeFont] = {}
_cache_lock = threading.Lock()


def load_font(path: str | Path, size: int, variation: str | None = None) -> FreeTypeFont:
    """Load a font face, reusing an already-instantiated one when possible.

    `variation` selects a named instance of a variable font. Without it a variable font
    renders only its default instance, which collapses a whole family's weight range onto
    one face — and roughly half of the Arabic families on Google Fonts are variable.
    """
    key = (str(path), int(size), variation or "")
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype(str(path), size=int(size))
        if variation:
            font.set_variation_by_name(variation)
    except OSError as exc:
        raise ValueError(f"Unable to load font from {path!r}: {exc}") from exc
    with _cache_lock:
        _cache[key] = font
    return font


@lru_cache(maxsize=1024)
def font_variations(path: str | Path) -> tuple[str, ...]:
    """Named instances of a variable font, or an empty tuple for a static one."""
    try:
        font = ImageFont.truetype(str(path), size=12)
        names = font.get_variation_names()
    except (OSError, AttributeError):
        return ()
    return tuple(name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names)


def clear_font_cache() -> None:
    with _cache_lock:
        _cache.clear()


def discover_fonts(
    paths: Iterable[str | Path],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Collect font files under `paths`, filtered by filename substrings."""
    found: list[Path] = []
    for raw in paths:
        root = Path(raw)
        if root.is_file() and root.suffix.lower() in _FONT_EXTENSIONS:
            found.append(root)
        elif root.is_dir():
            found.extend(
                path
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.lower() in _FONT_EXTENSIONS
            )

    def keep(path: Path) -> bool:
        name = path.name
        if include and not any(fragment in name for fragment in include):
            return False
        return not any(fragment in name for fragment in exclude)

    return tuple(dict.fromkeys(path for path in found if keep(path)))


def _coverage_or_none(face: Path) -> frozenset[int] | None:
    """A face's drawable code points, or None if its container cannot be read.

    An unreadable font is dropped from the pool's coverage index rather than raising:
    a directory of a hundred faces picked up from a system font path will occasionally
    contain one bitmap-only or malformed file, and that is not a reason to fail a run.
    """
    try:
        return drawable_codepoints(face)
    except Exception:
        return None


class FontPool:
    """The set of faces available to a run, with coverage-aware selection."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        require_full_coverage: bool = True,
        shaper: TextShaper | None = None,
    ):
        self.faces = discover_fonts(paths, include=include, exclude=exclude)
        if not self.faces:
            raise ValueError(f"No font files found under {list(paths)!r}")
        self.require_full_coverage = require_full_coverage
        # Coverage must be judged against the characters that will actually be drawn,
        # which is the shaper's business — so the pool needs the same shaper the renderer
        # will use.
        self.shaper = shaper or resolve_shaper()
        self._coverage_cache: dict[tuple[str, str], bool] = {}
        # Read every face's drawable code points now, once, and keep them beside the
        # `Path` they belong to. Each answer is cached on disk, so this is ~50 ms per
        # *new* face and a few milliseconds per run thereafter — but doing it lazily
        # charges it to whichever sample asks first, in every worker, and that is what
        # used to make coverage the slowest stage in a run.
        #
        # Holding the sets here rather than looking them up per query matters too: a line
        # run asks "which of these hundred faces can draw this?" once per sample, and
        # going through `drawable_codepoints` each time meant rebuilding a hundred `Path`
        # objects per sample just to reach a memo. That was a fifth of the line pipeline.
        warm_coverage_cache(self.faces)
        self._drawable: tuple[tuple[Path, frozenset[int]], ...] = tuple(
            (face, coverage)
            for face in self.faces
            if (coverage := _coverage_or_none(face)) is not None
        )

    def __len__(self) -> int:
        return len(self.faces)

    def supporting(self, text: str) -> tuple[Path, ...]:
        """Faces that can draw `text`, honouring the pool's coverage requirement."""
        if not self.require_full_coverage:
            return self.faces
        needed = required_codepoints(self._probe(text))
        if not needed:
            return self.faces
        return tuple(face for face, drawable in self._drawable if needed <= drawable)

    def choose(self, text: str, rng: random.Random) -> Path:
        """Pick a face that can draw `text`.

        Falls back to the face with the best coverage rather than failing outright: a
        single unusual character should not stop a document being generated, but the
        choice should still be the least-bad one.
        """
        eligible = self.supporting(text)
        if eligible:
            return rng.choice(eligible)
        needed = required_codepoints(self._probe(text))
        if not needed or not self._drawable:
            return self.faces[0]
        return max(self._drawable, key=lambda item: len(needed & item[1]))[0]

    def covers(self, path: str | Path, text: str) -> bool:
        key = (str(path), self._probe(text))
        cached = self._coverage_cache.get(key)
        if cached is None:
            cached = supports_text(path, key[1]).is_complete
            self._coverage_cache[key] = cached
        return cached

    def _probe(self, text: str) -> str:
        """The distinct characters that will actually be *drawn* for `text`.

        Shaping happens first, and this is the whole point. Without Raqm the renderer
        draws Arabic presentation forms (U+FE70-FEFF), not the base letters — and a modern
        OpenType face such as Fustat or Mada covers the base block while carrying no
        presentation-form glyphs at all, because it joins letters via GSUB instead. Probing
        the logical string reports 100% coverage for such a font and every glyph then
        renders as tofu, with the label still claiming the text.

        Collapsing to a character set afterwards turns a per-document question into a
        per-alphabet one, so the coverage cache actually hits.
        """
        return "".join(sorted(set(self.shaper.shape(text).visual)))
