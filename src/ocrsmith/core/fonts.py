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
from pathlib import Path

from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont

from ..text.coverage import fonts_supporting, supports_text

__all__ = ["FontPool", "clear_font_cache", "discover_fonts", "load_font"]

_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
_cache: dict[tuple[str, int], FreeTypeFont] = {}
_cache_lock = threading.Lock()


def load_font(path: str | Path, size: int) -> FreeTypeFont:
    """Load a font face, reusing an already-instantiated one when possible."""
    key = (str(path), int(size))
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype(str(path), size=int(size))
    except OSError as exc:
        raise ValueError(f"Unable to load font from {path!r}: {exc}") from exc
    with _cache_lock:
        _cache[key] = font
    return font


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


class FontPool:
    """The set of faces available to a run, with coverage-aware selection."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        require_full_coverage: bool = True,
    ):
        self.faces = discover_fonts(paths, include=include, exclude=exclude)
        if not self.faces:
            raise ValueError(f"No font files found under {list(paths)!r}")
        self.require_full_coverage = require_full_coverage
        self._coverage_cache: dict[tuple[str, str], bool] = {}

    def __len__(self) -> int:
        return len(self.faces)

    def supporting(self, text: str) -> tuple[Path, ...]:
        """Faces that can draw `text`, honouring the pool's coverage requirement."""
        if not self.require_full_coverage:
            return self.faces
        eligible = fonts_supporting(self.faces, self._probe(text))
        return tuple(Path(path) for path in eligible)

    def choose(self, text: str, rng: random.Random) -> Path:
        """Pick a face that can draw `text`.

        Falls back to the face with the best coverage rather than failing outright: a
        single unusual character should not stop a document being generated, but the
        choice should still be the least-bad one.
        """
        eligible = self.supporting(text)
        if eligible:
            return rng.choice(eligible)
        probe = self._probe(text)
        return max(self.faces, key=lambda path: supports_text(path, probe).ratio)

    def covers(self, path: str | Path, text: str) -> bool:
        key = (str(path), self._probe(text))
        cached = self._coverage_cache.get(key)
        if cached is None:
            cached = supports_text(path, key[1]).is_complete
            self._coverage_cache[key] = cached
        return cached

    @staticmethod
    def _probe(text: str) -> str:
        """The distinct characters of `text`, which is all coverage depends on.

        Collapsing to a character set turns a per-document question into a per-alphabet
        one, so the coverage cache actually hits.
        """
        return "".join(sorted(set(text)))
