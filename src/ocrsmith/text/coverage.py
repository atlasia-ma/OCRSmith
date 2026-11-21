"""Font glyph coverage.

A font that lacks a glyph does not fail — it draws a blank or a tofu box while the label
still claims the character is there. At scale that quietly teaches a model to hallucinate.
Coverage is therefore checked up front and unsupported (font, text) pairs are rejected
before they can reach the dataset.

Coverage is read from the font's `cmap` and its outlines via fontTools, which is exact and
needs no rendering. The expensive part is the outline check, so it is done **once per
font, in bulk**, and the resulting code-point set is cached in memory and on disk.

Why bulk, and why on disk
-------------------------
The obvious implementation asks "can this font draw this character?" lazily and memoises
the answer. That is what this module used to do, and it was the single slowest thing in a
generation run: 38% of wall clock. Two reasons.

* Answering one code point needs the font's `glyf`, `cmap`, `post` and `hmtx` tables
  decompiled. A bounded cache of opened fonts (there were 64 slots for a pool of 101
  faces) therefore thrashed, and each eviction threw away megabytes of decompiled tables
  that were immediately rebuilt.
* Every worker process repeats the whole warm-up, so the cost multiplies by the worker
  count rather than amortising across it.

Computing the full drawable set costs about 50 ms per face, and a face's answer depends
only on its bytes — so it is cached under `OCRSMITH_CACHE_DIR` keyed by path, size and
mtime. After the first run, a pool of a hundred faces is ready in a few milliseconds, in
every worker.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CoverageReport",
    "cache_dir",
    "drawable_codepoints",
    "fonts_supporting",
    "has_glyph",
    "required_codepoints",
    "missing_glyphs",
    "supports_text",
    "warm_coverage_cache",
]

# Characters no font is expected to carry a glyph for and which never need one.
_IGNORED = set(" \t\n\r​‌‍‎‏﻿")

#: Bumped whenever the meaning of a cached entry changes, so a stale cache is ignored
#: rather than trusted. A wrong coverage answer is invisible in the output.
_CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Which characters of a text a font can actually draw."""

    font_path: str
    total: int
    missing: tuple[str, ...]

    @property
    def covered(self) -> int:
        return self.total - len(self.missing)

    @property
    def ratio(self) -> float:
        """Fraction of checkable characters the font can draw; 1.0 for empty text."""
        return 1.0 if self.total == 0 else self.covered / self.total

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def __bool__(self) -> bool:
        return self.is_complete


# -- the drawable-code-point set ------------------------------------------------------


def _has_ink(name: str, glyf, charstrings, glyph_set) -> bool:
    """Whether the named glyph puts marks on the page.

    Being in the `cmap` is not enough. A font may map a code point to a glyph with no
    outline at all — an empty shape carrying a normal advance width. Estedad does exactly
    this for the Arabic presentation forms (U+FE70–FEFF): it supplies its real joining
    shapes through GSUB substitution into differently named glyphs, and leaves the glyphs
    reachable by code point as hollow stubs. Without HarfBuzz applying GSUB the renderer
    lands on the stubs, advances the pen, and draws nothing — while the label goes on
    claiming every letter. A cmap-only check calls that font fully covering.

    Three ways to answer, cheapest first. For a TrueType outline the contour count is
    already in the parsed glyph and settles it (0 contours is empty, -1 is a composite and
    therefore inked). For CFF the charstring length does: an empty glyph compiles to a
    bare width-and-endchar of a handful of bytes. Only an exotic container falls through
    to actually tracing the outline.
    """
    if glyf is not None:
        try:
            return glyf[name].numberOfContours != 0
        except Exception:
            return True
    if charstrings is not None:
        try:
            return len(charstrings[name].bytecode or b"") > 4
        except Exception:
            # A charstring that is not stored as raw bytecode has to be traced.
            pass
    try:
        from fontTools.pens.boundsPen import BoundsPen

        pen = BoundsPen(glyph_set)
        glyph_set[name].draw(pen)
    except Exception:
        # An outline we cannot trace is not evidence of absence; assume it draws.
        return True
    return pen.bounds is not None


def _compute_drawable(font_path: str) -> frozenset[int]:
    """Every code point the font both maps and can actually draw."""
    from fontTools.ttLib import TTFont

    font = TTFont(font_path, fontNumber=0, lazy=True)
    try:
        cmap = font.getBestCmap()
        glyf = font.get("glyf", None)
        charstrings = None
        if glyf is None and "CFF " in font:
            try:
                charstrings = font["CFF "].cff[font["CFF "].cff.fontNames[0]].CharStrings
            except Exception:
                charstrings = None
        glyph_set = font.getGlyphSet() if (glyf is None and charstrings is None) else None
        return frozenset(
            codepoint for codepoint, name in cmap.items() if _has_ink(name, glyf, charstrings, glyph_set)
        )
    finally:
        with contextlib.suppress(Exception):
            font.close()


# -- caching ---------------------------------------------------------------------------


def cache_dir() -> Path | None:
    """Where coverage answers are kept between runs, or None when caching is off.

    `OCRSMITH_CACHE_DIR` overrides the location; setting it to an empty string disables
    the disk cache entirely, so a stale file can never make a coverage test pass.
    """
    override = os.environ.get("OCRSMITH_CACHE_DIR")
    if override is not None:
        return Path(override) if override.strip() else None
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "ocrsmith" / "fonts"


def _fingerprint(path: Path) -> str | None:
    """Identity of the font's *bytes*, cheaply. None when the file cannot be stated."""
    try:
        stat = path.stat()
    except OSError:
        return None
    import hashlib

    raw = f"{_CACHE_VERSION}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _ranges(codepoints: frozenset[int]) -> list[list[int]]:
    """Code points as inclusive ranges — a font's cmap is mostly contiguous runs."""
    out: list[list[int]] = []
    for codepoint in sorted(codepoints):
        if out and codepoint == out[-1][1] + 1:
            out[-1][1] = codepoint
        else:
            out.append([codepoint, codepoint])
    return out


def _unranges(ranges: Iterable[Sequence[int]]) -> frozenset[int]:
    return frozenset(codepoint for low, high in ranges for codepoint in range(int(low), int(high) + 1))


def _read_cached(directory: Path, key: str) -> frozenset[int] | None:
    try:
        payload = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("v") != _CACHE_VERSION:
        return None
    return _unranges(payload.get("ranges", ()))


def _write_cached(directory: Path, key: str, codepoints: frozenset[int]) -> None:
    """Publish an entry atomically, because every worker process races to write it."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"v": _CACHE_VERSION, "ranges": _ranges(codepoints)}, stream)
        os.replace(temporary, directory / f"{key}.json")
    except OSError:
        pass  # a cache that cannot be written is a slow run, not a failed one


#: In-process memo. Unbounded on purpose: it is keyed by font file, and a run's font pool
#: is bounded by the number of faces on disk. The bounded cache this replaced is what made
#: coverage the slowest part of a run, by evicting faces it was about to need again.
_MEMO: dict[str, frozenset[int]] = {}


def drawable_codepoints(font_path: str | Path) -> frozenset[int]:
    """Every code point the font at `font_path` can actually draw.

    Raises whatever fontTools raises for an unreadable file; callers that iterate over a
    directory of fonts are expected to tolerate one bad container.
    """
    path = Path(font_path)
    key = str(path)
    memoised = _MEMO.get(key)
    if memoised is not None:
        return memoised

    directory = cache_dir()
    fingerprint = _fingerprint(path) if directory is not None else None
    if directory is not None and fingerprint is not None:
        cached = _read_cached(directory, fingerprint)
        if cached is not None:
            _MEMO[key] = cached
            return cached

    computed = _compute_drawable(str(path))
    if directory is not None and fingerprint is not None:
        _write_cached(directory, fingerprint, computed)
    _MEMO[key] = computed
    return computed


def warm_coverage_cache(font_paths: Iterable[str | Path]) -> int:
    """Pre-compute coverage for every face, returning how many were readable.

    Called once when a font pool is built. Doing it here rather than on first use keeps
    the cost out of the middle of a run, where it would be charged to whichever sample
    happened to ask first and would read as erratic per-sample latency.
    """
    count = 0
    for path in font_paths:
        try:
            drawable_codepoints(path)
        except Exception:
            continue
        count += 1
    return count


def clear_coverage_cache() -> None:
    """Forget in-process coverage answers. For tests; the disk cache is untouched."""
    _MEMO.clear()


# -- public queries --------------------------------------------------------------------


def required_codepoints(text: str) -> frozenset[int]:
    """The code points a font must be able to draw in order to render `text`.

    Whitespace and the zero-width formatting marks are excluded. They are not absences a
    font can be blamed for: no face carries an outline for U+0020, and requiring one
    rejects every font for every string that contains a space.
    """
    return frozenset(ord(char) for char in text if char not in _IGNORED)


def has_glyph(font_path: str | Path, char: str) -> bool:
    """Whether the font at `font_path` can actually draw `char`."""
    if not char or char in _IGNORED:
        return True
    try:
        return ord(char[0]) in drawable_codepoints(font_path)
    except Exception:
        return True  # unreadable container: let the caller's own guard handle it


def missing_glyphs(font_path: str | Path, text: str) -> tuple[str, ...]:
    """Unique characters of `text` the font cannot draw, in first-seen order."""
    try:
        drawable = drawable_codepoints(font_path)
    except Exception:
        return ()
    missing: list[str] = []
    seen: set[str] = set()
    for char in text:
        if char in _IGNORED or char in seen:
            continue
        seen.add(char)
        if ord(char) not in drawable:
            missing.append(char)
    return tuple(missing)


def supports_text(font_path: str | Path, text: str) -> CoverageReport:
    """Full coverage report for rendering `text` with the font at `font_path`."""
    checkable = {char for char in text if char not in _IGNORED}
    return CoverageReport(
        font_path=str(font_path),
        total=len(checkable),
        missing=missing_glyphs(font_path, text),
    )


def fonts_supporting(
    font_paths: Iterable[str | Path],
    text: str,
    *,
    min_ratio: float = 1.0,
) -> Sequence[str]:
    """Subset of `font_paths` that can draw `text` to at least `min_ratio` coverage.

    Used to pick a font per sample instead of discovering mid-render that the chosen
    face has no Arabic in it. A face whose container cannot be read is ineligible rather
    than fatal.
    """
    needed = required_codepoints(text)
    total = len(needed)
    eligible: list[str] = []
    for path in font_paths:
        try:
            drawable = drawable_codepoints(path)
        except Exception:
            continue  # unreadable or exotic font container: treat as ineligible
        if total == 0:
            eligible.append(str(path))
            continue
        covered = len(needed & drawable)
        if covered / total >= min_ratio:
            eligible.append(str(path))
    return eligible
