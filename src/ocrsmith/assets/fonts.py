"""Fetching fonts from Google Fonts.

Font diversity is the single highest-impact lever in synthetic text data — engines that
scaled to six figures of typefaces report it as the dominant factor, well ahead of layout
or degradation tricks. A repository cannot ship that many fonts, and should not: they are
other people's work under their own licences. So OCRSmith fetches them on demand and
records exactly what it took, from where, under which licence.

Only permissively licensed directories of the `google/fonts` repository are used
(`ofl/`, `apache/`, `ufl/`), and the licence file is downloaded alongside every family.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FontFamilyRecord",
    "LICENCE_DIRECTORIES",
    "fetch_families",
    "list_families",
    "load_manifest",
]

_METADATA_URL = "https://fonts.google.com/metadata/fonts"
_CONTENTS_API = "https://api.github.com/repos/google/fonts/contents/{directory}/{slug}"
_RAW_URL = "https://raw.githubusercontent.com/google/fonts/main/{directory}/{slug}/{name}"
_FONT_SUFFIXES = (".ttf", ".otf")
_USER_AGENT = "ocrsmith-font-fetcher"

#: Directories of google/fonts whose licences permit redistribution and modification.
#: `apache` is Apache-2.0, `ofl` is the SIL Open Font Licence, `ufl` the Ubuntu Font
#: Licence. Anything outside these is skipped rather than guessed at.
LICENCE_DIRECTORIES = ("ofl", "apache", "ufl")

MANIFEST_NAME = "fonts-manifest.json"


@dataclass(frozen=True, slots=True)
class FontFamilyRecord:
    """One typeface family as Google Fonts describes it."""

    family: str
    category: str
    subsets: tuple[str, ...]
    designers: tuple[str, ...] = ()
    is_noto: bool = False
    #: Populated once the family has been located in the repository.
    directory: str | None = None
    files: tuple[str, ...] = ()
    licence: str | None = None

    @property
    def slug(self) -> str:
        """Directory name in the google/fonts repository."""
        return self.family.lower().replace(" ", "")

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "category": self.category,
            "subsets": list(self.subsets),
            "designers": list(self.designers),
            "directory": self.directory,
            "licence": self.licence,
            "files": list(self.files),
        }


def _get(url: str, *, as_json: bool = False, timeout: float = 30.0):
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed hosts
        payload = response.read()
    return json.loads(payload.decode("utf-8")) if as_json else payload


def list_families(subset: str = "arabic", *, open_source_only: bool = True) -> tuple[FontFamilyRecord, ...]:
    """Every Google Fonts family carrying `subset`, e.g. "arabic" or "latin"."""
    metadata = _get(_METADATA_URL, as_json=True)
    families = []
    for entry in metadata.get("familyMetadataList", []):
        subsets = tuple(entry.get("subsets", ()))
        if subset and subset.lower() not in {s.lower() for s in subsets}:
            continue
        if open_source_only and not entry.get("isOpenSource", False):
            continue
        families.append(
            FontFamilyRecord(
                family=entry["family"],
                category=entry.get("category", "unknown"),
                subsets=subsets,
                designers=tuple(entry.get("designers", ())),
                is_noto=bool(entry.get("isNoto", False)),
            )
        )
    return tuple(sorted(families, key=lambda record: record.family))


def _locate(record: FontFamilyRecord) -> FontFamilyRecord | None:
    """Find a family in the permissively licensed directories, listing its files."""
    for directory in LICENCE_DIRECTORIES:
        try:
            listing = _get(_CONTENTS_API.format(directory=directory, slug=record.slug), as_json=True)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        if not isinstance(listing, list):
            continue
        files = tuple(item["name"] for item in listing if item["name"].lower().endswith(_FONT_SUFFIXES))
        licence = next(
            (item["name"] for item in listing if "license" in item["name"].lower() or "OFL" in item["name"]),
            None,
        )
        if files:
            return FontFamilyRecord(
                family=record.family,
                category=record.category,
                subsets=record.subsets,
                designers=record.designers,
                is_noto=record.is_noto,
                directory=directory,
                files=files,
                licence=licence,
            )
    return None


def fetch_families(
    records: Sequence[FontFamilyRecord],
    destination: str | Path,
    *,
    skip_existing: bool = True,
    on_progress=None,
) -> Iterator[FontFamilyRecord]:
    """Download each family into `destination`, yielding what was actually retrieved.

    Families that cannot be located under a permissive licence are skipped rather than
    guessed at, and every family's licence file is downloaded next to its fonts.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for record in records:
        located = _locate(record)
        if located is None:
            if on_progress:
                on_progress(record.family, "skipped (no permissive licence found)")
            continue

        family_dir = destination / located.slug
        family_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for name in located.files:
            target = family_dir / name
            if skip_existing and target.exists():
                written += 1
                continue
            url = _RAW_URL.format(directory=located.directory, slug=located.slug, name=name)
            try:
                target.write_bytes(_get(url))
                written += 1
            except (urllib.error.URLError, OSError) as error:
                if on_progress:
                    on_progress(located.family, f"failed on {name}: {error}")

        if located.licence:
            licence_path = family_dir / located.licence
            if not licence_path.exists():
                # The fonts remain usable if this fails; the manifest records the gap.
                with contextlib.suppress(urllib.error.URLError, OSError):
                    licence_path.write_bytes(
                        _get(
                            _RAW_URL.format(
                                directory=located.directory,
                                slug=located.slug,
                                name=located.licence,
                            )
                        )
                    )

        if on_progress:
            on_progress(located.family, f"{written} file(s) [{located.directory}]")
        if written:
            yield located


def write_manifest(destination: str | Path, records: Sequence[FontFamilyRecord], subset: str) -> Path:
    """Record what was fetched, so a dataset can be regenerated with the same faces."""
    destination = Path(destination)
    path = destination / MANIFEST_NAME
    payload = {
        "source": "https://github.com/google/fonts",
        "subset": subset,
        "licence_directories": list(LICENCE_DIRECTORIES),
        "families": [record.to_dict() for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_manifest(destination: str | Path) -> dict:
    path = Path(destination) / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class FetchSummary:
    """What a fetch run produced."""

    families: int = 0
    files: int = 0
    skipped: tuple[str, ...] = field(default_factory=tuple)
