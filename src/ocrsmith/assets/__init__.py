"""Fetching third-party assets that the repository does not ship.

Fonts are other people's work under their own licences, and a synthetic-data engine needs
far more of them than any repository should carry. They are fetched on demand instead,
with the licence recorded alongside.
"""

from .fonts import (
    LICENCE_DIRECTORIES,
    FontFamilyRecord,
    fetch_families,
    list_families,
    load_manifest,
    write_manifest,
)

__all__ = [
    "LICENCE_DIRECTORIES",
    "FontFamilyRecord",
    "fetch_families",
    "list_families",
    "load_manifest",
    "write_manifest",
]
