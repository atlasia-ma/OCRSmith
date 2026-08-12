"""A generated sample: the image, its annotation, and how it came to exist.

Provenance is a first-class part of the record. A synthetic dataset is only useful if you
can answer "which font/background/degradation produced this failure mode?" months later,
and if you can regenerate any single sample from its seed without rerunning the job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .annotations import Page

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

__all__ = ["Provenance", "Sample"]


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to explain — and reproduce — one sample."""

    seed: int | None = None
    font_path: str | None = None
    font_size: int | None = None
    renderer: str | None = None
    background: str | None = None
    shaper: str | None = None
    source: str | None = None
    template: str | None = None
    degradations: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {key: value for key, value in asdict(self).items() if value not in (None, (), {})}
        if self.degradations:
            data["degradations"] = [dict(d) for d in self.degradations]
        return data


@dataclass(frozen=True, slots=True)
class Sample:
    """One rendered page with its annotation.

    The image is held in memory only for as long as a writer needs it; `to_dict` never
    includes pixels, so annotations can be streamed to JSONL while images go elsewhere.
    """

    id: str
    image: Image
    page: Page
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def text(self) -> str:
        return self.page.text

    @property
    def size(self) -> tuple[int, int]:
        return (self.page.width, self.page.height)

    def to_dict(self, image_path: str | None = None) -> dict:
        data: dict = {"id": self.id, "page": self.page.to_dict()}
        if image_path is not None:
            data["image_path"] = image_path
        provenance = self.provenance.to_dict()
        if provenance:
            data["provenance"] = provenance
        return data

    def replace_page(self, page: Page) -> Sample:
        return Sample(self.id, self.image, page, self.provenance)

    def replace_image(self, image: Image) -> Sample:
        return Sample(self.id, image, self.page, self.provenance)
