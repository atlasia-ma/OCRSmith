"""Document templates.

Each template describes one *genre* of document. Genre matters more than it looks: a model
trained only on flowing articles never learns that a receipt has right-aligned amounts,
that a form has label/value pairs, or that a newspaper's columns are read across a
gutter. The registry lets a generation run sample genres by weight, so a corpus can be
deliberately balanced instead of accidentally uniform.

Templates produce content only. Page size, fonts and degradation are decided elsewhere,
which is what lets the same article appear as a crisp A4 print and a creased phone photo
with identical markup ground truth.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .charts import ChartKind, sample_chart
from .content import DocumentBuilder, DocumentContent
from .formulas import sample_formula
from .text_source import TextProvider

__all__ = [
    "ArticleTemplate",
    "ContentsTemplate",
    "NotesTemplate",
    "SlideTemplate",
    "DocumentTemplate",
    "FormTemplate",
    "InvoiceTemplate",
    "LetterTemplate",
    "NewspaperTemplate",
    "PaperTemplate",
    "ReportTemplate",
    "TemplateRegistry",
    "default_registry",
]


@runtime_checkable
class DocumentTemplate(Protocol):
    """Builds one document's content from a text source."""

    name: str

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent: ...


@dataclass(frozen=True, slots=True)
class ArticleTemplate:
    """Title, lead paragraph, then sections of prose — the commonest document shape."""

    name: str = "article"
    min_sections: int = 2
    max_sections: int = 5

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        direction = options.get("direction")
        builder = DocumentBuilder(direction, template=self.name)
        builder.title(source.title(rng))
        builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        for _ in range(rng.randint(self.min_sections, self.max_sections)):
            builder.heading(source.title(rng))
            for _ in range(rng.randint(1, 3)):
                builder.paragraph(source.paragraph(rng, rng.randint(2, 5)))
            if rng.random() < 0.25:
                builder.list([source.phrase(rng, 6) for _ in range(rng.randint(2, 5))])
        return builder.build()


@dataclass(frozen=True, slots=True)
class ReportTemplate:
    """Headed report with a data table, a figure and a caption."""

    name: str = "report"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.header(source.phrase(rng, 4))
        builder.title(source.title(rng))
        builder.paragraph(source.paragraph(rng, 3))
        builder.heading(source.title(rng))
        builder.table(_sample_table(source, rng))
        builder.caption(source.phrase(rng, 6))
        builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        if rng.random() < 0.55:
            labels = [source.phrase(rng, 1) for _ in range(4)]
            builder.chart(
                sample_chart(rng, labels, title=source.phrase(rng, 3)),
                width=rng.randint(260, 420),
                height=rng.randint(190, 300),
            )
            builder.caption(source.phrase(rng, 5))
        elif rng.random() < 0.5:
            builder.figure(width=rng.randint(240, 420), height=rng.randint(160, 300))
            builder.caption(source.phrase(rng, 5))
        builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        return builder.build()


@dataclass(frozen=True, slots=True)
class LetterTemplate:
    """Correspondence: addressee, body, sign-off."""

    name: str = "letter"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.paragraph(source.phrase(rng, 5))
        builder.separator()
        builder.paragraph(source.phrase(rng, 4))
        for _ in range(rng.randint(2, 4)):
            builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        builder.paragraph(source.phrase(rng, 3))
        return builder.build()


@dataclass(frozen=True, slots=True)
class FormTemplate:
    """Label/value rows, the shape used by administrative forms and IDs."""

    name: str = "form"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.title(rng))
        builder.key_values([(source.phrase(rng, 2), source.phrase(rng, 3)) for _ in range(rng.randint(4, 9))])
        builder.separator()
        builder.heading(source.phrase(rng, 3))
        builder.key_values([(source.phrase(rng, 2), source.phrase(rng, 2)) for _ in range(rng.randint(3, 6))])
        return builder.build()


@dataclass(frozen=True, slots=True)
class InvoiceTemplate:
    """Header, line-item table with totals, then terms."""

    name: str = "invoice"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.phrase(rng, 3))
        builder.key_values(
            [
                (source.phrase(rng, 2), f"{rng.randint(1000, 99999)}"),
                (source.phrase(rng, 2), f"{rng.randint(1, 28)}/{rng.randint(1, 12)}/2026"),
            ]
        )
        rows = [
            [
                source.phrase(rng, 2),
                source.phrase(rng, 3),
                str(rng.randint(1, 20)),
                f"{rng.randint(10, 9999)}.{rng.randint(0, 99):02d}",
            ]
        ]
        for _ in range(rng.randint(3, 8)):
            rows.append(
                [
                    source.phrase(rng, 2),
                    source.phrase(rng, 4),
                    str(rng.randint(1, 20)),
                    f"{rng.randint(10, 9999)}.{rng.randint(0, 99):02d}",
                ]
            )
        builder.table(rows)
        builder.key_values([(source.phrase(rng, 1), f"{rng.randint(100, 99999)}.00")])
        builder.paragraph(source.paragraph(rng, 2))
        return builder.build()


@dataclass(frozen=True, slots=True)
class PaperTemplate:
    """An academic paper: prose interleaved with displayed equations.

    Formula conversion is the largest gain category in the document-parsing benchmarks,
    and it needs formulas *in context* - a page of nothing but equations is not what a
    model meets in the wild.
    """

    name: str = "paper"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.title(rng))
        builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
        for _ in range(rng.randint(2, 4)):
            builder.heading(source.title(rng))
            builder.paragraph(source.paragraph(rng, rng.randint(2, 4)))
            builder.formula(sample_formula(rng))
            if rng.random() < 0.4:
                builder.paragraph(source.paragraph(rng, 2))
        if rng.random() < 0.5:
            builder.chart(
                sample_chart(
                    rng,
                    [source.phrase(rng, 1) for _ in range(4)],
                    title=source.phrase(rng, 3),
                    kind=ChartKind.LINE,
                ),
                width=rng.randint(240, 400),
                height=rng.randint(170, 260),
            )
            builder.caption(source.phrase(rng, 5))
        return builder.build()


@dataclass(frozen=True, slots=True)
class NewspaperTemplate:
    """Dense multi-column prose with a masthead and short headlines."""

    name: str = "newspaper"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.header(source.phrase(rng, 3))
        builder.title(source.title(rng))
        for index in range(rng.randint(3, 6)):
            if index:
                builder.heading(source.title(rng), level=3)
            for _ in range(rng.randint(2, 4)):
                builder.paragraph(source.paragraph(rng, rng.randint(3, 6)))
        return builder.build()


@dataclass(frozen=True, slots=True)
class ContentsTemplate:
    """A table of contents: label, dot leaders, page number.

    Dot leaders are their own recognition problem - a long run of identical glyphs that
    models routinely miscount or hallucinate - and no synthetic corpus generates them.
    """

    name: str = "contents"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.phrase(rng, 2))
        page = rng.randint(1, 9)
        for _ in range(rng.randint(8, 16)):
            label = source.phrase(rng, rng.randint(2, 5))
            leader = "." * rng.randint(6, 30)
            builder.paragraph(f"{label} {leader} {page}")
            page += rng.randint(1, 12)
        return builder.build()


@dataclass(frozen=True, slots=True)
class SlideTemplate:
    """A presentation slide: a headline and a few short bullets, set large.

    Slides are a distinct visual regime - very large type, very little of it - and a model
    trained only on dense prose reads them poorly.
    """

    name: str = "slide"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name)
        builder.title(source.phrase(rng, rng.randint(2, 5)))
        builder.list([source.phrase(rng, rng.randint(3, 8)) for _ in range(rng.randint(3, 6))])
        if rng.random() < 0.4:
            builder.chart(
                sample_chart(rng, [source.phrase(rng, 1) for _ in range(4)], title=""),
                width=rng.randint(280, 420),
                height=rng.randint(180, 260),
            )
        builder.footer(source.phrase(rng, 2))
        return builder.build()


@dataclass(frozen=True, slots=True)
class NotesTemplate:
    """Handwritten notes: a heading and loose lines, drawn in a handwriting face.

    The Arabic benchmarks are handwriting-heavy (KHATT, Muharaf, and most of KITAB-Bench's
    handwritten domain), and a printed-only corpus transfers to them poorly. This is not a
    substitute for real handwriting data - the letterforms come from a font, so the
    variability of a human hand is missing - but it covers the layout and the visual
    regime, and it is honest about which it is.
    """

    name: str = "notes"

    def build(self, source: TextProvider, rng: random.Random, **options) -> DocumentContent:
        builder = DocumentBuilder(options.get("direction"), template=self.name, handwritten=True)
        builder.heading(source.phrase(rng, rng.randint(2, 4)))
        for _ in range(rng.randint(4, 9)):
            builder.paragraph(source.sentence(rng))
        if rng.random() < 0.5:
            builder.list([source.phrase(rng, rng.randint(2, 6)) for _ in range(rng.randint(2, 5))])
        return builder.build()


def _sample_table(source: TextProvider, rng: random.Random) -> list[list[str]]:
    cols = rng.randint(2, 5)
    rows = rng.randint(3, 7)
    header = [source.phrase(rng, 2) for _ in range(cols)]
    body = [
        [source.phrase(rng, 2) if col == 0 else str(rng.randint(0, 9999)) for col in range(cols)]
        for _ in range(rows)
    ]
    return [header, *body]


class TemplateRegistry:
    """Named templates with sampling weights."""

    def __init__(self):
        self._templates: dict[str, DocumentTemplate] = {}
        self._weights: dict[str, float] = {}

    def register(self, template: DocumentTemplate, weight: float = 1.0) -> TemplateRegistry:
        self._templates[template.name] = template
        self._weights[template.name] = max(0.0, weight)
        return self

    def get(self, name: str) -> DocumentTemplate:
        try:
            return self._templates[name]
        except KeyError:
            raise ValueError(
                f"Unknown template {name!r}. Available: {', '.join(sorted(self._templates))}"
            ) from None

    def sample(self, rng: random.Random, names: list[str] | None = None) -> DocumentTemplate:
        candidates = names or list(self._templates)
        weights = [self._weights.get(name, 1.0) for name in candidates]
        if not candidates or sum(weights) <= 0:
            raise ValueError("No templates available to sample from")
        return self._templates[rng.choices(candidates, weights=weights, k=1)[0]]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def __len__(self) -> int:
        return len(self._templates)

    def __iter__(self) -> Iterator[DocumentTemplate]:
        return iter(self._templates.values())

    def __contains__(self, name: object) -> bool:
        return name in self._templates


def default_registry() -> TemplateRegistry:
    """The built-in genres, weighted towards the shapes that dominate real corpora."""
    return (
        TemplateRegistry()
        .register(ArticleTemplate(), weight=3.0)
        .register(ReportTemplate(), weight=2.0)
        .register(NewspaperTemplate(), weight=1.5)
        .register(PaperTemplate(), weight=1.5)
        .register(LetterTemplate(), weight=1.0)
        .register(FormTemplate(), weight=1.0)
        .register(InvoiceTemplate(), weight=1.0)
        .register(ContentsTemplate(), weight=0.8)
        .register(SlideTemplate(), weight=0.8)
        .register(NotesTemplate(), weight=1.0)
    )
