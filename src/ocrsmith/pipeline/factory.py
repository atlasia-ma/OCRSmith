"""Turning a seed into annotated pages.

`SampleFactory.create(index)` is the whole generator in one call: it seeds a private RNG
from `(config.seed, index)`, samples a genre, a page shape, a typography and a capture
condition, renders the document, degrades it, and yields one `Sample` per page.

Everything random flows from that one seeded RNG and nothing reads global state, so
sample 8_412 can be regenerated on its own months later — which is what makes a synthetic
dataset debuggable at all.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from ..config.schema import GenerationConfig, PageConfig
from ..core.backgrounds import BackgroundSampler, adapt_to_background
from ..core.degradations import build_preset
from ..core.documents import (
    BorderStyle,
    CorpusTextProvider,
    DocumentRenderer,
    PageSpec,
    TableStyle,
    TemplateRegistry,
    TypographySampler,
    default_registry,
)
from ..core.documents.cropping import crop_page
from ..core.documents.page_spec import PAPER_SIZES_MM, PageProfile
from ..core.documents.templates import profile_for
from ..core.documents.typography import tint
from ..core.fonts import FontPool
from ..domain import Provenance, Sample
from ..text.diacritics import DiacriticsMode, DiacriticsPolicy, apply_diacritics
from ..text.script import Direction

__all__ = ["SampleFactory", "build_factory", "corpus_advice"]

_DIRECTIONS = {"rtl": Direction.RTL, "ltr": Direction.LTR}

#: `fonts.size_range` is written in pixels, but a pixel means nothing without a
#: resolution: 20px is a readable 10pt body at 150 dpi and an illegible 6.6pt one at 220.
#: Sampling the two independently produced pages whose type was too small to fill the
#: sheet and whose strokes dissolved under mild grain. The configured range is therefore
#: read as pixels *at this reference resolution* and scaled to the page's own dpi.
_REFERENCE_DPI = 150


def _weighted_choice(weights: dict, rng: random.Random):
    usable = {key: value for key, value in weights.items() if value > 0}
    keys = list(usable)
    return rng.choices(keys, weights=[usable[key] for key in keys], k=1)[0]


class SampleFactory:
    """Builds annotated pages from a `GenerationConfig`.

    Constructed once per process. Everything expensive — font discovery, the corpus, the
    template registry — is set up here so that `create` is cheap and stateless apart from
    its own RNG.
    """

    def __init__(
        self,
        config: GenerationConfig,
        *,
        text_provider: CorpusTextProvider | None = None,
        registry: TemplateRegistry | None = None,
    ):
        self.config = config
        self.registry = registry or default_registry()
        self.fonts = FontPool(
            config.fonts.paths,
            include=config.fonts.include,
            exclude=config.fonts.exclude,
            require_full_coverage=config.fonts.require_full_coverage,
        )
        # Two ways a document's content can arrive, and only one is loaded. "sentences"
        # invents a document from a pool via the template registry; "documents" lays out
        # text that already exists, parsed by an input adapter. Everything after this
        # point consumes `DocumentContent` and cannot tell which produced it.
        self.documents: list[str] = []
        if config.text.mode == "documents":
            self.text = text_provider or CorpusTextProvider(config.text.source.sentences)
            self.documents = build_documents(config)
            if not self.documents:
                raise ValueError("text.mode is 'documents' but the source yielded nothing to lay out")
        else:
            self.text = text_provider or build_text_provider(config)
        self.backgrounds = BackgroundSampler(
            config.page.background.kinds,
            image_paths=config.page.background.image_paths,
            tint_range=config.page.background.tint_range,
        )
        self.renderer = DocumentRenderer()

    # -- generation --------------------------------------------------------

    def create(self, index: int) -> Iterator[Sample]:
        """Yield one `Sample` per page of the document generated for `index`."""
        seed = self.config.sample_seed(index)
        rng = random.Random(seed)

        direction = self._direction(rng)
        parsed = None
        if self.documents:
            source_name, parsed = self._parse_document(rng, direction)
            content = parsed
        else:
            template = self.registry.sample(rng, self._template_names(rng))
            source_name = template.name
            # The sheet is chosen first so the template can be told how much room it has.
            # Direction only orders the columns, so a provisional value is enough here and
            # the real one is applied once the content has reported its own script.
            profile = profile_for(source_name)
            spec = self._spec_from_profile(profile, direction or Direction.RTL, rng)
            content = template.build(
                self.text,
                rng,
                direction=direction,
                numerals=self.config.text.normalization.numerals,
                size_hint=self._capacity(spec, profile),
            )

        # The page must read in the same direction as its text. With `direction: auto` the
        # configured value is None, and defaulting the *spec* to LTR ordered the columns
        # left-first while the text inside them ran right-to-left - so an Arabic page was
        # read in the wrong column order.
        profile = None if parsed is not None else profile_for(source_name)
        if parsed is not None:
            # Text first, then a sheet it fills. Sampling the two independently is what
            # put one paragraph in the corner of an A3: with source documents shorter
            # than a large sheet holds, no chunk can ever fill one, so the sheet has to
            # follow the text rather than the other way round.
            content = self._chunk_to_fit(parsed, rng)
            spec = self._page_spec_for(content, direction or content.direction, rng)
        else:
            if spec.direction is not (content.direction or spec.direction):
                spec = replace(spec, direction=content.direction)

        content = self._with_footer(content, rng)
        content, diacritics_kept = self._apply_diacritics(content, rng)
        background, background_kind = self.backgrounds.sample(rng)
        # The ink follows the stock. A photograph of dark slate is a surface a document can
        # genuinely be printed on, so it keeps its picture and gets light ink, rather than
        # being washed pale so that black text has somewhere to sit.
        background, ink = adapt_to_background(background, rng, (spec.width, spec.height))
        typography = self._typography(
            content,
            rng,
            spec.dpi,
            ink=ink,
            heavy=background_kind == "image",
            profile=profile,
        )
        preset_name = _weighted_choice(self.config.degradations.presets, rng)
        degradations = build_preset(preset_name)

        # Drained rather than streamed, because what overflowed is only known once the
        # last page has been composed and every page's provenance should carry it. The
        # cost is bounded by `max_pages` - one document's pages, never the run's.
        overflow: list = []
        pages = list(
            self.renderer.iter_pages(
                content,
                spec,
                typography,
                background=background,
                table_style=self._table_style(
                    rng,
                    ink,
                    accent=typography.accent,
                    plain_stock=background_kind in ("paper", "solid"),
                ),
                rng=rng,
                max_pages=self.config.page.max_pages,
                overflow=overflow,
            )
        )
        dropped = len(overflow)
        dropped_words = sum(len(block.text.split()) for block in overflow)

        capture_cfg = self.config.page.capture
        for rendered in pages:
            image, page = rendered.image.convert("RGB"), rendered.page
            capture = "full"
            if capture_cfg.partial_probability and rng.random() < capture_cfg.partial_probability:
                # Cropped before degradation, not after: a partial capture is what the
                # sensor saw, so the blur, noise and lighting belong to the fragment.
                capture = _weighted_choice(capture_cfg.modes, rng)
                image, page = crop_page(
                    image,
                    page,
                    rng,
                    mode=capture,
                    keep=rng.uniform(*sorted(capture_cfg.keep_range)),
                )
            image, page, records = degradations.apply(image, page, rng)
            yield Sample(
                id=f"{index:08d}_{rendered.number:02d}",
                image=image,
                page=page,
                provenance=Provenance(
                    seed=seed,
                    font_path=str(getattr(typography.body.font, "path", "")),
                    font_size=int(getattr(typography.body.font, "size", 0)),
                    renderer="document",
                    background=background_kind,
                    shaper=type(self.renderer.text_renderer.shaper).__name__,
                    source=self.config.text.source.type,
                    template=source_name,
                    degradations=tuple(record.to_dict() for record in records),
                    extra={
                        "index": index,
                        "diacritics_kept": round(diacritics_kept, 3),
                        "page": rendered.number,
                        "preset": preset_name,
                        "paper": spec.width,
                        "dpi": spec.dpi,
                        "columns": spec.columns,
                        "direction": page.direction.value,
                        "overflow_blocks": dropped,
                        "overflow_words": dropped_words,
                        "capture": capture,
                    },
                ),
            )

    # -- sampling helpers --------------------------------------------------

    def _parse_document(self, rng: random.Random, direction: Direction | None):
        """A whole supplied document, parsed by an input adapter. Not yet chunked."""
        from ..adapters import parse_document

        raw = self.documents[rng.randrange(len(self.documents))]
        parsed = parse_document(raw, fmt=self.config.text.format, direction=direction)
        return str(parsed.metadata.get("source_format", "document")), parsed

    def _chunk_to_fit(self, parsed, rng: random.Random):
        """One chunk of `parsed`, sized within the configured budget.

        A random chunk rather than the first, so a long source document contributes its
        middle and its end to the corpus instead of the same opening page every time.
        """
        from ..adapters import sample_chunk

        limits = self.config.text.chunk
        return sample_chunk(
            parsed,
            rng,
            max_chars=rng.randint(*sorted(limits.max_chars)),
            max_blocks=rng.randint(*sorted(limits.max_blocks)),
        )

    def _capacity(self, spec: PageSpec, profile: PageProfile | None = None) -> int:
        """Roughly how many characters this sheet holds, for sizing a template's content.

        Measured against the type the page will actually be set in. Reading the size from
        `fonts.size_range` instead put a newspaper's 7.5 pt body at nearly twice its real
        size, and capacity falls with the *square* of the type size - so the hint came back
        about a third of the truth and a five-column broadsheet was handed two columns of
        copy.
        """
        if profile is not None:
            midpoint = sum(profile.body_pt) / 2.0
            body = max(6, int(round(midpoint * spec.dpi / 72.0)))
        else:
            body = max(6, self._body_size_range(spec.dpi)[1])
        return int(self._estimated_capacity(spec, body))

    @staticmethod
    def _estimated_capacity(spec: PageSpec, body_size: int) -> float:
        """Roughly how many characters this sheet holds at this type size."""
        # Arabic averages a little over half the body size per character at these sizes,
        # and a line box is about 1.35x the body size tall.
        chars_per_line = max(1.0, spec.column_width / (body_size * 0.55))
        lines = max(1.0, spec.content_height / (body_size * 1.35))
        return chars_per_line * lines * spec.columns

    def _page_spec_for(self, content, direction: Direction | None, rng: random.Random) -> PageSpec:
        """The sheet, out of several drawn from the configured distributions, this text fills best.

        Still sampled from the configured weights - papers, dpi, margins and columns keep
        the shape the config asked for - but a handful of candidates are drawn and the one
        whose capacity best matches the text is kept. A page is allowed to be part empty,
        the way a closing page is; it is not allowed to hold one paragraph in a corner.
        """
        target = rng.uniform(0.6, 0.9)
        chars = max(1, len(content.all_text))
        best: PageSpec | None = None
        best_error = float("inf")
        for _ in range(8):
            try:
                candidate = self._page_spec(self.config.page, direction, rng)
            except ValueError:
                continue  # margins left no room on this sheet; try another
            body_size = max(6, self._body_size_range(candidate.dpi)[1])
            # Every column must earn its place. Text that fills one column of a two-column
            # sheet leaves the other blank, and half an empty page beside a block of type
            # is the single most obviously synthetic thing a generator produces.
            while candidate.columns > 1 and chars / self._estimated_capacity(candidate, body_size) < 0.55:
                candidate = replace(candidate, columns=candidate.columns - 1)
            fill = chars / self._estimated_capacity(candidate, body_size)
            error = abs(fill - target)
            if error < best_error:
                best, best_error = candidate, error
        if best is None:
            return self._page_spec(self.config.page, direction, rng)
        return best

    def _template_names(self, rng: random.Random) -> list[str]:
        weights = self.config.templates.weights
        return [name for name in weights if weights[name] > 0 and name in self.registry]

    def _direction(self, rng: random.Random) -> Direction | None:
        return _DIRECTIONS.get(self.config.text.direction)

    def _spec_from_profile(
        self, profile: PageProfile, direction: Direction | None, rng: random.Random
    ) -> PageSpec:
        """The sheet a document of this type is actually printed on.

        Everything here is sampled *within* the type rather than across all types: a
        receipt varies in length and margin, but it is always an 80 mm roll set in one
        column at 7-9 point, because that is what a receipt is. The dpi range stays global
        because resolution is a property of the capture, not of the document.
        """
        page = self.config.page
        # A profile supplies the conventions of the type; an explicit config still wins.
        # Someone who writes `papers: {a4: 1.0}` is stating a requirement, not expressing a
        # preference the generator may overrule because it decided the page is a newspaper.
        pick = lambda value, fallback: fallback if value is None else value  # noqa: E731

        paper = _weighted_choice(pick(page.papers, profile.papers), rng)
        dpi = rng.randint(*sorted(page.dpi_range))
        columns = int(_weighted_choice(pick(page.columns, profile.columns), rng))
        landscape = rng.random() < pick(page.landscape_probability, profile.landscape_probability)

        # Margin as a fraction of the short edge, so it scales with the sheet instead of
        # swallowing a business card whole.
        short_edge_mm = min(PAPER_SIZES_MM[paper.lower()])
        if page.margin_mm_range is not None:
            margin = min(rng.uniform(*sorted(page.margin_mm_range)), short_edge_mm * 0.14)
        else:
            margin = short_edge_mm * rng.uniform(*sorted(profile.margin_ratio))

        body_size = self._profile_body_px(profile, dpi, rng)
        columns = self._fit_columns(paper, dpi, margin, landscape, columns, body_size)

        return PageSpec.from_paper(
            paper,
            dpi=dpi,
            margin_mm=margin,
            landscape=landscape,
            columns=columns,
            column_gap=max(10, int(dpi * (0.08 if columns > 3 else 0.15))),
            direction=direction or Direction.LTR,
            header_height=int(body_size * 2)
            if rng.random() < pick(page.header_probability, profile.header_probability)
            else 0,
            footer_height=int(body_size * 2)
            if rng.random() < pick(page.footer_probability, profile.footer_probability)
            else 0,
        )

    @staticmethod
    def _profile_body_px(profile: PageProfile, dpi: int, rng: random.Random) -> int:
        """The profile's point size, in pixels at this page's resolution.

        Points are physical and pixels are not, which is the whole reason the profile
        states type size in points: 10 pt body text has to come out the same physical size
        whether the page was captured at 150 dpi or 300.
        """
        points = rng.uniform(*sorted(profile.body_pt))
        return max(6, int(round(points * dpi / 72.0)))

    def _page_spec(self, page: PageConfig, direction: Direction | None, rng: random.Random) -> PageSpec:
        # Supplied documents have no genre to consult, so unset fields fall back to a
        # general-purpose spread rather than to any one type's conventions.
        paper = _weighted_choice(page.papers or {"a4": 4.0, "a5": 1.0, "letter": 1.0}, rng)
        dpi = rng.randint(*sorted(page.dpi_range))
        margin = rng.uniform(*sorted(page.margin_mm_range or (12.0, 28.0)))
        columns = int(_weighted_choice(page.columns or {1: 3.0, 2: 1.0}, rng))
        # Same rescaling as the body text, so a header band still clears its own type.
        body_size = self._body_size_range(dpi)[1]
        landscape = rng.random() < (page.landscape_probability or 0.05)

        # Margins are a proportion of the sheet in real documents, not an absolute. 30 mm
        # is generous on A4 and swallows an ID card whole - which is how a page ended up
        # with a column narrower than a single word.
        short_edge_mm = min(PAPER_SIZES_MM[paper.lower()])
        margin = min(margin, short_edge_mm * 0.14)

        # Column count follows from how wide a column would actually be. Three columns on
        # a receipt is not a rare layout, it is not a layout: no real document sets text
        # eight characters wide. Roughly half the body size per character.
        columns = self._fit_columns(paper, dpi, margin, landscape, columns, body_size)

        return PageSpec.from_paper(
            paper,
            dpi=dpi,
            margin_mm=margin,
            landscape=landscape,
            columns=columns,
            column_gap=max(12, int(dpi * 0.15)),
            direction=direction or Direction.LTR,
            header_height=int(body_size * 2)
            if rng.random() < (page.header_probability if page.header_probability is not None else 0.35)
            else 0,
            footer_height=int(body_size * 2)
            if rng.random() < (page.footer_probability if page.footer_probability is not None else 0.45)
            else 0,
        )

    #: Below this many characters a column stops being a column. Newspapers run narrow -
    #: high 20s - and nothing legible runs narrower.
    _MIN_COLUMN_CHARS = 26

    @staticmethod
    def _fit_columns(
        paper: str,
        dpi: int,
        margin_mm: float,
        landscape: bool,
        wanted: int,
        body_size: int,
    ) -> int:
        """The most columns this sheet can carry without setting text absurdly narrow."""
        width_mm, height_mm = PAPER_SIZES_MM[paper.lower()]
        if landscape:
            width_mm = height_mm
        page_px = width_mm / 25.4 * dpi
        content_px = page_px - 2 * (margin_mm / 25.4 * dpi)
        gap = max(12, int(dpi * 0.15))
        needed = SampleFactory._MIN_COLUMN_CHARS * body_size * 0.5

        for columns in range(wanted, 1, -1):
            if (content_px - gap * (columns - 1)) / columns >= needed:
                return columns
        return 1

    def _typography(self, content, rng: random.Random, dpi: int, ink=None, heavy: bool = False, profile=None):
        # Probe against everything the page will draw - table cells and list items
        # included - and against the *shaped* form, which is what actually reaches the
        # rasteriser.
        probe = content.all_text[:4000]
        faces = self.fonts.supporting(probe)
        if not faces:
            # Falling back to the whole pool would hand the document a face that cannot
            # draw its script at all, which is the exact failure coverage exists to
            # prevent. Fall back to the best face instead, and only that one.
            faces = (self.fonts.choose(probe, rng),)
        if profile is not None:
            sizes = sorted(
                (
                    max(6, int(round(min(profile.body_pt) * dpi / 72.0))),
                    max(7, int(round(max(profile.body_pt) * dpi / 72.0))),
                )
            )
        else:
            sizes = self._body_size_range(dpi)
        sampler = TypographySampler(faces, body_size_range=sizes)
        return sampler.sample(
            rng,
            handwritten=bool(content.metadata.get("handwritten")),
            ink=ink,
            heavy=heavy,
        )

    def _body_size_range(self, dpi: int) -> tuple[int, int]:
        """The configured pixel range, rescaled from `_REFERENCE_DPI` to this page's dpi.

        Keeps the *physical* type size stable across the dpi range, so a corpus that
        varies resolution varies sharpness and noise rather than how big the text looks.
        """
        low, high = sorted(self.config.fonts.size_range)
        factor = dpi / _REFERENCE_DPI
        return (max(6, round(low * factor)), max(7, round(high * factor)))

    @staticmethod
    def _table_style(
        rng: random.Random,
        ink: tuple[int, int, int] = (0, 0, 0),
        accent: tuple[int, int, int] | None = None,
        plain_stock: bool = True,
    ) -> TableStyle:
        # Weighted towards rules that show the columns. `HORIZONTAL` and `HEADER_ONLY` are
        # real styles - a booktabs table in a paper draws no verticals at all - but a page
        # of them reads as a stack of rows rather than as a table, and a layout model has
        # nothing in the pixels telling it where one column ends. Kept for variety, no
        # longer half the distribution.
        border = rng.choices(
            [BorderStyle.ALL, BorderStyle.OUTER, BorderStyle.HEADER_ONLY, BorderStyle.HORIZONTAL],
            weights=[7, 3, 1, 1],
            k=1,
        )[0]
        light_ink = sum(ink) / 3 > 128
        rule = (206, 208, 214) if light_ink else (40, 40, 40)
        # The table follows the document's second colour where there is one, so a navy
        # report has navy rules rather than a navy heading above a grey table.
        if accent is not None and not light_ink:
            rule = tint(accent, rng.uniform(0.0, 0.35))

        # A shaded header band is a second background *inside* the page, and that is the
        # reason it used to be off everywhere: grey banding over a photographed surface or
        # over aged stock reads as a paste-up rather than as a table. On plain stock it
        # reads as what it is - an invoice, a report, a price list - so it is allowed
        # there and only there, pale enough that the rules still carry the structure.
        header_fill = None
        zebra_fill = None
        if plain_stock and not light_ink and rng.random() < 0.45:
            base = accent or (40, 46, 60)
            header_fill = tint(base, rng.uniform(0.78, 0.9))
            if rng.random() < 0.4:
                zebra_fill = tint(base, rng.uniform(0.93, 0.97))

        return TableStyle(
            border=border,
            border_width=rng.choice([1, 1, 2]),
            border_color=rule,
            cell_padding=rng.randint(5, 12),
            header_fill=header_fill,
            zebra_fill=zebra_fill,
        )

    def _apply_diacritics(self, content, rng: random.Random):
        """Vary how vocalised this document is, sampled once for the whole page.

        Applied to the assembled content rather than at corpus load, so the fraction is a
        property of the *document* - which is how real vocalisation works, a whole text
        being marked or not - and so it can be recorded in provenance for an ablation.
        """
        settings = self.config.text.diacritics
        if settings.mode == "keep":
            return content, 1.0

        policy = DiacriticsPolicy(
            mode=DiacriticsMode(settings.mode),
            keep_range=tuple(settings.keep_range),
            mixed_weights=tuple(settings.mixed_weights),
        )
        kept = 1.0
        blocks = []
        for block in content.blocks:
            text, kept = apply_diacritics(block.text, policy, rng) if block.text else (block.text, kept)
            table = block.table
            if table is not None:
                table = type(table)(
                    table.rows,
                    table.cols,
                    tuple(
                        type(cell)(
                            cell.row,
                            cell.col,
                            apply_diacritics(cell.text, policy, rng)[0],
                            cell.bbox,
                            cell.row_span,
                            cell.col_span,
                            cell.is_header,
                            cell.lines,
                        )
                        for cell in table.cells
                    ),
                    table.has_header_row,
                )
            items = tuple(apply_diacritics(item, policy, rng)[0] for item in block.items)
            blocks.append(type(block)(block.type, text, items, table, dict(block.attributes)))
        return type(content)(tuple(blocks), content.direction, dict(content.metadata)), kept

    def _with_footer(self, content, rng: random.Random):
        footer_p = self.config.page.footer_probability
        footer_p = 0.45 if footer_p is None else footer_p
        if footer_p <= 0 or rng.random() > footer_p:
            return content
        template = rng.choice(["{page}", "- {page} -", "Page {page}", "صفحة {page}"])
        return type(content)(content.blocks, content.direction, {**content.metadata, "footer": template})


def load_corpus_texts(config: GenerationConfig) -> list[str]:
    """The rows of the corpus named by the config, normalised, in source order.

    Sources are read once per process and reduced to sentences; only the text is kept, so
    the memory cost is the corpus, not the dataset library's row objects.
    """
    source = config.text.source
    if source.type == "inline":
        # Normalised like any other source rather than returned as written. Skipping it
        # made `text.normalization` silently mean nothing for inline text, so a corpus
        # configured for Arabic-Indic numerals rendered western digits — and the config
        # said otherwise, which is worse than not having the setting.
        return _normalised(config, list(source.sentences))

    from ..datasets.loaders import CSVTextLoader, HuggingFaceTextLoader, ParquetTextLoader

    loaders = {
        "csv": CSVTextLoader,
        "parquet": ParquetTextLoader,
        "huggingface": HuggingFaceTextLoader,
    }
    loader = loaders[source.type](text_column=source.column, title_column=source.title_column)
    texts: list[str] = []
    try:
        if source.type == "huggingface":
            records = loader.iter_texts(str(source.path), split=source.split)
        else:
            records = iter(loader.load_texts(str(Path(source.path))))
        for record in records:
            texts.append(record["content"] if isinstance(record, dict) else str(record))
            if source.limit and len(texts) >= source.limit:
                break
    except Exception as exc:
        # A gated or offline corpus should degrade to the configured fallback rather than
        # killing a run, but it must say so: silently generating from six placeholder
        # sentences would look like success and produce a worthless dataset.
        if not source.sentences:
            raise
        logging.getLogger(__name__).warning(
            "Text source %r is unavailable (%s); falling back to %d inline sentences. "
            "The generated corpus will have almost no lexical variety.",
            source.path,
            exc,
            len(source.sentences),
        )
        return _normalised(config, list(source.sentences))

    if not texts:
        if not source.sentences:
            raise ValueError(f"Text source {source.path!r} yielded no usable rows")
        return _normalised(config, list(source.sentences))

    return _normalised(config, texts)


def _normalised(config: GenerationConfig, texts: list[str]) -> list[str]:
    """Apply `text.normalization` to every row, whatever the row came from.

    Every one of these rewrites the ground truth, which is why they all default to off —
    and why they have to be applied uniformly. A policy that holds for a CSV corpus and
    not for the inline fallback is a corpus whose labels depend on where the text was
    read from.
    """
    from ..text.normalization import NormalizationPolicy, NumeralSystem, normalize_text

    normalization = config.text.normalization
    # In "documents" mode the structure lives in the line breaks - a heading is a heading
    # because it starts its own line - so collapsing whitespace before an adapter reads
    # the text destroys the markup. Left on, a whole Markdown document collapsed onto one
    # line and its leading `#` swallowed every word as a 945-character title.
    collapse = normalization.collapse_whitespace and config.text.mode != "documents"
    policy = NormalizationPolicy(
        collapse_whitespace=collapse,
        strip_diacritics=normalization.strip_diacritics,
        strip_tatweel=normalization.strip_tatweel,
        unify_alef=normalization.unify_alef,
        unify_ya=normalization.unify_ya,
        numerals=NumeralSystem(normalization.numerals),
    )
    return [normalize_text(text, policy) for text in texts]


def build_text_provider(config: GenerationConfig) -> CorpusTextProvider:
    """The corpus reduced to a sentence pool, for the invent-a-document path."""
    return CorpusTextProvider(load_corpus_texts(config))


def build_documents(config: GenerationConfig) -> list[str]:
    """The corpus kept as whole documents, for the lay-out-what-I-gave-you path."""
    return load_corpus_texts(config)


def build_factory(config: GenerationConfig):
    """The factory `config.render.mode` asks for.

    Both factories expose `create(index) -> Iterator[Sample]` and nothing else, so every
    caller downstream — the runner, the writers, the quality gate, `ocrsmith preview` —
    works against either without knowing which it has. Adding a third renderer means
    adding a branch here and nowhere else.
    """
    if config.render.mode == "line":
        from .line_factory import LineSampleFactory

        return LineSampleFactory(config)
    return SampleFactory(config)


#: Below this many distinct sentences, a run is re-rendering the same words over and over
#: however many samples it produces. The figure is deliberately generous: a thousand
#: sentences is still a small corpus, and anything under it is unambiguously too few.
_THIN_CORPUS = 1000


def corpus_advice(config: GenerationConfig) -> str | None:
    """A sentence naming the variety the corpus does not have, or None when it is fine.

    The commonest way to waste a generation run is to leave `text.source.type` at
    `inline` and generate a hundred thousand pages of six sentences. Nothing fails: the
    pages are well laid out, the labels are exact, the quality gate passes every one of
    them, and the dataset teaches a model a vocabulary of forty words. The only visible
    symptom arrives weeks later as a model that reads the training distribution
    perfectly and nothing else.

    Layout realism is cheap to add and lexical variety is not, so this is the one thing
    worth saying out loud before a long run rather than after it.
    """
    source = config.text.source
    if source.type == "inline":
        return (
            f"text.source.type is 'inline': every sample will be built from the same "
            f"{len(source.sentences)} built-in sentence(s). That exercises the layout and "
            f"the capture conditions, but the corpus will have no lexical variety at all. "
            f"Point text.source at a real corpus before generating anything you intend to "
            f"train on - see examples/configs/darija_scan.yaml."
        )
    try:
        available = len(load_corpus_texts(config))
    except Exception:
        return None
    if available < _THIN_CORPUS:
        return (
            f"The corpus yielded {available} text(s). Below roughly {_THIN_CORPUS} the same "
            f"sentences recur across the dataset, so samples stop being independent even "
            f"though they look it."
        )
    return None
