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
from pathlib import Path

from ..config.schema import GenerationConfig, PageConfig
from ..core.backgrounds import BackgroundSampler
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
from ..core.fonts import FontPool
from ..domain import Provenance, Sample
from ..text.diacritics import DiacriticsMode, DiacriticsPolicy, apply_diacritics
from ..text.script import Direction

__all__ = ["SampleFactory"]

_DIRECTIONS = {"rtl": Direction.RTL, "ltr": Direction.LTR}


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

        template = self.registry.sample(rng, self._template_names(rng))
        direction = self._direction(rng)
        content = template.build(self.text, rng, direction=direction)
        content = self._with_footer(content, rng)
        content, diacritics_kept = self._apply_diacritics(content, rng)

        spec = self._page_spec(self.config.page, direction, rng)
        typography = self._typography(content, rng)
        background, background_kind = self.backgrounds.sample(rng)
        preset_name = _weighted_choice(self.config.degradations.presets, rng)
        degradations = build_preset(preset_name)

        pages = self.renderer.iter_pages(
            content,
            spec,
            typography,
            background=background,
            table_style=self._table_style(rng),
            rng=rng,
            max_pages=self.config.page.max_pages,
        )

        for rendered in pages:
            image, page, records = degradations.apply(rendered.image.convert("RGB"), rendered.page, rng)
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
                    template=template.name,
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
                    },
                ),
            )

    # -- sampling helpers --------------------------------------------------

    def _template_names(self, rng: random.Random) -> list[str]:
        weights = self.config.templates.weights
        return [name for name in weights if weights[name] > 0 and name in self.registry]

    def _direction(self, rng: random.Random) -> Direction | None:
        return _DIRECTIONS.get(self.config.text.direction)

    def _page_spec(self, page: PageConfig, direction: Direction | None, rng: random.Random) -> PageSpec:
        paper = _weighted_choice(page.papers, rng)
        dpi = rng.randint(*sorted(page.dpi_range))
        margin = rng.uniform(*sorted(page.margin_mm_range))
        columns = int(_weighted_choice(page.columns, rng))
        body_size = self.config.fonts.size_range[1]
        return PageSpec.from_paper(
            paper,
            dpi=dpi,
            margin_mm=margin,
            landscape=rng.random() < page.landscape_probability,
            columns=columns,
            column_gap=max(12, int(dpi * 0.15)),
            direction=direction or Direction.LTR,
            header_height=int(body_size * 2) if rng.random() < page.header_probability else 0,
            footer_height=int(body_size * 2) if rng.random() < page.footer_probability else 0,
        )

    def _typography(self, content, rng: random.Random):
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
        sampler = TypographySampler(faces, body_size_range=tuple(self.config.fonts.size_range))
        return sampler.sample(rng)

    @staticmethod
    def _table_style(rng: random.Random) -> TableStyle:
        border = rng.choices(
            [BorderStyle.ALL, BorderStyle.HORIZONTAL, BorderStyle.OUTER, BorderStyle.HEADER_ONLY],
            weights=[4, 2, 1, 2],
            k=1,
        )[0]
        shade = rng.random()
        return TableStyle(
            border=border,
            border_width=rng.choice([1, 1, 2]),
            cell_padding=rng.randint(5, 12),
            header_fill=(226, 230, 236) if shade < 0.35 else None,
            zebra_fill=(243, 245, 248) if shade > 0.75 else None,
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
        if self.config.page.footer_probability <= 0 or rng.random() > self.config.page.footer_probability:
            return content
        template = rng.choice(["{page}", "- {page} -", "Page {page}", "صفحة {page}"])
        return type(content)(content.blocks, content.direction, {**content.metadata, "footer": template})


def build_text_provider(config: GenerationConfig) -> CorpusTextProvider:
    """Load the corpus named by the config into a `CorpusTextProvider`.

    Sources are read once per process and reduced to sentences; only the text is kept, so
    the memory cost is the corpus, not the dataset library's row objects.
    """
    source = config.text.source
    if source.type == "inline":
        return CorpusTextProvider(source.sentences)

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
        return CorpusTextProvider(source.sentences)

    if not texts:
        if not source.sentences:
            raise ValueError(f"Text source {source.path!r} yielded no usable rows")
        return CorpusTextProvider(source.sentences)

    from ..text.normalization import NormalizationPolicy, NumeralSystem, normalize_text

    normalization = config.text.normalization
    policy = NormalizationPolicy(
        collapse_whitespace=normalization.collapse_whitespace,
        strip_diacritics=normalization.strip_diacritics,
        strip_tatweel=normalization.strip_tatweel,
        unify_alef=normalization.unify_alef,
        unify_ya=normalization.unify_ya,
        numerals=NumeralSystem(normalization.numerals),
    )
    return CorpusTextProvider(normalize_text(text, policy) for text in texts)
