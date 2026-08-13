"""Dataset statistics.

A synthetic corpus is defined by its distributions, so the only way to know what you built
is to measure them. These counters answer the questions that decide whether a dataset is
usable: is any template dominating, did the RTL share come out where it was meant to, how
many characters has the model actually seen, and is the character inventory wide enough to
cover the alphabet you care about.

The accumulator takes one sample at a time and holds nothing but counters, so it can run
inside a generation pass over an arbitrarily large corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..domain import Sample

__all__ = ["DatasetStats", "scan_jsonl"]


def _vocalisation_bucket(text: str) -> str:
    """Coarse label for how diacritised a page is.

    Bucketed rather than averaged because the interesting question is compositional - what
    share of the corpus is bare, partial, or fully marked - not what the mean is.
    """
    from ..text.diacritics import diacritic_ratio

    ratio = diacritic_ratio(text)
    if ratio <= 0.01:
        return "none"
    if ratio < 0.35:
        return "partial"
    return "full"


@dataclass
class DatasetStats:
    """Streaming counters over a corpus."""

    pages: int = 0
    documents: set = field(default_factory=set)
    words: int = 0
    characters: int = 0
    lines: int = 0
    templates: Counter = field(default_factory=Counter)
    presets: Counter = field(default_factory=Counter)
    directions: Counter = field(default_factory=Counter)
    region_types: Counter = field(default_factory=Counter)
    backgrounds: Counter = field(default_factory=Counter)
    fonts: Counter = field(default_factory=Counter)
    characters_seen: Counter = field(default_factory=Counter)
    page_sizes: Counter = field(default_factory=Counter)
    #: How vocalised each page was, bucketed. Arabic OCR lives or dies on this.
    diacritics: Counter = field(default_factory=Counter)
    _line_heights: list = field(default_factory=list)

    def add(self, sample: Sample) -> DatasetStats:
        page = sample.page
        provenance = sample.provenance

        self.pages += 1
        if provenance.extra.get("index") is not None:
            self.documents.add(provenance.extra["index"])
        self.page_sizes[f"{page.width}x{page.height}"] += 1
        self.directions[page.direction.value] += 1
        if provenance.template:
            self.templates[provenance.template] += 1
        if provenance.extra.get("preset"):
            self.presets[provenance.extra["preset"]] += 1
        if provenance.background:
            self.backgrounds[provenance.background] += 1
        if provenance.font_path:
            self.fonts[Path(provenance.font_path).name] += 1

        self.diacritics[_vocalisation_bucket(page.text)] += 1
        for region in page.regions:
            self.region_types[region.type.value] += 1
        for line in page.iter_lines():
            self.lines += 1
            self._line_heights.append(line.bbox.height)
            self.words += len(line.words) or len(line.text.split())
            self.characters += len(line.text)
            self.characters_seen.update(line.text)
        return self

    def add_record(self, record: dict) -> DatasetStats:
        """Accumulate from a serialised JSONL record rather than a live `Sample`."""
        page = record.get("page", {})
        provenance = record.get("provenance", {})
        extra = provenance.get("extra", {})

        self.pages += 1
        if extra.get("index") is not None:
            self.documents.add(extra["index"])
        self.page_sizes[f"{page.get('width')}x{page.get('height')}"] += 1
        self.directions[page.get("direction", "ltr")] += 1
        if provenance.get("template"):
            self.templates[provenance["template"]] += 1
        if extra.get("preset"):
            self.presets[extra["preset"]] += 1
        if provenance.get("background"):
            self.backgrounds[provenance["background"]] += 1
        if provenance.get("font_path"):
            self.fonts[Path(provenance["font_path"]).name] += 1

        self.diacritics[_vocalisation_bucket(record.get("text", ""))] += 1
        for region in page.get("regions", []):
            self.region_types[region.get("type", "unknown")] += 1
            for line in region.get("lines", []):
                self.lines += 1
                x0, y0, x1, y1 = line["bbox"]
                self._line_heights.append(abs(y1 - y0))
                text = line.get("text", "")
                self.words += len(line.get("words", [])) or len(text.split())
                self.characters += len(text)
                self.characters_seen.update(text)
        return self

    # -- summaries ---------------------------------------------------------

    @property
    def line_height_percentiles(self) -> dict[str, float]:
        """Where the small text is. The p5 value is the one that predicts OCR failure."""
        if not self._line_heights:
            return {}
        ordered = sorted(self._line_heights)

        def percentile(fraction: float) -> float:
            index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
            return round(ordered[index], 1)

        return {"p5": percentile(0.05), "p50": percentile(0.5), "p95": percentile(0.95)}

    @property
    def alphabet_size(self) -> int:
        return len(self.characters_seen)

    def to_dict(self) -> dict:
        return {
            "pages": self.pages,
            "documents": len(self.documents),
            "lines": self.lines,
            "words": self.words,
            "characters": self.characters,
            "alphabet_size": self.alphabet_size,
            "line_height_px": self.line_height_percentiles,
            "templates": dict(self.templates.most_common()),
            "degradation_presets": dict(self.presets.most_common()),
            "directions": dict(self.directions.most_common()),
            "region_types": dict(self.region_types.most_common()),
            "backgrounds": dict(self.backgrounds.most_common()),
            "vocalisation": dict(self.diacritics.most_common()),
            "top_fonts": dict(self.fonts.most_common(10)),
            "top_page_sizes": dict(self.page_sizes.most_common(5)),
            "rarest_characters": [char for char, _ in self.characters_seen.most_common()[-15:]],
        }

    def to_markdown(self) -> str:
        """A dataset card fragment, ready to paste into a model or dataset README."""
        data = self.to_dict()
        lines = [
            "## Dataset statistics",
            "",
            f"- **Pages**: {data['pages']:,} across {data['documents']:,} documents",
            f"- **Lines**: {data['lines']:,}",
            f"- **Words**: {data['words']:,}",
            f"- **Characters**: {data['characters']:,} ({data['alphabet_size']} distinct)",
        ]
        if data["line_height_px"]:
            heights = data["line_height_px"]
            lines.append(
                f"- **Line height (px)**: p5 {heights['p5']}, median {heights['p50']}, p95 {heights['p95']}"
            )
        for title, key in (
            ("Templates", "templates"),
            ("Capture conditions", "degradation_presets"),
            ("Reading direction", "directions"),
            ("Region types", "region_types"),
            ("Vocalisation", "vocalisation"),
        ):
            if data[key]:
                lines.append("")
                lines.append(f"### {title}")
                lines.append("")
                total = sum(data[key].values()) or 1
                for name, count in data[key].items():
                    lines.append(f"- `{name}`: {count:,} ({count / total:.1%})")
        return "\n".join(lines) + "\n"


def scan_jsonl(directory: str | Path) -> DatasetStats:
    """Accumulate statistics from the JSONL shards written by `JsonlSink`."""
    stats = DatasetStats()
    for path in sorted(Path(directory).glob("annotations-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    stats.add_record(json.loads(line))
    return stats
