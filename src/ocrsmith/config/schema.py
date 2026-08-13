"""Generation configuration.

One validated object describes an entire corpus: where the text comes from, how pages are
shaped, which document genres appear and in what proportion, which capture conditions are
simulated, and where the result is written. Everything that varies per sample is expressed
as a *distribution* (a range or a weight map) rather than a fixed value, because a corpus
is defined by its distributions and hard-coding a value silently narrows the dataset.

The config is a plain pydantic model with no runtime objects in it, so it round-trips
through `model_dump()` and can be handed to a worker process intact — which is what makes
every sample reproducible from `(seed, index)` alone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "FALLBACK_SENTENCES",
    "BackgroundConfig",
    "DegradationConfig",
    "DiacriticsConfig",
    "FontConfig",
    "GenerationConfig",
    "OutputConfig",
    "PageConfig",
    "QualityConfig",
    "RunConfig",
    "TemplateConfig",
    "TextConfig",
    "TextSourceConfig",
]

IntRange = tuple[int, int]
FloatRange = tuple[float, float]

#: A handful of sentences so that `GenerationConfig()` is valid and a fresh clone can
#: render something without reaching the network. Also the fallback when a configured
#: corpus turns out to be unreachable mid-run.
FALLBACK_SENTENCES = [
    "المغرب بلد يقع في شمال إفريقيا ويطل على المحيط الأطلسي والبحر الأبيض المتوسط.",
    "تهدف أطلسيا إلى بناء نماذج ذكاء اصطناعي مفتوحة المصدر للغة الدارجة المغربية.",
    "يحتوي هذا التقرير على جداول وأرقام ومعلومات إضافية مفيدة للقارئ المهتم.",
    "في سنة 2024 أطلقت المجموعة نموذج OCR جديد يدعم اللغتين العربية والفرنسية.",
    "تعتمد الطريقة المقترحة على توليد بيانات اصطناعية متنوعة وواقعية قدر الإمكان.",
    "Synthetic pages are only useful when their labels describe the pixels exactly.",
]


class FontConfig(BaseModel):
    """Where fonts come from and how big they are drawn."""

    paths: list[str] = Field(default_factory=lambda: ["assets/fonts"])
    #: Body text size in pixels; every other role is derived from it.
    size_range: IntRange = (18, 30)
    #: Reject a (font, text) pair when the face cannot draw every character, rather than
    #: emitting tofu boxes that contradict the label.
    require_full_coverage: bool = True
    #: Optional filename substrings to keep, e.g. ["Noto", "Amiri"].
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @field_validator("size_range")
    @classmethod
    def _ordered(cls, value: IntRange) -> IntRange:
        low, high = value
        if low <= 0:
            raise ValueError("font sizes must be positive")
        return (min(low, high), max(low, high))


class TextSourceConfig(BaseModel):
    """The corpus the documents are written from."""

    type: Literal["csv", "parquet", "huggingface", "inline"] = "inline"
    path: str | None = None
    column: str = "text"
    title_column: str | None = None
    split: str = "train"
    name: str | None = None
    #: Used when `type` is "inline"; also the fallback when a source yields nothing.
    sentences: list[str] = Field(default_factory=lambda: list(FALLBACK_SENTENCES))
    #: Cap on how many rows are read, so a smoke run does not download a whole corpus.
    limit: int | None = None

    @model_validator(mode="after")
    def _path_required_for_file_sources(self) -> TextSourceConfig:
        if self.type in ("csv", "parquet", "huggingface") and not self.path:
            raise ValueError(f"text source of type {self.type!r} needs a path")
        if self.type == "inline" and not self.sentences:
            raise ValueError("inline text source needs at least one sentence")
        return self


class DiacriticsConfig(BaseModel):
    """How vocalised the corpus is.

    Real Arabic is *partially* diacritised and the proportion varies by genre, so a corpus
    that is uniformly marked or uniformly bare teaches a model to expect that uniformity.
    Marks are only ever removed, never invented: adding them would fabricate ground truth.
    """

    mode: Literal["keep", "strip", "partial", "mixed"] = "keep"
    keep_range: tuple[float, float] = (0.1, 0.6)
    #: Weights for "mixed": fully marked, partially marked, unmarked.
    mixed_weights: tuple[float, float, float] = (0.15, 0.25, 0.60)


class NormalizationConfig(BaseModel):
    """Label-affecting text transforms. Every one of these changes the ground truth."""

    strip_diacritics: bool = False
    strip_tatweel: bool = False
    unify_alef: bool = False
    unify_ya: bool = False
    numerals: Literal["keep", "western", "arabic_indic", "eastern_arabic_indic"] = "keep"
    collapse_whitespace: bool = True


class TextConfig(BaseModel):
    source: TextSourceConfig = Field(default_factory=TextSourceConfig)
    diacritics: DiacriticsConfig = Field(default_factory=DiacriticsConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    #: "auto" reads the direction from the text itself.
    direction: Literal["auto", "rtl", "ltr"] = "auto"


class BackgroundConfig(BaseModel):
    """Paper appearance, sampled per document."""

    kinds: dict[str, float] = Field(
        default_factory=lambda: {"paper": 3.0, "solid": 1.0, "gradient": 0.5, "image": 0.0}
    )
    #: Directory or file paths used by the "image" kind.
    image_paths: list[str] = Field(default_factory=list)
    #: Base paper tint sampled between these two colours.
    tint_range: tuple[tuple[int, int, int], tuple[int, int, int]] = ((238, 234, 226), (255, 255, 255))

    @model_validator(mode="after")
    def _needs_one_kind(self) -> BackgroundConfig:
        if not any(weight > 0 for weight in self.kinds.values()):
            raise ValueError("at least one background kind must have a positive weight")
        if self.kinds.get("image", 0) > 0 and not self.image_paths:
            raise ValueError("background kind 'image' needs image_paths")
        return self


class PageConfig(BaseModel):
    """Page geometry, sampled per document."""

    papers: dict[str, float] = Field(default_factory=lambda: {"a4": 4.0, "a5": 1.0, "letter": 1.0})
    dpi_range: IntRange = (110, 200)
    margin_mm_range: FloatRange = (12.0, 28.0)
    columns: dict[int, float] = Field(default_factory=lambda: {1: 3.0, 2: 1.0})
    landscape_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    header_probability: float = Field(default=0.35, ge=0.0, le=1.0)
    footer_probability: float = Field(default=0.45, ge=0.0, le=1.0)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    #: Hard ceiling on how many pages one document may occupy.
    max_pages: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _weights_are_usable(self) -> PageConfig:
        if not any(weight > 0 for weight in self.papers.values()):
            raise ValueError("at least one paper size must have a positive weight")
        if not any(weight > 0 for weight in self.columns.values()):
            raise ValueError("at least one column count must have a positive weight")
        return self


class TemplateConfig(BaseModel):
    """Which document genres appear, and how often."""

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "article": 3.0,
            "report": 2.0,
            "newspaper": 1.5,
            "paper": 1.5,
            "letter": 1.0,
            "form": 1.0,
            "invoice": 1.0,
        }
    )

    @model_validator(mode="after")
    def _at_least_one(self) -> TemplateConfig:
        if not any(weight > 0 for weight in self.weights.values()):
            raise ValueError("at least one template must have a positive weight")
        return self


class DegradationConfig(BaseModel):
    """Capture conditions, weighted so a corpus can be balanced deliberately."""

    presets: dict[str, float] = Field(
        default_factory=lambda: {"clean": 1.0, "scan": 4.0, "photo": 3.0, "fax": 0.5, "archive": 1.0}
    )

    @model_validator(mode="after")
    def _at_least_one(self) -> DegradationConfig:
        if not any(weight > 0 for weight in self.presets.values()):
            raise ValueError("at least one degradation preset must have a positive weight")
        return self


class OutputConfig(BaseModel):
    """Where the dataset is written and in what shape."""

    dir: str = "outputs/dataset"
    format: Literal["jsonl", "parquet", "webdataset", "coco", "paddleocr", "chat"] = "jsonl"
    image_format: Literal["png", "jpeg", "webp"] = "png"
    image_quality: int = Field(default=92, ge=1, le=100)
    images_subdir: str = "images"
    #: Samples per shard. Sharding is what makes a large run resumable and parallel.
    shard_size: int = Field(default=1000, ge=1)
    #: Also emit per-line crops for recognition training.
    line_crops: bool = False
    #: Fraction of samples routed to a held-out split.
    eval_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)


class QualityConfig(BaseModel):
    """Whether generated samples are checked before they reach the dataset."""

    #: Run the default validators and drop samples that fail.
    enabled: bool = True
    #: Give up if more than this fraction of samples is rejected — a high rejection rate
    #: means the configuration is producing unusable pages, not that the data is unlucky.
    max_rejection_rate: float = Field(default=0.5, ge=0.0, le=1.0)


class RunConfig(BaseModel):
    """How much to generate, and with how many processes."""

    num_samples: int = Field(default=100, ge=1)
    workers: int = Field(default=1, ge=1)
    #: Skip samples whose index is below this, so an interrupted run can continue.
    start_index: int = Field(default=0, ge=0)
    #: Give up on a document after this many consecutive failures rather than spinning.
    max_consecutive_failures: int = Field(default=50, ge=1)


class GenerationConfig(BaseModel):
    """A complete corpus specification."""

    seed: int | None = 1234
    fonts: FontConfig = Field(default_factory=FontConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    page: PageConfig = Field(default_factory=PageConfig)
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    degradations: DegradationConfig = Field(default_factory=DegradationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    def sample_seed(self, index: int) -> int:
        """Deterministic per-sample seed.

        Derived rather than drawn, so sample 8_412 of a run is regenerable on its own
        without replaying the 8_411 before it.
        """
        base = 0 if self.seed is None else int(self.seed)
        return (base * 1_000_003 + index * 2_654_435_761) % (2**32)
