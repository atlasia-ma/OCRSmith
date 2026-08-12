"""Quality assurance: what the corpus contains, and whether it is usable."""

from .stats import DatasetStats, scan_jsonl
from .validators import (
    BoxesInsidePage,
    LegibleLineHeight,
    MinContrast,
    MinInkCoverage,
    NonEmptyText,
    NoOverlappingRegions,
    ValidationPipeline,
    ValidationReport,
    Validator,
    Verdict,
    default_validators,
)

__all__ = [
    "BoxesInsidePage",
    "DatasetStats",
    "LegibleLineHeight",
    "MinContrast",
    "MinInkCoverage",
    "NoOverlappingRegions",
    "NonEmptyText",
    "ValidationPipeline",
    "ValidationReport",
    "Validator",
    "Verdict",
    "default_validators",
    "scan_jsonl",
]
