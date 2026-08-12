"""Evaluation: metrics and a benchmark harness for datasets generated here."""

from .harness import EvaluationReport, SampleScore, evaluate, load_references
from .metrics import (
    DetectionScore,
    cer,
    detection_score,
    levenshtein,
    normalized_edit_similarity,
    table_similarity,
    wer,
)

__all__ = [
    "DetectionScore",
    "EvaluationReport",
    "SampleScore",
    "cer",
    "detection_score",
    "evaluate",
    "levenshtein",
    "load_references",
    "normalized_edit_similarity",
    "table_similarity",
    "wer",
]
