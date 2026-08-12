"""Benchmark harness.

Generating data and evaluating on it belong in the same repository: a benchmark whose
ground truth was produced by a different codebase than the one under test is where silent
label conventions hide. The harness reads a generated dataset's annotations, pairs them
with a model's predictions by sample id, and reports per-sample and aggregate scores.

Predictions are supplied as plain `{sample_id: text}` or `{sample_id: {...}}` mappings, so
any model can be scored without importing anything from it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from .metrics import cer, normalized_edit_similarity, wer

__all__ = ["EvaluationReport", "SampleScore", "evaluate", "load_references"]


@dataclass(frozen=True, slots=True)
class SampleScore:
    """One sample's scores."""

    sample_id: str
    cer: float
    wer: float
    similarity: float
    reference_characters: int

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "similarity": round(self.similarity, 4),
            "reference_characters": self.reference_characters,
        }


@dataclass
class EvaluationReport:
    """Aggregate scores over a benchmark split."""

    scores: list[SampleScore] = field(default_factory=list)
    missing_predictions: list[str] = field(default_factory=list)
    #: Which ground-truth field was scored: "text", "markdown" or "html".
    target: str = "text"
    ignore_diacritics: bool = False

    @property
    def sample_count(self) -> int:
        return len(self.scores)

    @property
    def macro_cer(self) -> float:
        """Mean CER over samples: every page counts the same."""
        return mean(score.cer for score in self.scores) if self.scores else 0.0

    @property
    def micro_cer(self) -> float:
        """Length-weighted CER: every character counts the same.

        Reported alongside the macro figure because they disagree exactly when the corpus
        mixes short captions with dense pages — which is the interesting case.
        """
        total = sum(score.reference_characters for score in self.scores)
        if not total:
            return 0.0
        return sum(score.cer * score.reference_characters for score in self.scores) / total

    @property
    def macro_wer(self) -> float:
        return mean(score.wer for score in self.scores) if self.scores else 0.0

    @property
    def mean_similarity(self) -> float:
        return mean(score.similarity for score in self.scores) if self.scores else 0.0

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "ignore_diacritics": self.ignore_diacritics,
            "samples": self.sample_count,
            "missing_predictions": len(self.missing_predictions),
            "cer_macro": round(self.macro_cer, 4),
            "cer_micro": round(self.micro_cer, 4),
            "wer_macro": round(self.macro_wer, 4),
            "similarity": round(self.mean_similarity, 4),
        }

    def worst(self, count: int = 10) -> list[SampleScore]:
        """The samples a model failed hardest on — where debugging should start."""
        return sorted(self.scores, key=lambda score: -score.cer)[:count]

    def to_markdown(self) -> str:
        data = self.to_dict()
        return "\n".join(
            [
                "## Evaluation",
                "",
                f"- **Samples scored**: {data['samples']:,}"
                + (f" ({data['missing_predictions']} missing)" if data["missing_predictions"] else ""),
                f"- **Target**: `{data['target']}`"
                + (" (diacritics ignored)" if data["ignore_diacritics"] else ""),
                f"- **CER**: {data['cer_macro']:.4f} macro / {data['cer_micro']:.4f} micro",
                f"- **WER**: {data['wer_macro']:.4f}",
                f"- **Similarity**: {data['similarity']:.4f}",
                "",
            ]
        )


def load_references(directory: str | Path, target: str = "text") -> dict[str, str]:
    """Read `{sample_id: ground_truth}` from a generated dataset's JSONL shards."""
    references: dict[str, str] = {}
    for path in sorted(Path(directory).glob("annotations-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                references[record["id"]] = record.get(target, "")
    return references


def _as_text(value) -> str:
    if isinstance(value, Mapping):
        for key in ("text", "markdown", "html", "prediction", "output"):
            if key in value:
                return str(value[key])
        return ""
    return str(value)


def evaluate(
    references: Mapping[str, str] | Iterable[tuple[str, str]],
    predictions: Mapping[str, object],
    *,
    target: str = "text",
    ignore_diacritics: bool = False,
    ignore_case: bool = False,
) -> EvaluationReport:
    """Score `predictions` against `references`, pairing them by sample id.

    A sample with no prediction is counted as a total miss rather than skipped: silently
    dropping the ones a model refused to answer flatters it.
    """
    if not isinstance(references, Mapping):
        references = dict(references)

    report = EvaluationReport(target=target, ignore_diacritics=ignore_diacritics)
    for sample_id, reference in references.items():
        raw = predictions.get(sample_id)
        if raw is None:
            report.missing_predictions.append(sample_id)
        hypothesis = "" if raw is None else _as_text(raw)
        report.scores.append(
            SampleScore(
                sample_id=sample_id,
                cer=cer(reference, hypothesis, ignore_diacritics=ignore_diacritics, ignore_case=ignore_case),
                wer=wer(reference, hypothesis, ignore_diacritics=ignore_diacritics, ignore_case=ignore_case),
                similarity=normalized_edit_similarity(reference, hypothesis),
                reference_characters=len(reference),
            )
        )
    return report
