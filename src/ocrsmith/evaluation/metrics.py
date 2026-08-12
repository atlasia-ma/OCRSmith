"""OCR evaluation metrics.

A dataset is only worth as much as the benchmark that measures a model trained on it, and
a benchmark built from someone else's metric implementation is a benchmark you cannot
debug. These are the standard measures, implemented directly:

* **CER / WER** — edit distance over characters or words, normalised by reference length.
* **NED** — normalised edit distance, reported as a similarity in `[0, 1]`.
* **TEDS-style table similarity** — structure and cell content compared together, which is
  what distinguishes a table model that found the grid from one that only read the text.
* **Detection precision / recall / F1** at an IoU threshold, for box prediction.

Arabic needs one extra decision that Latin-only implementations get wrong by default:
whether diacritics count. Both answers are defensible, so it is an explicit argument
rather than a silent convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..domain.geometry import BBox
from ..text.normalization import strip_diacritics

__all__ = [
    "DetectionScore",
    "cer",
    "detection_score",
    "levenshtein",
    "normalized_edit_similarity",
    "table_similarity",
    "wer",
]


def levenshtein(reference: Sequence, hypothesis: Sequence) -> int:
    """Edit distance between two sequences, in O(min(n, m)) memory."""
    if reference == hypothesis:
        return 0
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_item != hyp_item),  # substitution
                )
            )
        previous = current
    return previous[-1]


def _prepare(text: str, *, ignore_diacritics: bool, ignore_case: bool) -> str:
    if ignore_diacritics:
        text = strip_diacritics(text)
    if ignore_case:
        text = text.lower()
    return " ".join(text.split())


def cer(
    reference: str, hypothesis: str, *, ignore_diacritics: bool = False, ignore_case: bool = False
) -> float:
    """Character error rate. 0.0 is perfect; values above 1.0 are possible.

    An empty reference scores 0.0 against an empty hypothesis and 1.0 otherwise, rather
    than dividing by zero — the convention that keeps a corpus average meaningful.
    """
    ref = _prepare(reference, ignore_diacritics=ignore_diacritics, ignore_case=ignore_case)
    hyp = _prepare(hypothesis, ignore_diacritics=ignore_diacritics, ignore_case=ignore_case)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(
    reference: str, hypothesis: str, *, ignore_diacritics: bool = False, ignore_case: bool = False
) -> float:
    """Word error rate, over whitespace-delimited tokens."""
    ref = _prepare(reference, ignore_diacritics=ignore_diacritics, ignore_case=ignore_case).split()
    hyp = _prepare(hypothesis, ignore_diacritics=ignore_diacritics, ignore_case=ignore_case).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def normalized_edit_similarity(reference: str, hypothesis: str) -> float:
    """1 - normalised edit distance, clamped to `[0, 1]`. Higher is better."""
    ref, hyp = " ".join(reference.split()), " ".join(hypothesis.split())
    if not ref and not hyp:
        return 1.0
    longest = max(len(ref), len(hyp))
    return max(0.0, 1.0 - levenshtein(ref, hyp) / longest) if longest else 1.0


def table_similarity(reference, hypothesis) -> float:
    """TEDS-style similarity between two `Table` objects, in `[0, 1]`.

    Structure and content are weighted equally: a prediction that recovers the grid but
    garbles the text, and one that reads every cell but collapses the rows, are equally
    wrong, and a metric that hides either is not measuring table understanding.
    """
    if reference is None or hypothesis is None:
        return 0.0

    cells = max(reference.rows * reference.cols, hypothesis.rows * hypothesis.cols)
    if cells == 0:
        return 1.0

    shape_penalty = abs(reference.rows - hypothesis.rows) + abs(reference.cols - hypothesis.cols)
    structure = max(0.0, 1.0 - shape_penalty / (reference.rows + reference.cols or 1))

    total = 0.0
    compared = 0
    for row in range(reference.rows):
        for col in range(reference.cols):
            expected = reference.cell_at(row, col)
            if expected is None:
                continue
            compared += 1
            predicted = hypothesis.cell_at(row, col)
            total += 0.0 if predicted is None else normalized_edit_similarity(expected.text, predicted.text)
    content = total / compared if compared else 1.0
    return round((structure + content) / 2, 6)


@dataclass(frozen=True, slots=True)
class DetectionScore:
    """Precision, recall and F1 at one IoU threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    iou_threshold: float

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "iou_threshold": self.iou_threshold,
        }


def detection_score(
    reference: Sequence[BBox], predicted: Sequence[BBox], *, iou_threshold: float = 0.5
) -> DetectionScore:
    """Greedy one-to-one matching of predicted boxes to reference boxes.

    Greedy rather than optimal: it is what the standard detection benchmarks do, and
    matching the convention matters more here than squeezing out the last fraction of a
    point.
    """
    unmatched = list(reference)
    true_positives = 0
    for box in predicted:
        best_index, best_iou = -1, 0.0
        for index, candidate in enumerate(unmatched):
            score = box.iou(candidate)
            if score > best_iou:
                best_index, best_iou = index, score
        if best_index >= 0 and best_iou >= iou_threshold:
            unmatched.pop(best_index)
            true_positives += 1
    return DetectionScore(
        true_positives=true_positives,
        false_positives=len(predicted) - true_positives,
        false_negatives=len(unmatched),
        iou_threshold=iou_threshold,
    )
