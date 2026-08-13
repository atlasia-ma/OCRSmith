"""Sample validation.

Synthetic data fails quietly. A blank page, a label describing text that a degradation
erased, a box a pixel outside the canvas — none of these raise, and all of them teach a
model something false. Validation turns those into a decision the pipeline can act on:
reject the sample, or record why it was kept.

Each validator answers one question and explains itself, so a rejection rate is
diagnosable ("38% failed MinContrast") rather than merely alarming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..domain import RegionType, Sample

__all__ = [
    "BoxesInsidePage",
    "LegibleLineHeight",
    "MinContrast",
    "MinInkCoverage",
    "NonEmptyText",
    "NoOverlappingRegions",
    "ValidationPipeline",
    "ValidationReport",
    "Validator",
    "Verdict",
    "default_validators",
]


@dataclass(frozen=True, slots=True)
class Verdict:
    """One validator's answer, with the measurement that produced it."""

    validator: str
    passed: bool
    reason: str = ""
    value: float | None = None

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Every verdict for one sample."""

    sample_id: str
    verdicts: tuple[Verdict, ...] = ()

    @property
    def passed(self) -> bool:
        return all(verdict.passed for verdict in self.verdicts)

    @property
    def failures(self) -> tuple[Verdict, ...]:
        return tuple(verdict for verdict in self.verdicts if not verdict.passed)

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "passed": self.passed,
            "failures": [
                {"validator": v.validator, "reason": v.reason, "value": v.value} for v in self.failures
            ],
        }


class Validator(ABC):
    """Answers one question about a sample."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def check(self, sample: Sample) -> Verdict: ...

    def _ok(self, value: float | None = None) -> Verdict:
        return Verdict(self.name, True, value=value)

    def _fail(self, reason: str, value: float | None = None) -> Verdict:
        return Verdict(self.name, False, reason, value)


class NonEmptyText(Validator):
    """A page with no transcribable text supervises nothing."""

    def __init__(self, min_characters: int = 8, min_words: int = 2):
        self.min_characters = min_characters
        self.min_words = min_words

    def check(self, sample: Sample) -> Verdict:
        text = sample.text.strip()
        words = len(text.split())
        if len(text) < self.min_characters or words < self.min_words:
            return self._fail(f"only {len(text)} characters / {words} words", value=float(len(text)))
        return self._ok(float(len(text)))


class MinInkCoverage(Validator):
    """Catch pages that are blank, or that a degradation has washed out.

    Measured as the fraction of pixels darker than the page's own median, which is robust
    to shadow and vignette in a way an absolute threshold is not.
    """

    def __init__(self, min_fraction: float = 0.002, margin: int = 40):
        self.min_fraction = min_fraction
        self.margin = margin

    def check(self, sample: Sample) -> Verdict:
        grey = np.asarray(sample.image.convert("L"), dtype=np.float32)
        if grey.size == 0:
            return self._fail("empty image", 0.0)
        threshold = float(np.median(grey)) - self.margin
        fraction = float((grey < threshold).mean())
        if fraction < self.min_fraction:
            return self._fail(f"ink covers only {fraction:.4%} of the page", fraction)
        return self._ok(fraction)


class MinContrast(Validator):
    """Text that cannot be distinguished from its background is unreadable ground truth."""

    def __init__(
        self,
        min_difference: float = 25.0,
        max_regions: int = 12,
        max_failing_fraction: float = 0.5,
    ):
        self.min_difference = min_difference
        self.max_regions = max_regions
        self.max_failing_fraction = max_failing_fraction

    def check(self, sample: Sample) -> Verdict:
        grey = np.asarray(sample.image.convert("L"), dtype=np.float32)
        differences: list[float] = []
        for region in sample.page.ordered_regions()[: self.max_regions]:
            if not region.lines:
                continue
            x0, y0, x1, y1 = region.bbox.as_int()
            patch = grey[max(0, y0) : y1, max(0, x0) : x1]
            if patch.size < 16:
                continue
            # Contrast is a *local* property: compare the darkest pixels of the block
            # against its own paper. Comparing a block's mean to the page mean instead
            # would flag every sparse page, because a text block is mostly paper too.
            ink, paper = np.percentile(patch, [10, 90])
            differences.append(float(paper - ink))
        if not differences:
            return self._ok()
        # Judged per block, not by the best one on the page. A faded scan often keeps its
        # heading dark while every body block washes out; taking the maximum let that page
        # through carrying a full transcription of text no reader could recover.
        failing = sum(1 for d in differences if d < self.min_difference) / len(differences)
        if failing > self.max_failing_fraction:
            worst = min(differences)
            return self._fail(
                f"{failing:.0%} of text blocks stand out from their background by under "
                f"{self.min_difference:g} levels (worst {worst:.1f})",
                failing,
            )
        return self._ok(failing)


class BoxesInsidePage(Validator):
    """A box outside the canvas is rejected outright by every detection format."""

    def __init__(self, tolerance: float = 1.0):
        self.tolerance = tolerance

    def check(self, sample: Sample) -> Verdict:
        page = sample.page
        limit = page.bbox.pad(self.tolerance)
        for region in page.regions:
            if not limit.contains(region.bbox):
                return self._fail(f"region {region.type.value} leaves the page")
            for line in region.lines:
                for word in line.words:
                    if not limit.contains(word.bbox):
                        return self._fail(f"word {word.text!r} leaves the page")
        return self._ok()


class NoOverlappingRegions(Validator):
    """Overlapping blocks mean the layout engine lost track, not that documents overlap."""

    def __init__(self, max_iou: float = 0.15):
        self.max_iou = max_iou

    def check(self, sample: Sample) -> Verdict:
        boxes = [
            region.bbox
            for region in sample.page.regions
            if region.type is not RegionType.SEPARATOR and region.lines
        ]
        worst = 0.0
        for index, left in enumerate(boxes):
            for right in boxes[index + 1 :]:
                worst = max(worst, left.iou(right))
        if worst > self.max_iou:
            return self._fail(f"two regions overlap at IoU {worst:.2f}", worst)
        return self._ok(worst)


class LegibleLineHeight(Validator):
    """Text too small to resolve after downscaling is noise with a confident label."""

    def __init__(self, min_height: float = 8.0, max_failing_fraction: float = 0.1):
        self.min_height = min_height
        self.max_failing_fraction = max_failing_fraction

    def check(self, sample: Sample) -> Verdict:
        heights = [line.bbox.height for line in sample.page.iter_lines()]
        if not heights:
            return self._ok()
        failing = sum(1 for height in heights if height < self.min_height) / len(heights)
        if failing > self.max_failing_fraction:
            return self._fail(f"{failing:.0%} of lines are under {self.min_height:g}px", failing)
        return self._ok(failing)


@dataclass
class ValidationPipeline:
    """Runs validators over a sample and can filter a stream of them."""

    validators: list[Validator] = field(default_factory=list)
    #: Stop at the first failure. Cheap when validation is used purely as a gate.
    fail_fast: bool = True

    def check(self, sample: Sample) -> ValidationReport:
        verdicts: list[Verdict] = []
        for validator in self.validators:
            verdict = validator.check(sample)
            verdicts.append(verdict)
            if self.fail_fast and not verdict.passed:
                break
        return ValidationReport(sample.id, tuple(verdicts))

    def filter(self, samples, *, on_reject=None):
        """Yield only the samples that pass, reporting rejections to `on_reject`."""
        for sample in samples:
            report = self.check(sample)
            if report.passed:
                yield sample
            elif on_reject is not None:
                on_reject(report)


def default_validators() -> ValidationPipeline:
    """The checks worth running on every corpus."""
    return ValidationPipeline(
        [
            NonEmptyText(),
            BoxesInsidePage(),
            MinInkCoverage(),
            MinContrast(),
            NoOverlappingRegions(),
            LegibleLineHeight(),
        ]
    )
