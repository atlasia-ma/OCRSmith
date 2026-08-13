"""Comparing a synthetic corpus against a real one.

A divergence number on its own is not actionable. What makes this useful is that every
feature maps onto a generator knob, so the report says *which* knob is wrong and in which
direction — "synthetic strokes are 22% thinner than real, widen the weight distribution or
raise ink spread" rather than "domain gap: 0.31".

Two measures per feature:

* **standardised mean difference** (Cohen's d) — how far apart the centres are, in units
  of pooled spread. Signed, so it says which way.
* **overlap coefficient** — what fraction of the two distributions coincide. A distribution
  can have the right mean and the wrong spread, and a mean-only measure would call that a
  match; this catches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, ImageFeatures, iter_image_features

__all__ = ["ComparisonReport", "FeatureComparison", "compare_corpora", "compare_features"]

#: Cohen's-d thresholds. Conventional small/medium/large effect sizes, used here as
#: "matched / drifting / mismatched" so a report is readable without a statistics refresher.
_SMALL, _MEDIUM = 0.2, 0.5
#: Reported when two corpora are each internally constant but differ from one another.
_LARGE_EFFECT = 10.0


@dataclass(frozen=True, slots=True)
class FeatureComparison:
    """How one feature differs between two corpora."""

    feature: str
    synthetic_mean: float
    real_mean: float
    synthetic_std: float
    real_std: float
    cohens_d: float
    overlap: float

    @property
    def verdict(self) -> str:
        magnitude = abs(self.cohens_d)
        if magnitude < _SMALL:
            return "matched"
        if magnitude < _MEDIUM:
            return "drifting"
        return "mismatched"

    @property
    def direction(self) -> str:
        if abs(self.cohens_d) < _SMALL:
            return "="
        return "synthetic higher" if self.cohens_d > 0 else "synthetic lower"

    @property
    def relative_difference(self) -> float:
        """Signed difference as a fraction of the real mean, for a human-readable gap."""
        if not self.real_mean:
            return 0.0
        return (self.synthetic_mean - self.real_mean) / abs(self.real_mean)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "synthetic_mean": round(self.synthetic_mean, 4),
            "real_mean": round(self.real_mean, 4),
            "cohens_d": round(self.cohens_d, 3),
            "overlap": round(self.overlap, 3),
            "relative_difference": round(self.relative_difference, 3),
            "verdict": self.verdict,
        }


#: What to change when a feature is off. The value of this report is that it points at a
#: knob rather than at an abstraction.
_REMEDIES = {
    "ink_fraction": "page density - font size range, margins, or how much text a template emits",
    "ink_darkness": "ink colour range in the typography sampler",
    "paper_lightness": "background tint_range",
    "contrast": "ink colour and background tint, or the Contrast degradation",
    "stroke_width": "font weight distribution, InkSpread / InkErosion probability",
    "edge_density": "Blur and Downscale strength - crisp renders have far more edges",
    "high_frequency_energy": "Blur, Downscale and GaussianNoise strength",
    "block_noise": "JpegArtifacts quality range",
    "illumination_range": "IlluminationField, Shadow and Vignette strength",
    "aspect_ratio": "paper sizes and landscape probability",
}


@dataclass
class ComparisonReport:
    """Feature-by-feature comparison of two corpora."""

    comparisons: list[FeatureComparison]
    synthetic_count: int = 0
    real_count: int = 0

    @property
    def mismatched(self) -> list[FeatureComparison]:
        return [c for c in self.comparisons if c.verdict == "mismatched"]

    @property
    def mean_overlap(self) -> float:
        """Average distribution overlap; the single number, reported with its parts."""
        return float(np.mean([c.overlap for c in self.comparisons])) if self.comparisons else 0.0

    def to_dict(self) -> dict:
        return {
            "synthetic_images": self.synthetic_count,
            "real_images": self.real_count,
            "mean_overlap": round(self.mean_overlap, 3),
            "mismatched": [c.feature for c in self.mismatched],
            "features": [c.to_dict() for c in self.comparisons],
        }

    def to_markdown(self) -> str:
        lines = [
            "## Synthetic vs real",
            "",
            f"- **Images**: {self.synthetic_count:,} synthetic, {self.real_count:,} real",
            f"- **Mean distribution overlap**: {self.mean_overlap:.1%}",
            "",
            "| feature | synthetic | real | gap | overlap | verdict |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in sorted(self.comparisons, key=lambda c: -abs(c.cohens_d)):
            lines.append(
                f"| `{item.feature}` | {item.synthetic_mean:.3g} | {item.real_mean:.3g} "
                f"| {item.relative_difference:+.0%} | {item.overlap:.0%} | {item.verdict} |"
            )
        if self.mismatched:
            lines += ["", "### What to change", ""]
            for item in self.mismatched:
                lines.append(
                    f"- **`{item.feature}`** is {item.direction} "
                    f"({item.relative_difference:+.0%}) - "
                    f"{_REMEDIES.get(item.feature, 'review the config')}"
                )
        else:
            lines += ["", "No feature is mismatched at a large effect size."]
        return "\n".join(lines) + "\n"


def _overlap(left: np.ndarray, right: np.ndarray, bins: int = 24) -> float:
    """Overlap coefficient of two samples, via a shared histogram.

    Catches the case a mean comparison misses: the same centre with a different spread is
    not the same distribution, and a generator that produces every page at exactly the
    average is a worse match than one that spreads correctly.
    """
    if left.size == 0 or right.size == 0:
        return 0.0
    low = float(min(left.min(), right.min()))
    high = float(max(left.max(), right.max()))
    if high - low < 1e-12:
        return 1.0
    edges = np.linspace(low, high, bins + 1)
    left_hist, _ = np.histogram(left, bins=edges)
    right_hist, _ = np.histogram(right, bins=edges)
    left_norm = left_hist / max(1, left_hist.sum())
    right_norm = right_hist / max(1, right_hist.sum())
    return float(np.minimum(left_norm, right_norm).sum())


def compare_features(synthetic: list[ImageFeatures], real: list[ImageFeatures]) -> ComparisonReport:
    """Compare two already-extracted feature sets."""
    if not synthetic or not real:
        raise ValueError("Both corpora need at least one readable image")

    synthetic_matrix = np.vstack([item.as_vector() for item in synthetic])
    real_matrix = np.vstack([item.as_vector() for item in real])

    comparisons = []
    for index, name in enumerate(FEATURE_NAMES):
        left, right = synthetic_matrix[:, index], real_matrix[:, index]
        left_std = float(left.std(ddof=1)) if left.size > 1 else 0.0
        right_std = float(right.std(ddof=1)) if right.size > 1 else 0.0
        pooled = float(np.sqrt((left_std**2 + right_std**2) / 2))
        difference = float(left.mean() - right.mean())
        if pooled <= 1e-9 and abs(difference) > 1e-9:
            # Both corpora are internally constant but sit at different values. The
            # standardised difference is unbounded; reporting 0.0 would call a total
            # separation a perfect match, so it is clamped to a decisively large effect.
            cohens_d = float(np.sign(difference)) * _LARGE_EFFECT
        else:
            cohens_d = difference / pooled if pooled > 1e-9 else 0.0
        comparisons.append(
            FeatureComparison(
                feature=name,
                synthetic_mean=float(left.mean()),
                real_mean=float(right.mean()),
                synthetic_std=left_std,
                real_std=right_std,
                cohens_d=cohens_d,
                overlap=_overlap(left, right),
            )
        )
    return ComparisonReport(comparisons, len(synthetic), len(real))


def compare_corpora(synthetic_dir: str | Path, real_dir: str | Path, *, limit: int = 0) -> ComparisonReport:
    """Extract features from two directories of images and compare them."""
    synthetic = [features for _, features in iter_image_features(synthetic_dir, limit)]
    real = [features for _, features in iter_image_features(real_dir, limit)]
    return compare_features(synthetic, real)
