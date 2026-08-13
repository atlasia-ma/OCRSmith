"""Validating that a synthetic corpus resembles a real one, and that changes help.

Two things this package answers that the rest of the project only asserts:

* **Does the output look like real documents?** `compare_corpora` measures it, feature by
  feature, and names the generator knob behind each gap.
* **Does a feature actually help?** `build_ablation` produces corpora that differ in
  exactly one knob and share a seed, so a downstream accuracy difference can only be
  attributed to that knob.
"""

from .ablation import PRESET_ABLATIONS, AblationPlan, AblationVariant, build_ablation
from .comparison import (
    ComparisonReport,
    FeatureComparison,
    compare_corpora,
    compare_features,
)
from .features import FEATURE_NAMES, ImageFeatures, extract_features, iter_image_features

__all__ = [
    "FEATURE_NAMES",
    "PRESET_ABLATIONS",
    "AblationPlan",
    "AblationVariant",
    "ComparisonReport",
    "FeatureComparison",
    "ImageFeatures",
    "build_ablation",
    "compare_corpora",
    "compare_features",
    "extract_features",
    "iter_image_features",
]
