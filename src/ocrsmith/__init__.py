"""OCRSmith — a synthetic document and OCR dataset forge.

The public surface is deliberately small: load a config, run a generation job, or drive
the pieces directly when you need something the config does not express.

    from ocrsmith import load_config, run_generation

    config = load_config("my_corpus.yaml")
    result = run_generation(config)
"""

from .config import GenerationConfig, load_config
from .domain import BBox, Line, Page, Provenance, Region, RegionType, Sample, Word
from .pipeline import SampleFactory, iter_samples, run_generation

__all__ = [
    "BBox",
    "GenerationConfig",
    "Line",
    "Page",
    "Provenance",
    "Region",
    "RegionType",
    "Sample",
    "SampleFactory",
    "Word",
    "iter_samples",
    "load_config",
    "run_generation",
]

__version__ = "1.0.0"
