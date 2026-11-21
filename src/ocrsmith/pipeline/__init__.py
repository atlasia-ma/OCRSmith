"""Generation pipeline: seeds in, annotated samples out."""

from .factory import SampleFactory, build_factory, build_text_provider, corpus_advice
from .line_factory import LineSampleFactory
from .runner import (
    GenerationResult,
    ShardPlan,
    effective_workers,
    iter_samples,
    parallelism_advice,
    plan_shards,
    run_generation,
    run_shard,
)

__all__ = [
    "GenerationResult",
    "LineSampleFactory",
    "SampleFactory",
    "ShardPlan",
    "build_factory",
    "build_text_provider",
    "corpus_advice",
    "iter_samples",
    "effective_workers",
    "parallelism_advice",
    "plan_shards",
    "run_generation",
    "run_shard",
]
