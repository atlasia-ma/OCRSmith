"""Generation pipeline: seeds in, annotated pages out."""

from .factory import SampleFactory, build_text_provider
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
    "SampleFactory",
    "ShardPlan",
    "build_text_provider",
    "iter_samples",
    "effective_workers",
    "parallelism_advice",
    "plan_shards",
    "run_generation",
    "run_shard",
]
