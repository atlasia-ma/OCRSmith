"""Running a generation job.

Two properties drive the design:

* **Nothing is materialised.** The pipeline is a generator all the way down; a run of ten
  million pages holds one page in memory at a time. The previous engine built a list of
  every text, then a list of every task, then a list of every annotation, so peak memory
  scaled with dataset size and a large run simply died.
* **Work is sharded, not streamed between processes.** Each worker generates *and writes*
  its own shard. Sending images back through a pickle queue would make the parent process
  the bottleneck and cap throughput at one core's worth of serialisation.

A shard that completes is left on disk complete. Re-running skips shards that already
exist, which turns "the job died at 80%" into a resume rather than a restart.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config.schema import GenerationConfig
from ..datasets.writers import build_sink
from ..domain import Sample
from .factory import SampleFactory

__all__ = ["GenerationResult", "ShardPlan", "iter_samples", "plan_shards", "run_generation"]


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """One unit of parallel work: a contiguous range of document indices."""

    index: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(slots=True)
class GenerationResult:
    """What a run produced."""

    documents: int = 0
    pages: int = 0
    shards: int = 0
    failures: int = 0
    output_dir: Path = field(default_factory=Path)
    skipped_shards: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "pages": self.pages,
            "shards": self.shards,
            "failures": self.failures,
            "output_dir": str(self.output_dir),
            "skipped_shards": list(self.skipped_shards),
        }


def plan_shards(config: GenerationConfig) -> tuple[ShardPlan, ...]:
    """Split the requested document range into shards of `output.shard_size`."""
    start = config.run.start_index
    stop = start + config.run.num_samples
    size = config.output.shard_size
    return tuple(
        ShardPlan(index=number, start=lower, stop=min(lower + size, stop))
        for number, lower in enumerate(range(start, stop, size))
    )


def iter_samples(
    config: GenerationConfig,
    indices: Sequence[int] | Iterator[int] | None = None,
    *,
    factory: SampleFactory | None = None,
) -> Iterator[Sample]:
    """Stream samples for `indices` (default: the configured run range).

    A document may occupy several pages, so this yields more samples than there are
    indices. Failures are skipped rather than fatal — one unlucky corpus row should not
    end a twelve-hour job — but a run of consecutive failures aborts, because that means
    the configuration is wrong rather than the data unlucky.
    """
    factory = factory or SampleFactory(config)
    if indices is None:
        indices = range(config.run.start_index, config.run.start_index + config.run.num_samples)

    consecutive_failures = 0
    for index in indices:
        try:
            produced = False
            for sample in factory.create(index):
                produced = True
                yield sample
            consecutive_failures = 0 if produced else consecutive_failures + 1
        except Exception:
            consecutive_failures += 1
        if consecutive_failures >= config.run.max_consecutive_failures:
            raise RuntimeError(
                f"Aborting after {consecutive_failures} consecutive failures around index "
                f"{index}; the configuration is probably wrong rather than the data unlucky."
            )


def _shard_marker(directory: Path, shard: int) -> Path:
    return directory / f".shard-{shard:05d}.done"


def run_shard(config: GenerationConfig, plan: ShardPlan) -> dict:
    """Generate and write one shard. Safe to call in a worker process."""
    directory = Path(config.output.dir)
    directory.mkdir(parents=True, exist_ok=True)
    marker = _shard_marker(directory, plan.index)
    if marker.exists():
        return {"shard": plan.index, "pages": 0, "documents": 0, "skipped": True}

    factory = SampleFactory(config)
    sink = build_sink(
        config.output.format,
        directory,
        shard=plan.index,
        image_format=config.output.image_format,
        image_quality=config.output.image_quality,
        images_subdir=config.output.images_subdir,
    )

    pages = 0
    documents = set()
    with sink:
        for sample in iter_samples(config, range(plan.start, plan.stop), factory=factory):
            sink.write(sample)
            pages += 1
            documents.add(sample.provenance.extra.get("index"))

    marker.write_text(str(pages), encoding="utf-8")
    return {"shard": plan.index, "pages": pages, "documents": len(documents), "skipped": False}


def run_generation(config: GenerationConfig, *, progress=None) -> GenerationResult:
    """Run a complete generation job, in parallel when `run.workers > 1`.

    `progress` is called with each completed shard's summary, so a CLI can render a bar
    without the pipeline knowing anything about terminals.
    """
    directory = Path(config.output.dir)
    directory.mkdir(parents=True, exist_ok=True)
    plans = plan_shards(config)
    result = GenerationResult(output_dir=directory)
    skipped: list[int] = []

    def absorb(summary: dict) -> None:
        result.pages += summary["pages"]
        result.documents += summary["documents"]
        result.shards += 0 if summary["skipped"] else 1
        if summary["skipped"]:
            skipped.append(summary["shard"])
        if progress is not None:
            progress(summary)

    workers = min(config.run.workers, len(plans)) if plans else 1
    if workers <= 1 or len(plans) == 1:
        for plan in plans:
            absorb(run_shard(config, plan))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        payload = config.model_dump()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_worker, payload, plan.index, plan.start, plan.stop): plan for plan in plans
            }
            for future in as_completed(futures):
                try:
                    absorb(future.result())
                except Exception:
                    result.failures += 1

    result.skipped_shards = tuple(sorted(skipped))
    return result


def _worker(config_payload: dict, index: int, start: int, stop: int) -> dict:
    """Process-pool entry point.

    The config crosses as a plain dict rather than a model instance: pydantic objects
    pickle, but a dict is version-agnostic and makes the process boundary explicit.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = GenerationConfig.model_validate(config_payload)
    return run_shard(config, ShardPlan(index=index, start=start, stop=stop))
