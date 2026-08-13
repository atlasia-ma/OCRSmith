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
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config.schema import GenerationConfig
from ..datasets.writers import build_sink
from ..domain import Sample
from ..quality import default_validators
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
    #: `(shard index, error)` for every shard that died. A bare count says a run went wrong
    #: without saying what went wrong, which is the difference between a minute of
    #: debugging and an hour of it.
    shard_errors: tuple[tuple[int, str], ...] = ()
    #: Samples dropped by the quality gate, counted by the check that rejected them.
    rejected: dict = field(default_factory=dict)
    output_dir: Path = field(default_factory=Path)
    skipped_shards: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "pages": self.pages,
            "shards": self.shards,
            "failures": self.failures,
            "shard_errors": [list(item) for item in self.shard_errors],
            "rejected": dict(self.rejected),
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
        return {"shard": plan.index, "pages": 0, "documents": 0, "rejected": {}, "skipped": True}

    factory = SampleFactory(config)
    sink = build_sink(
        config.output.format,
        directory,
        shard=plan.index,
        image_format=config.output.image_format,
        image_quality=config.output.image_quality,
        images_subdir=config.output.images_subdir,
    )
    gate = default_validators() if config.quality.enabled else None

    pages = 0
    rejected: Counter[str] = Counter()
    documents = set()
    with sink:
        for sample in iter_samples(config, range(plan.start, plan.stop), factory=factory):
            if gate is not None:
                report = gate.check(sample)
                if not report.passed:
                    rejected[report.failures[0].validator] += 1
                    continue
            sink.write(sample)
            pages += 1
            documents.add(sample.provenance.extra.get("index"))

    produced = pages + sum(rejected.values())
    if produced and sum(rejected.values()) / produced > config.quality.max_rejection_rate:
        # A high rejection rate is a configuration problem, and writing a shard that is
        # mostly holes would hide it behind a plausible-looking dataset.
        raise RuntimeError(
            f"Shard {plan.index} rejected {sum(rejected.values())}/{produced} samples "
            f"({dict(rejected)}); check the configuration rather than the data."
        )

    marker.write_text(str(pages), encoding="utf-8")
    return {
        "shard": plan.index,
        "pages": pages,
        "documents": len(documents),
        "rejected": dict(rejected),
        "skipped": False,
    }


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
        for name, count in summary.get("rejected", {}).items():
            result.rejected[name] = result.rejected.get(name, 0) + count
        if summary["skipped"]:
            skipped.append(summary["shard"])
        if progress is not None:
            progress(summary)

    errors: list[tuple[int, str]] = []

    def record(plan: ShardPlan, error: BaseException) -> None:
        """Keep the reason. One dead shard should not end a twelve-hour run, but losing
        *why* it died turns a one-minute diagnosis into an afternoon."""
        errors.append((plan.index, f"{type(error).__name__}: {error}"))
        result.failures += 1

    workers = min(config.run.workers, len(plans)) if plans else 1
    if workers <= 1 or len(plans) == 1:
        for plan in plans:
            try:
                absorb(_run_one(config, plan))
            except Exception as error:  # noqa: PERF203 - one shard failing is not fatal
                record(plan, error)
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
                except Exception as error:
                    record(futures[future], error)

    result.skipped_shards = tuple(sorted(skipped))
    result.shard_errors = tuple(errors)

    if errors and not result.pages and not skipped:
        # Every shard died and nothing was written. Returning a result here would report a
        # successful run of zero pages, which is how a broken configuration reaches a
        # training set as an empty directory nobody questioned.
        reasons = "; ".join(f"shard {index}: {reason}" for index, reason in errors[:3])
        raise RuntimeError(f"All {len(errors)} shard(s) failed and nothing was written. {reasons}")

    return result


def _run_one(config: GenerationConfig, plan: ShardPlan) -> dict:
    """Seam for running a single shard in-process, so both paths share failure handling."""
    return run_shard(config, plan)


def _worker(config_payload: dict, index: int, start: int, stop: int) -> dict:
    """Process-pool entry point.

    The config crosses as a plain dict rather than a model instance: pydantic objects
    pickle, but a dict is version-agnostic and makes the process boundary explicit.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = GenerationConfig.model_validate(config_payload)
    return run_shard(config, ShardPlan(index=index, start=start, stop=stop))
