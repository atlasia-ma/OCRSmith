"""Command line interface."""

from __future__ import annotations

import contextlib
import json
import random
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from .config import DEFAULT_CONFIG_PATH, load_config
from .datasets.writers import sink_names
from .pipeline import (
    SampleFactory,
    corpus_advice,
    parallelism_advice,
    plan_shards,
    run_generation,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="OCRSmith — generate synthetic document and OCR datasets.",
)
console = Console()

_CONFIG_OPTION = typer.Option(None, "--config", "-c", help="Path to a YAML corpus specification.")
_SET_OPTION = typer.Option(
    None, "--set", "-s", help="Override a config key, e.g. --set run.workers=8 (repeatable)."
)


@app.command()
def generate(
    config: Path = _CONFIG_OPTION,
    set_: list[str] = _SET_OPTION,
    num_samples: int = typer.Option(None, "--num-samples", "-n", help="Documents to generate."),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Where to write the dataset."),
    workers: int = typer.Option(None, "--workers", "-w", help="Worker processes."),
    seed: int = typer.Option(None, "--seed", help="Base seed; every sample derives from it."),
    output_format: str = typer.Option(None, "--format", "-f", help=f"One of: {', '.join(sink_names())}."),
) -> None:
    """Generate a dataset."""
    overrides = list(set_ or [])
    for key, value in (
        ("run.num_samples", num_samples),
        ("run.workers", workers),
        ("seed", seed),
        ("output.dir", str(output_dir) if output_dir else None),
        ("output.format", output_format),
    ):
        if value is not None:
            overrides.append(f"{key}={json.dumps(value)}")

    spec = load_config(config, overrides)
    _require_fonts(spec)
    shards = plan_shards(spec)
    console.print(
        f"[bold]{spec.run.num_samples}[/] documents → [bold]{len(shards)}[/] shard(s) "
        f"in [cyan]{spec.output.dir}[/] as [cyan]{spec.output.format}[/] "
        f"using [bold]{spec.run.workers}[/] worker(s)"
    )
    for advice in (parallelism_advice(spec), corpus_advice(spec)):
        if advice:
            console.print(f"[yellow]{advice}[/]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} shards"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("generating", total=len(shards))
        result = run_generation(spec, progress=lambda _summary: progress.advance(task))

    console.print(
        f"[green]Done.[/] {result.pages} pages from {result.documents} documents "
        f"across {result.shards} shard(s)."
    )
    if result.skipped_shards:
        console.print(f"[yellow]Skipped {len(result.skipped_shards)} already-complete shard(s).[/]")
    if result.rejected:
        # What the quality gate threw away, and why. A run that silently discards a
        # quarter of its samples is a configuration to fix, not a cost to absorb, and the
        # only way to know is to be told which check is doing the rejecting.
        dropped = sum(result.rejected.values())
        share = dropped / max(1, dropped + result.pages)
        reasons = ", ".join(
            f"{name} {count}" for name, count in sorted(result.rejected.items(), key=lambda i: -i[1])
        )
        console.print(f"[yellow]Quality gate dropped {dropped} sample(s) ({share:.0%}): {reasons}[/]")
    if result.failures:
        console.print(f"[red]{result.failures} shard(s) failed.[/]")
        for index, reason in result.shard_errors:
            console.print(f"  [red]shard {index}:[/] {reason}")
        raise typer.Exit(code=1)


def _require_fonts(spec) -> None:
    """Stop before the run when there are no fonts, and say what to do about it.

    The repository ships no typefaces, so this is the first thing a new install hits.
    Without the check it surfaced as a `RuntimeError` traceback from inside a dead
    worker shard — technically accurate, and a poor way to learn that you need to run
    one more command.
    """
    from .core.fonts import discover_fonts

    if discover_fonts(spec.fonts.paths, include=spec.fonts.include, exclude=spec.fonts.exclude):
        return
    console.print(f"[red]No font files found under[/] {list(spec.fonts.paths)}")
    console.print("OCRSmith ships no typefaces of its own, so fetch some first:")
    console.print("  [cyan]ocrsmith fetch-fonts --subset arabic --output assets/fonts[/]")
    console.print("  [cyan]ocrsmith fetch-fonts --subset arabic --hands --output assets/fonts_hands[/]")
    raise typer.Exit(code=1)


@app.command()
def preview(
    config: Path = _CONFIG_OPTION,
    set_: list[str] = _SET_OPTION,
    count: int = typer.Option(3, "--count", "-n", help="Documents to preview."),
    output_dir: Path = typer.Option(Path("outputs/preview"), "--output", "-o"),
    boxes: bool = typer.Option(False, "--boxes", help="Draw region and line boxes."),
) -> None:
    """Render a handful of samples to look at, without writing a dataset."""
    from PIL import ImageDraw

    spec = load_config(config, set_ or [])
    factory = SampleFactory(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for index in range(count):
        for sample in factory.create(index):
            image = sample.image.convert("RGB")
            if boxes:
                # Every level the annotation actually carries, not just the outer two.
                # Words and table cells are the labels that make this corpus usable for
                # detection and table structure, and an overlay that omits them undersells
                # what the file contains.
                draw = ImageDraw.Draw(image)
                scale = max(1, round(min(image.size) / 900))
                for region in sample.page.regions:
                    draw.rectangle(region.bbox.as_int(), outline=(220, 60, 60), width=3 * scale)
                    if region.table is not None:
                        for cell in region.table.cells:
                            draw.rectangle(cell.bbox.as_int(), outline=(240, 150, 30), width=2 * scale)
                    for line in region.lines:
                        draw.rectangle(line.bbox.as_int(), outline=(40, 110, 220), width=1 * scale)
                        for word in line.words:
                            draw.rectangle(word.bbox.as_int(), outline=(30, 170, 90), width=1 * scale)
            image.save(output_dir / f"{sample.id}.png")
            (output_dir / f"{sample.id}.md").write_text(sample.page.to_markdown(), encoding="utf-8")
            written += 1
    console.print(f"[green]Wrote {written} preview page(s) to[/] {output_dir}")


@app.command("fetch-fonts")
def fetch_fonts(
    subset: str = typer.Option("arabic", "--subset", "-s", help="Unicode subset, e.g. arabic, latin."),
    output_dir: Path = typer.Option(Path("assets/fonts"), "--output", "-o"),
    limit: int = typer.Option(0, "--limit", "-n", help="Fetch at most this many families (0 = all)."),
    category: str = typer.Option(
        None,
        "--category",
        help="Only families in this Google Fonts category: handwriting, serif, sans-serif, display.",
    ),
    hands: bool = typer.Option(
        False,
        "--hands",
        help="Only families whose letterforms come from a written hand (wider than --category handwriting).",
    ),
    include_noto: bool = typer.Option(True, "--include-noto/--no-noto"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be fetched."),
) -> None:
    """Download open-licensed font families from Google Fonts.

    Font diversity is the highest-impact lever in synthetic text data, and a repository
    should not ship other people's typefaces. Every family's licence is downloaded
    alongside it and recorded in a manifest, so a dataset stays reproducible.
    """
    from .assets import fetch_families, list_families, write_manifest

    records = list_families(subset)
    if category:
        wanted = category.strip().lower().replace("-", " ")
        records = tuple(r for r in records if r.category.strip().lower().replace("-", " ") == wanted)
    if hands:
        # Wider than `--category handwriting` on purpose. Google Fonts puts exactly one
        # Arabic family in that category, while the styles people actually write in -
        # ruqʿah above all - are filed under Serif and Display. Fetching by category
        # alone gives a "handwriting" pool of one face.
        from .core.documents.typography import _looks_handwritten

        records = tuple(
            r for r in records if _looks_handwritten(r.family) or r.category.lower() == "handwriting"
        )
    if not include_noto:
        records = tuple(record for record in records if not record.is_noto)
    if limit:
        records = records[:limit]
    if not records:
        console.print("[yellow]No families matched. Try --dry-run without the filters.[/]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{len(records)}[/] open-licensed families with the [cyan]{subset}[/] subset")
    if dry_run:
        table = Table(title="would fetch")
        table.add_column("family")
        table.add_column("category")
        table.add_column("designers")
        for record in records:
            table.add_row(record.family, record.category, ", ".join(record.designers)[:44])
        console.print(table)
        return

    fetched = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} families"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("fetching", total=len(records))
        for record in fetch_families(records, output_dir, on_progress=lambda *_: progress.advance(task)):
            fetched.append(record)
        progress.update(task, completed=len(records))

    manifest = write_manifest(output_dir, fetched, subset)
    files = sum(len(record.files) for record in fetched)
    console.print(f"[green]Fetched {len(fetched)} families ({files} files) into[/] {output_dir}")
    console.print(f"Manifest: {manifest}")


@app.command()
def fonts(
    config: Path = _CONFIG_OPTION,
    text: str = typer.Option(
        "مرحبا بالعالم 2024 OCR", "--text", "-t", help="Text to check coverage against."
    ),
) -> None:
    """List the configured fonts and whether they can draw a given string."""
    from .core.fonts import FontPool
    from .text.coverage import supports_text

    spec = load_config(config)
    pool = FontPool(spec.fonts.paths, include=spec.fonts.include, exclude=spec.fonts.exclude)
    table = Table(title=f"{len(pool)} fonts")
    table.add_column("font")
    table.add_column("coverage", justify="right")
    table.add_column("missing")
    for face in pool.faces:
        report = supports_text(face, text)
        colour = "green" if report.is_complete else "yellow" if report.ratio > 0.5 else "red"
        table.add_row(face.name, f"[{colour}]{report.ratio:.0%}[/]", "".join(report.missing)[:20])
    console.print(table)


@app.command()
def doctor(config: Path = _CONFIG_OPTION) -> None:
    """Report whether this machine can produce correct Arabic output."""
    from .core.fonts import discover_fonts
    from .text import resolve_shaper
    from .text.shaping import harfbuzz_available, raqm_available

    spec = load_config(config)
    shaper = resolve_shaper()
    faces = discover_fonts(spec.fonts.paths, include=spec.fonts.include, exclude=spec.fonts.exclude)

    table = Table(title="ocrsmith doctor")
    table.add_column("check")
    table.add_column("result")
    table.add_column("note")
    engine = type(shaper).__name__
    if harfbuzz_available():
        note = "OCRSmith shapes with HarfBuzz and rasterises with FreeType"
        colour = "green"
    elif raqm_available():
        note = "shaping delegated to Pillow's Raqm"
        colour = "green"
    else:
        note = (
            "substituting Arabic presentation forms; faces that join through GSUB - "
            # `\[` escapes the bracket for rich, which would otherwise read
            # `[shaping]` as a markup tag and print `pip install 'ocrsmith'` - a
            # command that installs the wrong thing.
            r"every handwriting face - cannot be used. pip install 'ocrsmith\[shaping]'"
        )
        colour = "yellow"
    table.add_row("text engine", f"[{colour}]{engine}[/]", note)
    table.add_row("Pillow Raqm", "yes" if raqm_available() else "no", "")
    table.add_row(
        "fonts",
        f"[green]{len(faces)}[/]" if faces else "[red]0[/]",
        ", ".join(str(path) for path in spec.fonts.paths),
    )
    table.add_row("renderer", spec.render.mode, "render.mode")
    table.add_row("text source", spec.text.source.type, str(spec.text.source.path or "inline"))
    table.add_row("config", "loaded", str(config or DEFAULT_CONFIG_PATH))
    console.print(table)

    sample = shaper.shape("مرحبا")
    console.print(f"shaping check: logical={sample.logical!r} reshaped={sample.was_reshaped}")

    # How many faces can actually draw the script, which is not the same as how many are
    # on disk. Without Raqm the renderer draws Arabic presentation forms, and a face that
    # joins letters through GSUB alone carries no glyphs for them - so it is present,
    # readable, and unusable. Reporting only the file count hides that completely.
    if faces:
        from .core.fonts import FontPool

        pool = FontPool(spec.fonts.paths, include=spec.fonts.include, exclude=spec.fonts.exclude)
        usable = len(pool.supporting("مرحبا بالعالم 2024 OCR"))
        colour = "green" if usable > len(faces) * 0.8 else "yellow"
        advice = (
            ""
            if harfbuzz_available() or raqm_available()
            else r"  -> pip install 'ocrsmith\[shaping]' to use the rest"
        )
        console.print(
            f"usable faces: [{colour}]{usable}[/] of {len(faces)} can draw Arabic and Latin{advice}"
        )

    advice = corpus_advice(spec)
    if advice:
        console.print(f"[yellow]{advice}[/]")
    if not faces:
        raise typer.Exit(code=1)


@app.command()
def validate(
    dataset: Path = typer.Argument(..., help="A generated dataset directory."),
    limit: int = typer.Option(0, "--limit", "-n", help="Stop after this many pages (0 = all)."),
) -> None:
    """Re-check a generated dataset's annotations against its images."""
    from PIL import Image

    from .domain import Sample, page_from_dict
    from .quality import default_validators

    pipeline = default_validators()
    failures: dict[str, int] = {}
    checked = 0

    for path in sorted(dataset.glob("annotations-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = dataset / record["image_path"]
            if not image_path.exists():
                failures["MissingImage"] = failures.get("MissingImage", 0) + 1
                continue
            sample = Sample(record["id"], Image.open(image_path), page_from_dict(record["page"]))
            report = pipeline.check(sample)
            checked += 1
            for verdict in report.failures:
                failures[verdict.validator] = failures.get(verdict.validator, 0) + 1
            if limit and checked >= limit:
                break
        if limit and checked >= limit:
            break

    table = Table(title=f"validated {checked} page(s)")
    table.add_column("check")
    table.add_column("failures", justify="right")
    table.add_column("rate", justify="right")
    if failures:
        for name, count in sorted(failures.items(), key=lambda item: -item[1]):
            table.add_row(name, str(count), f"{count / max(1, checked):.1%}")
    else:
        table.add_row("[green]all checks passed[/]", "0", "0.0%")
    console.print(table)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def stats(
    dataset: Path = typer.Argument(..., help="A generated dataset directory."),
    markdown: Path = typer.Option(None, "--markdown", "-m", help="Write a dataset-card fragment here."),
) -> None:
    """Summarise what a generated dataset actually contains."""
    from .quality import scan_jsonl

    summary = scan_jsonl(dataset)
    if summary.pages == 0:
        console.print(f"[red]No annotation shards found in[/] {dataset}")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(summary.to_dict(), ensure_ascii=False))
    if markdown:
        markdown.write_text(summary.to_markdown(), encoding="utf-8")
        console.print(f"[green]Wrote dataset card fragment to[/] {markdown}")


@app.command()
def evaluate(
    dataset: Path = typer.Argument(..., help="The benchmark dataset directory."),
    predictions: Path = typer.Argument(..., help="JSON or JSONL of {id: prediction}."),
    target: str = typer.Option("text", "--target", help="Ground truth field: text, markdown or html."),
    ignore_diacritics: bool = typer.Option(False, "--ignore-diacritics"),
    worst: int = typer.Option(5, "--worst", help="Show this many worst-scoring samples."),
) -> None:
    """Score a model's predictions against a generated benchmark."""
    from .evaluation import evaluate as run_evaluation
    from .evaluation import load_references

    references = load_references(dataset, target=target)
    if not references:
        console.print(f"[red]No annotations found in[/] {dataset}")
        raise typer.Exit(code=1)

    raw = predictions.read_text(encoding="utf-8")
    if predictions.suffix == ".jsonl":
        parsed = {}
        for line in raw.splitlines():
            if line.strip():
                record = json.loads(line)
                parsed[record["id"]] = record.get("prediction", record.get("text", ""))
    else:
        parsed = json.loads(raw)

    report = run_evaluation(references, parsed, target=target, ignore_diacritics=ignore_diacritics)
    console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
    if worst:
        table = Table(title=f"{worst} worst samples")
        table.add_column("sample")
        table.add_column("CER", justify="right")
        table.add_column("WER", justify="right")
        for score in report.worst(worst):
            table.add_row(score.sample_id, f"{score.cer:.3f}", f"{score.wer:.3f}")
        console.print(table)


@app.command()
def compare(
    synthetic: Path = typer.Argument(..., help="A generated dataset directory (or any image folder)."),
    real: Path = typer.Argument(..., help="A folder of real document images."),
    limit: int = typer.Option(0, "--limit", "-n", help="Images per side (0 = all)."),
    markdown: Path = typer.Option(None, "--markdown", "-m", help="Write the report here."),
) -> None:
    """Measure how far a synthetic corpus is from a real one, feature by feature.

    The sim-to-real gap is usually answered with an assertion. This measures it, and every
    feature maps onto a generator knob - so the report names what to change rather than
    reporting an abstract divergence.
    """
    from .validation import compare_corpora

    report = compare_corpora(synthetic, real, limit=limit)
    table = Table(title=f"{report.synthetic_count} synthetic vs {report.real_count} real")
    table.add_column("feature")
    table.add_column("synthetic", justify="right")
    table.add_column("real", justify="right")
    table.add_column("gap", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("verdict")
    colours = {"matched": "green", "drifting": "yellow", "mismatched": "red"}
    for item in sorted(report.comparisons, key=lambda c: -abs(c.cohens_d)):
        table.add_row(
            item.feature,
            f"{item.synthetic_mean:.3g}",
            f"{item.real_mean:.3g}",
            f"{item.relative_difference:+.0%}",
            f"{item.overlap:.0%}",
            f"[{colours[item.verdict]}]{item.verdict}[/]",
        )
    console.print(table)
    console.print(f"Mean distribution overlap: [bold]{report.mean_overlap:.1%}[/]")
    if markdown:
        markdown.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"[green]Wrote report to[/] {markdown}")


@app.command()
def ablate(
    name: str = typer.Argument(..., help="Which ablation, e.g. degradations, fonts, layout, diacritics."),
    config: Path = _CONFIG_OPTION,
    set_: list[str] = _SET_OPTION,
    output_root: Path = typer.Option(Path("outputs/ablations"), "--output", "-o"),
    num_samples: int = typer.Option(None, "--num-samples", "-n", help="Documents per arm."),
    generate_now: bool = typer.Option(False, "--generate", help="Generate the corpora, not just plan them."),
) -> None:
    """Build the corpora that answer "does this feature actually help?".

    Every arm shares the base seed, so the arms differ in exactly the knob under test and
    in nothing else - otherwise a downstream difference could come from either the feature
    or the sample, and the experiment answers nothing.
    """
    from .validation import PRESET_ABLATIONS, build_ablation

    if name not in PRESET_ABLATIONS:
        console.print(f"[red]Unknown ablation[/] {name}. Available: {', '.join(sorted(PRESET_ABLATIONS))}")
        raise typer.Exit(code=1)

    plan = build_ablation(name, load_config(config, set_ or []), output_root, num_samples=num_samples)
    table = Table(title=f"ablation: {name}")
    table.add_column("arm")
    table.add_column("changes")
    table.add_column("corpus")
    for variant in plan.variants:
        table.add_row(variant.name, ", ".join(variant.overrides) or "(baseline)", variant.config.output.dir)
    console.print(table)
    console.print(f"Manifest: {plan.write_manifest(output_root)}")

    if not generate_now:
        console.print("[yellow]Planned only.[/] Re-run with --generate to build the corpora.")
        return

    for variant in plan.variants:
        console.print(f"generating [cyan]{variant.name}[/] -> {variant.config.output.dir}")
        result = run_generation(variant.config)
        console.print(f"  {result.pages} pages")
    console.print(
        "[green]Done.[/] Train the same model on each arm and compare on a held-out "
        "*real* benchmark - a synthetic test set would reward the generator's own biases."
    )


@app.command()
def bench(
    config: Path = _CONFIG_OPTION,
    set_: list[str] = _SET_OPTION,
    count: int = typer.Option(0, "--count", "-n", help="Samples per mode (0 = pick a sensible default)."),
    mode: str = typer.Option("both", "--mode", "-m", help="Which renderer to time: document, line, or both."),
) -> None:
    """Measure single-core generation throughput.

    One core, in this process, with no writing to disk: the number this reports is what
    the renderer costs, not what your storage costs. Multiply by cores for a run, and
    check `output.shard_size` first — a shard is the unit of parallelism, so a run with
    two shards uses two processes however many workers were configured.

    The first call on a machine also pays for the font coverage cache, so the warm-up
    below is not decoration: without it the first few samples carry a one-off cost that
    belongs to the installation rather than to the renderer.
    """
    import time

    from .pipeline import build_factory

    modes = ["document", "line"] if mode == "both" else [mode]
    defaults = {"document": 24, "line": 500}

    table = Table(title="single-core throughput")
    table.add_column("mode")
    table.add_column("samples", justify="right")
    table.add_column("seconds", justify="right")
    table.add_column("per second", justify="right")
    table.add_column("ms each", justify="right")

    for name in modes:
        if name not in defaults:
            console.print(f"[red]Unknown mode {name!r}. Use document, line or both.[/]")
            raise typer.Exit(code=2)
        spec = load_config(config, [*(set_ or []), f'render.mode="{name}"'])
        factory = build_factory(spec)
        total = count or defaults[name]
        warmup = max(2, total // 20)
        for index in range(warmup):
            list(factory.create(index))

        produced = 0
        started = time.perf_counter()
        for index in range(total):
            for _sample in factory.create(index):
                produced += 1
        elapsed = time.perf_counter() - started
        table.add_row(
            name,
            str(produced),
            f"{elapsed:.2f}",
            f"[bold]{produced / elapsed:.1f}[/]",
            f"{elapsed / max(1, produced) * 1000:.1f}",
        )

    console.print(table)


@app.command("show-config")
def show_config(config: Path = _CONFIG_OPTION, set_: list[str] = _SET_OPTION) -> None:
    """Print the fully resolved configuration."""
    console.print_json(json.dumps(load_config(config, set_ or []).model_dump(), ensure_ascii=False))


def _force_utf8_output() -> None:
    """Make the console able to print Arabic.

    A default Windows terminal encodes stdout as cp1252, which cannot represent a single
    Arabic character — so a tool whose entire purpose is Arabic text would crash while
    reporting that it works. Replacing unencodable characters is better than dying.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):  # exotic or redirected streams
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_output()
    random.seed()  # CLI-level entropy; per-sample seeds are always derived, never drawn
    app()


if __name__ == "__main__":
    main()
