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
from .pipeline import SampleFactory, plan_shards, run_generation

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
    shards = plan_shards(spec)
    console.print(
        f"[bold]{spec.run.num_samples}[/] documents → [bold]{len(shards)}[/] shard(s) "
        f"in [cyan]{spec.output.dir}[/] as [cyan]{spec.output.format}[/] "
        f"using [bold]{spec.run.workers}[/] worker(s)"
    )

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
    if result.failures:
        console.print(f"[red]{result.failures} shard(s) failed.[/]")
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
                draw = ImageDraw.Draw(image)
                for region in sample.page.regions:
                    draw.rectangle(region.bbox.as_int(), outline=(220, 60, 60), width=3)
                    for line in region.lines:
                        draw.rectangle(line.bbox.as_int(), outline=(40, 110, 220), width=1)
            image.save(output_dir / f"{sample.id}.png")
            (output_dir / f"{sample.id}.md").write_text(sample.page.to_markdown(), encoding="utf-8")
            written += 1
    console.print(f"[green]Wrote {written} preview page(s) to[/] {output_dir}")


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
    from .text import raqm_available, resolve_shaper

    spec = load_config(config)
    shaper = resolve_shaper()
    faces = discover_fonts(spec.fonts.paths, include=spec.fonts.include, exclude=spec.fonts.exclude)

    table = Table(title="ocrsmith doctor")
    table.add_column("check")
    table.add_column("result")
    table.add_column("note")
    table.add_row(
        "Pillow Raqm",
        "[green]yes[/]" if raqm_available() else "[yellow]no[/]",
        "shaping delegated to HarfBuzz"
        if raqm_available()
        else "falling back to arabic-reshaper + python-bidi (still correct)",
    )
    table.add_row("shaper", type(shaper).__name__, "")
    table.add_row(
        "fonts",
        f"[green]{len(faces)}[/]" if faces else "[red]0[/]",
        ", ".join(str(path) for path in spec.fonts.paths),
    )
    table.add_row("text source", spec.text.source.type, str(spec.text.source.path or "inline"))
    table.add_row("config", "loaded", str(config or DEFAULT_CONFIG_PATH))
    console.print(table)

    sample = shaper.shape("مرحبا")
    console.print(f"shaping check: logical={sample.logical!r} reshaped={sample.was_reshaped}")
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
