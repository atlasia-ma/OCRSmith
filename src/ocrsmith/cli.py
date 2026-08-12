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
