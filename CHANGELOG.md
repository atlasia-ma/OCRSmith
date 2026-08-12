# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ocrsmith.quality`: validation and statistics.
  - Six validators for the failures that do not raise — blank pages, washed-out text,
    boxes off the canvas, overlapping blocks, illegibly small lines, empty labels. Each
    explains itself, so a rejection rate is diagnosable rather than merely alarming.
  - The generation pipeline runs them as a gate; a shard that rejects more than
    `quality.max_rejection_rate` aborts, because that is a configuration problem and a
    shard full of holes would hide it behind a plausible-looking dataset.
  - `DatasetStats` accumulates the distributions that define a corpus — templates, capture
    conditions, direction, region types, alphabet, line-height percentiles — in constant
    memory, and renders a dataset-card fragment.
- `ocrsmith.evaluation`: CER, WER, normalised edit similarity, TEDS-style table similarity
  and IoU-thresholded detection P/R/F1, plus a harness that scores a model's predictions
  against a generated benchmark. Diacritic sensitivity is an explicit argument rather than
  a silent convention.
- `ocrsmith.domain.page_from_dict` and friends: annotations round-trip, so validation,
  statistics and evaluation all read the same records the writers produce.
- CLI: `validate`, `stats` and `evaluate`.
- `ocrsmith.pipeline`: streaming generation.
  - `SampleFactory.create(index)` derives a per-sample seed from `(config.seed, index)`
    and yields one annotated `Sample` per page. Any sample can be regenerated on its own,
    without replaying the run that produced it.
  - `iter_samples` is a generator end to end, so a ten-million-page run holds one page in
    memory at a time.
  - Work is sharded, and each worker *writes* its own shard rather than shipping images
    back through a pickle queue. Completed shards are marked, so an interrupted job
    resumes instead of restarting.
- `ocrsmith.datasets.writers`: six output formats behind one `SampleSink` protocol —
  `jsonl`, `parquet`, `webdataset`, `coco` (word/line/region instances), `paddleocr`
  (detection labels plus line crops) and `chat` (image + instruction + Markdown answer for
  vision-language fine-tuning). Sinks write incrementally, so a killed run leaves valid
  shards rather than one truncated file.
- `ocrsmith.config`: a single validated `GenerationConfig` describing a whole corpus, with
  every per-sample choice expressed as a distribution. Dotted-key CLI overrides
  (`--set run.workers=8`) and a JSON-serialisable payload that crosses the process
  boundary intact.
- New `ocrsmith` CLI (typer + rich): `generate`, `preview`, `fonts`, `doctor`,
  `show-config`. `doctor` reports whether this machine can produce correct Arabic; the CLI
  forces UTF-8 output so a cp1252 Windows console cannot crash a tool whose entire purpose
  is Arabic text.
- `ocrsmith.core.degradations`: capture-condition modelling, where every degradation takes
  the annotation as well as the image and returns both.
  - Geometric: `Rotation` and `PerspectiveWarp` derive an explicit forward point mapping
    and push the whole annotation through it, so boxes track the ink instead of drifting
    silently. Rotated words gain polygons.
  - Photometric: gaussian noise, paper grain, defocus and motion blur, brightness,
    contrast, JPEG artefacts, downscaling, ink spread/erosion, bleed-through, shadow,
    vignette, glare, stains and folds.
  - Five presets — `clean`, `scan`, `photo`, `fax`, `archive` — ordered along the physical
    capture chain, so a corpus can be composed by capture condition rather than by
    undifferentiated noise.
- `ocrsmith.core.documents`: a document engine that produces full pages, not single lines.
  - `DocumentBuilder` / `DocumentContent` separate *what a document says* from *how it
    looks*, so the same content laid out in one column or two yields identical markup
    ground truth.
  - `PageSpec.from_paper` derives the canvas from paper size and DPI, with margins,
    multi-column layout (right-first for RTL) and reserved header/footer bands.
  - `DocumentRenderer` pours blocks down columns and onto further pages, splitting prose
    and moving structure whole. Nothing is clipped, nothing overlaps, and pages can be
    streamed one at a time.
  - `TableRenderer` emits per-cell boxes, content-derived column widths, in-cell wrapping,
    RTL column mirroring, and five border styles.
  - `TypographySampler` samples one coherent font family per document with per-role sizes
    and spacing, rather than a random face per block.
  - Six document genres (article, report, newspaper, letter, form, invoice) behind a
    weighted `TemplateRegistry`.
- `ocrsmith.core.rendering`: a text renderer that emits pixels and their annotation in
  the same pass, so the two cannot disagree.
  - Words are drawn individually, which yields exact per-word boxes and is typographically
    safe for Arabic (letters never join across a space).
  - `visual_word_order` implements word-level bidi run reordering, so mixed
    Arabic/Latin/digit lines are drawn in the right place while the label stays logical.
  - Wrapping is lossless: no word is silently dropped, oversized words are broken rather
    than overflowed, and lines that do not fit the height budget are *reported* so the
    label can be trimmed to exactly what was drawn.
  - `TextStyle` covers alignment (including natural RTL), line/word spacing, stroke,
    underline, strikethrough, synthetic bold/italic, and baseline/word-gap jitter, with an
    injectable `random.Random` so any sample is reproducible.
- `ocrsmith.text`: a text subsystem that owns everything happening to a string before it
  becomes pixels.
  - Script and base-direction detection (`detect_script`, `detect_direction`).
  - `NormalizationPolicy` — opt-in, idempotent transforms for diacritics, tatweel,
    alef/ya/ta-marbuta unification, numeral systems and whitespace. Each one changes the
    label, so each one is explicit and recorded.
  - Bidi/shaping with two interchangeable backends: `TransparentShaper` when Pillow has
    Raqm, `ReshaperBidiShaper` (arabic-reshaper + python-bidi) otherwise. Both keep the
    logical string as the label so datasets are identical across machines.
  - Font glyph coverage via fontTools `cmap` (`supports_text`, `fonts_supporting`), so a
    font that cannot draw a character is rejected instead of silently emitting tofu.

- Development tooling: ruff lint/format configuration, pytest configuration with
  `pythonpath = ["src"]` (so a clean clone is testable without installing), coverage
  settings, and a GitHub Actions CI matrix over Linux/Windows and Python 3.10/3.12.
- `CONTRIBUTING.md` describing the TDD workflow, branch naming, and release process.
- Regression tests covering rendering without layout constraints.

### Changed

- `pyproject.toml`: dependency pins relaxed to compatible ranges, heavy tabular and
  Hugging Face dependencies moved into the `data` extra, project metadata and URLs
  completed.
- `HuggingFaceTextLoader` accepts any iterable of mappings, streams records via
  `iter_texts`, and defers the `datasets` import to call time.

### Removed

- The v0 engine and everything coupled to it: `OCRSmithEngine`, `core/app.py`, the four
  `*Manager` classes, and the `augmentation`, `text_renderers`, `text_placement`,
  `backgrounds` and `fonts` strategy packages. Their responsibilities now live in
  `core/rendering`, `core/documents`, `core/degradations`, `core/fonts.py`,
  `core/backgrounds.py` and `pipeline/`, with word-level ground truth throughout.
- `ocrsmith.utils`, whose remaining helpers had no callers.

### Fixed

- `wrap_text_by_pixels` raised `TypeError` when `layout.max_width` / `layout.max_height`
  were unset, which made every sample fail for configs that omit them.
- Whitespace-only input no longer produces a zero-sized canvas.
