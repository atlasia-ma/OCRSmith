# Architecture

OCRSmith is a pipeline of small, independently testable subsystems. Each one owns a
single stage of the journey from a string in a corpus to a labelled page on disk, and the
boundaries between them are plain data objects rather than shared state.

```
GenerationConfig ──► SampleFactory ──► Sample ──► SampleSink
                          │
        ┌─────────────────┼──────────────────┬───────────────┐
        ▼                 ▼                  ▼               ▼
   text/ + fonts   documents/ (layout)   rendering/     degradations/
```

## The invariant everything serves

> **The annotation describes exactly what was drawn — no more, no less.**

Every design decision below follows from it. A synthetic dataset that violates it looks
perfectly fine and quietly teaches a model something false, which is the worst failure
mode a data tool can have.

## `ocrsmith.config`

One pydantic model, `GenerationConfig`, describes an entire corpus. Two properties matter:

- **Every per-sample choice is a distribution.** Paper size, DPI, column count, template,
  capture condition and background are weight maps or ranges. Pinning one narrows the
  dataset in a way that is invisible until a model fails on whatever was excluded.
- **It is plain data.** No runtime objects, so it round-trips through `model_dump()` and
  crosses a process boundary intact. That is what makes `(seed, index)` sufficient to
  reproduce any sample.

## `ocrsmith.text`

Everything that happens to a string before it becomes pixels.

- `script.py` — script classification and first-strong-character base direction.
- `normalization.py` — `NormalizationPolicy`: opt-in, idempotent transforms for diacritics,
  tatweel, alef/ya unification, numeral systems and whitespace. **Every one of these
  rewrites the ground truth**, so none is applied silently.
- `shaping.py` — the logical/visual split. `ShapedText.logical` is the label;
  `ShapedText.visual` is what the rasteriser draws. Two interchangeable backends:
  `TransparentShaper` when Pillow has Raqm (HarfBuzz does the work), `ReshaperBidiShaper`
  otherwise. Labels are identical either way, so a dataset does not depend on how Pillow
  was compiled.
- `coverage.py` — exact glyph coverage from the font's `cmap` via fontTools, cached per
  file. A face that cannot draw a character is rejected rather than emitting tofu.

## `ocrsmith.domain`

Immutable, pixel-free value objects: the contract between the renderers that produce
pixels and the writers that serialise them.

```
Page ─┬─ Region (typed, reading_order) ─┬─ Line ── Word
      │                                 └─ Table ── TableCell
      └─ direction, attributes
```

`Page` knows how to serialise itself to a dict, to Markdown and to HTML; `Table`
additionally to OTSL. `page_from_dict` reads it all back, so validation, statistics and
evaluation consume exactly what the writers produced.

`Sample` = image + `Page` + `Provenance`. Provenance records the seed, font, background,
renderer, shaper, template and every degradation applied — enough to answer "which
setting produced this failure mode?" months later.

## `ocrsmith.core.rendering`

Turns a string into pixels **and** into the annotation describing them, in one pass.

Words are drawn one at a time rather than a line at a time. That costs a little speed and
buys exact per-word boxes, and it is typographically safe for Arabic because letters join
only *inside* a word, never across a space — each word is still shaped as a unit.

- `bidi_layout.py` — word-level directional runs decide the left-to-right order words are
  drawn in, while the annotation keeps logical order. `"سنة 2024 OCR"` places its Latin
  run the right way round.
- `wrapping.py` — lossless. No word is dropped, oversized words are broken rather than
  overflowed, and lines that exceed the height budget are *reported* so the caller can
  trim the label to match the pixels.
- `metrics.py` — measures the *visual* form, and caches per `(font, size, shaper)`.
  Rebuilding metrics per block discards that cache between blocks and costs about 3× the
  total layout time.

## `ocrsmith.core.documents`

Content in, pages out.

- `content.py` — `DocumentBuilder` / `DocumentContent` separate *what a document says*
  from *how it looks*. The same content laid out in one column or two produces identical
  markup ground truth, which is exactly the invariance a document model should learn.
- `page_spec.py` — canvases derived from paper size and DPI, with margins, multi-column
  layout (right column first for RTL) and reserved header/footer bands.
- `typography.py` — one coherent font family per document, with per-role sizes and
  spacing. Documents are typographically consistent; a page that picks a random face per
  paragraph looks like a font catalogue.
- `flow.py` — pours blocks down columns and onto further pages. **Prose splits, structure
  does not**: a paragraph continues on the next page, but a heading, table or figure moves
  whole. Nothing is clipped; a block that fits nowhere is dropped and reported rather than
  producing a blank page.
- `table_renderer.py` — content-derived column widths, in-cell wrapping, RTL column
  mirroring, five border styles, and a box for every cell.
- `templates.py` — six genres behind a weighted registry. Genre matters: a model trained
  only on flowing articles never learns that a receipt has right-aligned amounts or that a
  form is label/value pairs.

## `ocrsmith.core.degradations`

The gap between a clean render and a real capture. A degradation takes the annotation as
well as the image and returns both.

- **Geometric** (`Rotation`, `PerspectiveWarp`) derive an explicit forward point mapping
  and push every box, word and table cell through it. Rotated words gain polygons, because
  an axis-aligned box no longer describes them tightly.
- **Photometric** (sixteen of them) are chosen for physical cause rather than visual
  effect: toner spreading into fibres, text showing through a thin sheet, the resolution
  loss that destroys Arabic diacritics and that blur does not reproduce, a phone camera
  between a page and a ceiling light.
- **Presets** order them along the physical capture chain — optics, then shading, then
  sampling and compression.

## `ocrsmith.pipeline`

- `factory.py` — `SampleFactory.create(index)` seeds a private RNG from
  `(config.seed, index)` and yields one `Sample` per page. Nothing reads global random
  state, so samples are independent of generation order.
- `runner.py` — generators end to end; work split into shards; each worker generates *and
  writes* its own shard, because shipping images back through a pickle queue would cap
  throughput at one core's worth of serialisation. Completed shards are marked, so an
  interrupted job resumes.

## `ocrsmith.quality`

Validators for the failures that do not raise, run as a gate inside the pipeline. A shard
that rejects more than `quality.max_rejection_rate` aborts: that is a configuration
problem, and writing a shard full of holes would hide it.

`DatasetStats` accumulates the distributions that define the corpus in constant memory and
renders a dataset-card fragment.

## `ocrsmith.datasets`

Loaders in, writers out. Six writers behind one `SampleSink` protocol, all writing
incrementally so a killed run leaves valid shards rather than a truncated file.

## `ocrsmith.evaluation`

Metrics and a harness, in the same repository as the generator on purpose: a benchmark
whose ground truth came from a different codebase than the one under test is where silent
label conventions hide.
