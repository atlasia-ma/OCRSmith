# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-08-13

Six tracks of work chosen from a survey of what the current OCR literature says synthetic
data is missing, and from the limitations AtlasOCR — trained on OCRSmith output — reports
about itself.

### Added

- **Non-prose corpus content.** A sentence corpus contains no dates, no reference numbers
  and no partial words — precisely the cases where a recogniser has no language model to
  lean on. `FieldGenerator` produces dates (three formats), amounts with currencies,
  reference codes, phone numbers and percentages, in whichever numeral system the config
  asks for; `CorpusTextProvider.fragment()` yields partial words as an occlusion or crop
  boundary produces, and `.contextless()` yields unrelated words in sequence. The form and
  invoice templates now use them, because that is what those documents actually hold.

- **Three more genres and a handwriting setting.** Genre coverage is not decoration: a
  model trained only on flowing prose miscounts dot leaders, misreads slide-sized type,
  and transfers poorly to the handwriting-heavy Arabic benchmarks.
  - `contents` — a table of contents with dot leaders, which are their own recognition
    problem: a long run of identical glyphs that models routinely miscount.
  - `slide` — a headline and a few short bullets set large; a distinct visual regime that
    dense-prose training reads poorly.
  - `notes` — handwritten notes. `TypographySampler(handwritten=True)` prefers a
    handwriting or calligraphic family and loosens the setting with baseline and word-gap
    jitter, because a hand holds neither a constant baseline nor an even word gap. This is
    **not** a substitute for real handwriting data — the letterforms still come from a
    font — but it covers the layout and the visual regime.

- **Charts with their data** (`ocrsmith.core.documents.charts`). Chart-to-JSON is a
  first-class task in the Arabic document benchmarks and no synthetic generator covered it.
  Bar, horizontal-bar, line and pie charts are drawn *from* their series values, so the
  JSON ground truth cannot drift from the picture. Axis and title labels are real
  annotated text, so a chart supervises recognition and detection as well.
- **Formulas typeset from a tree** (`ocrsmith.core.documents.formulas`). Formula
  conversion is the largest gain category in the document-parsing benchmarks, and a
  `FORMULA` region previously rendered as plain text. A small typesetter handles
  fractions, powers, indices, radicals and sums/integrals with limits, positioned on a
  baseline — and emits the LaTeX from the same tree, so the two cannot disagree. No LaTeX
  toolchain required.
  - `choose_math_font` applies glyph coverage to mathematics: most text faces have no
    summation sign, and choosing blindly produced a formula of empty boxes whose LaTeX
    confidently asserted a sum.
- `RegionType.CHART`, `DocumentBuilder.chart()` / `.formula()`, and a `paper` template
  that interleaves prose with displayed equations. Chart JSON and formula LaTeX both reach
  the Markdown ground truth.

- **Physical degradations** (`ocrsmith.core.degradations.physical`). Rendering-based
  synthesis produces a page lying perfectly flat under perfectly uniform light; real
  captures never do, and modelling the difference is worth several points on documents
  photographed in the wild.
  - `Wrinkles` — local displacement plus the shading its ridges catch.
  - `PageCurl` — a bound page bending away from the sensor, compressing text towards one
    edge and darkening into the gutter. Flat perspective warping cannot produce this.
  - `IlluminationField` — smooth, arbitrary lighting, distinct from `Shadow` (a linear
    ramp) and `Vignette` (radial and centred).

  All three are displacement or field effects rather than colour transforms, and the
  deformations carry the annotation through the same field — verified by measuring where
  the ink actually landed and requiring the boxes to be there (IoU > 0.85).

  The `photo` and `archive` presets now use them.

- **Diacritics control** (`ocrsmith.text.diacritics`). Arabic OCR handles vocalisation
  badly, and it is the first limitation AtlasOCR reports about itself. The cause is
  distributional: real Arabic is *partially* diacritised, and the proportion varies by
  genre. `DiacriticsPolicy` samples per document across four modes (`keep`, `strip`,
  `partial`, `mixed`), and records the kept fraction in provenance so a diacritics
  ablation is possible later. `DatasetStats` reports the corpus split across bare, partial
  and fully marked pages.

  Marks are only ever **removed**, never invented. Adding vocalisation to bare text needs
  a diacritiser model and would make the label assert vowels nobody wrote — a fabricated
  ground truth that looks entirely plausible. Point `text.source` at a diacritised corpus
  and let the policy vary it downwards.

- **`ocrsmith fetch-fonts`** — downloads open-licensed families from Google Fonts on
  demand. Font diversity is the highest-impact lever in synthetic text data, and a
  repository should not ship other people's typefaces. Only permissively licensed
  directories of `google/fonts` are used (`ofl`, `apache`, `ufl`), every family's licence
  file is downloaded beside its fonts, and a manifest records exactly what was taken so a
  dataset stays reproducible.
- **Variable-font expansion.** Roughly half the Arabic families on Google Fonts are
  variable, and a variable font renders only its default instance — so `light`, `regular`
  and `bold` of such a family all collapsed onto the same face. Families are now expanded
  into their named instances via `font_variations` / `load_font(..., variation=...)`, and
  `Face` carries the instance alongside the file.
- `DocumentContent.all_text`, covering table cells and list items.

Measured on the bundled fonts plus one `fetch-fonts --subset arabic` run:
**11 -> 57 families, 101 -> 381 drawable faces**, including 13 display and calligraphic
families of the kind that synthetic corpora usually lack entirely.

Note for non-Raqm builds: only 16 of the 105 fetched files carry Arabic presentation
forms, so the coverage gate (fixed in 1.0.2) correctly rejects most of them. Installing
Pillow with Raqm raises the usable pool from 85 to 203 faces.

## [1.0.2] - 2026-08-13

### Fixed

- **Font coverage was checked against the wrong string, so Arabic could render as tofu.**
  Coverage was judged on the *logical* text, but without Raqm the renderer draws Arabic
  *presentation forms* (U+FE70-FEFF). A modern OpenType face such as Fustat or Mada covers
  the base Arabic block while carrying no presentation-form glyphs at all — it joins
  letters via GSUB instead — so the probe reported 100% coverage and every glyph then
  rendered as an empty box, with the label still claiming the text. `FontPool` now shapes
  before probing, using the same shaper the renderer will use. Raqm builds are unaffected
  and correctly keep judging the logical form.
- **The coverage probe missed table cells and list items.** A table block's own text is
  empty; its content lives in the cells. An invoice whose prose was four words therefore
  chose a font on the strength of those four words and drew its entire table as tofu.
  Added `DocumentContent.all_text`, which includes cells and list items, and the pipeline
  now probes against it.
- **The font fallback silently discarded the guarantee.** When no face covered a document,
  the pipeline fell back to the *entire* pool and could hand the document a face that
  cannot draw its script at all. It now falls back to the single best-covering face.

Found by rendering sample documents and looking at them — every automated check passed
while the images were visibly broken.

## [1.0.1] - 2026-08-13

### Fixed

- **Shaping was 68% of generation time.** arabic-reshaper 3.0.0 guards its ligature-regex
  cache with `hasattr(self, '__ligatures_re')`, but writes the cache to
  `self.__ligatures_re` — which Python mangles to `_ArabicReshaper__ligatures_re` inside
  the class body, while the string literal passed to `hasattr` is not mangled. The guard
  therefore checks a name that is never set, and every call rebuilt the regex, re-reading
  ~290 configparser entries. Laying out one page called it ~1,800 times.

  Two mitigations, both contained to `ocrsmith.text.shaping`: results are cached (shaping
  is a pure function of the string), and the reshaper's cache is warmed once so the
  library's own guard fires from the second call. 2,000 distinct strings: 26s -> 0.36s.

### Changed

- Wrapping now measures a line the way the renderer draws it — summing word advances plus
  space advances — instead of measuring the whole candidate line as one shaped run. This
  is a correctness improvement as well as a speed one: the two could previously disagree
  about where a line ends. It also keys the measurement cache on *words*, which repeat,
  rather than on line prefixes, which never do.

Test suite runtime: 95s -> 20s. Generation throughput on a mixed Arabic corpus improved
~1.6x end to end; the remaining cost is genuine rasterisation and degradation work.

## [1.0.0] - 2026-08-12

The first release of OCRSmith as a document forge rather than a line-image generator.
The v0 engine produced one image with one string; v1 produces whole pages with word-level
boxes, typed layout regions, table structure and markup ground truth, and validates every
page before it reaches the dataset.

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

[1.1.0]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.1.0
[1.0.2]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.0.2
[1.0.1]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.0.1
[1.0.0]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.0.0
