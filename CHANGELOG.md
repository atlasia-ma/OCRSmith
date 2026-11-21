# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A HarfBuzz text engine, and with it Arabic handwriting at all.** `ocrsmith[shaping]`
  brings `uharfbuzz` and `freetype-py`; OCRSmith then applies the font's own `GSUB`/`GPOS`
  and rasterises the resulting glyph indices itself, the way a browser does.

  This was not an optimisation, it was the removal of a hard blocker. Without Raqm the
  fallback rewrites Arabic into the presentation-form block (U+FE70-FEFF), which is
  deprecated and which modern faces do not implement — they join through `GSUB` and carry
  no FExx glyphs. The coverage gate therefore, correctly, refused them. On the bundled
  pool that was 32 of 101 faces, and **every handwriting and calligraphic face without
  exception**: Aref Ruqaa, the standard ruqʿah hand, has 110 Arabic glyphs and zero
  presentation forms. Arabic handwriting synthesis was not poor, it was impossible, and
  the only symptom was a `handwriting_families` list that came back empty.

  With the engine: 92 of 101 faces usable instead of 69, 115 of 126 once the hands are
  fetched, correct `GPOS` mark placement (a shadda anchored to its letter rather than
  stacked on the advance width), and real ligatures. It costs about 30% of line-mode
  throughput, 88 to 62 strips per second. Shaping is per word, which is the existing
  contract and what keeps word boxes exact — letters never join across a space.

  Raqm remains supported and is tried second. It is also harder to obtain than its
  documentation suggests: neither the conda Pillow nor the current PyPI Windows wheel has
  it working, because the bundled libraqm looks for a `fribidi.dll` that is not shipped.
  The `shaping` extra needs no system libraries.

- **`fetch-fonts --hands` and `--category`.** `--hands` fetches the families whose
  letterforms come from a written hand. Deliberately wider than `--category handwriting`,
  because Google Fonts puts exactly one Arabic family in that category while the styles
  people actually write in — ruqʿah above all — are filed under Serif and Display. It
  returns 11 families where the category returns 1.

- **A line renderer, for recognition data.** `render.mode: line` draws single recognition
  strips directly instead of laying out a page and cutting it up: 11 ms per sample against
  664 ms for a page, because a strip is about a thousandth of the pixels of an A4 page at
  200 dpi. It is not a separate, simpler generator — it shares the font pool, the
  backgrounds, the text renderer and the degradation presets, so a strip carries the same
  per-word annotation and exports in the same formats. `build_factory(config)` picks the
  renderer and is the only place that knows which is running.

  Two things a strip generator usually lacks. The first is **neighbour bleed**: a sliver of
  the line above or below, clipped into the margins and deliberately absent from the label,
  because a real crop out of a real detector contains ink belonging to another line, and a
  corpus of perfectly isolated strips teaches a model that the first and last rows of
  pixels are always background. The second is the **content distribution**. A recogniser is
  not fed sentences, it is fed whatever a detector cut out: single words from form fields,
  a reference number with no sentence around it, unrelated words from a table row, a word
  sliced in half by a box edge. Those are sampled explicitly rather than inherited from the
  shape of the corpus.

- **Line-scale degradation presets.** A preset encodes a physical scale as well as a
  capture condition. The page presets are written for a sheet of paper, and running them
  over a 40-pixel strip does not degrade it, it destroys it: a vignette across a strip is a
  horizontal gradient, a coffee stain is the whole image, a page curl bends a page that is
  not there, and a 4-degree skew throws the text off a canvas one line box tall.
  `build_preset(name, scale="line")` returns the same conditions at crop scale. The names
  match, so a corpus specification does not have to be rewritten to switch modes.

- **`ocrsmith bench`.** Single-core throughput for either renderer, with nothing written to
  disk, so the number is what the renderer costs rather than what the storage costs. It
  warms up first, because the first call on a machine also pays for the font coverage cache
  and that belongs to the installation rather than to the renderer.

- **A second colour.** Documents were uniformly black on grey. A per-document accent is now
  sampled from a short list of colours documents are actually printed in — corporate navy,
  petrol, maroon, forest, ochre, aubergine — and reaches the title, usually the headings,
  and the table rules, so the sheet stays coherent the way a real house style does. Chosen
  once per document, never per block, and never on dark stock, where the ink is already
  light and a navy heading would be unreadable. Pale header and zebra bands come back with
  it, but only on plain stock: the reason they were removed in the first place was that
  shaded banding over a photographed surface reads as a paste-up, and that is still true.

- **`generate` reports what the quality gate threw away**, with the check that rejected it.
  A run that silently discards a quarter of its samples is a configuration to fix, not a
  cost to absorb, and a bare page count gives no way to know.

### Changed

- **Documents generate about 2.3x faster — 1.52 s to 0.66 s per page on the default config**
  — from four structural changes rather than micro-optimisation.

  *Glyph coverage is computed once per face and cached on disk.* It was 38% of wall clock.
  The lazy per-code-point check needed the font's `glyf`, `cmap`, `post` and `hmtx` tables
  decompiled, and the bounded cache of opened fonts had 64 slots for a pool of 101 faces, so
  it thrashed and rebuilt megabytes of tables it was about to need again. Every worker
  process repeated the whole warm-up, so the cost multiplied by the worker count instead of
  amortising across it. Coverage is now a bulk set per face, keyed by the font's path, size
  and mtime, under `OCRSMITH_CACHE_DIR`. Pool construction went from 7.9 s to 0.15 s, and
  `_has_ink` reads the contour count out of the parsed glyph rather than tracing the outline,
  which is 8x cheaper for an identical answer.

  *Skew and perspective compose into one homography and one resample.* Two full-page
  resamples were 52% of what was left — more than layout, text rendering and every other
  degradation together. Both are homographies and homographies compose, so `PlanarCapture`
  is the same transform rather than an approximation of it, and it is also *sharper*,
  because the page is interpolated once instead of twice. `Rotation` and `PerspectiveWarp`
  remain for a caller who wants one effect alone.

  *The smooth shading stages compose into a single field.* Shadow, vignette, glare, stains,
  creases and uneven light are all `pixel * gain + bias` with slowly-varying coefficients.
  Six of them in a row is six passes over a multi-megapixel array to apply something that is
  almost constant over any hundred pixels. `FieldDegradation` builds the field at a quarter
  resolution, the pipeline composes every consecutive one into a single pair, and it is
  upsampled and applied once. The fold is exact: `(g1, b1)` then `(g2, b2)` is
  `x*g1*g2 + b1*g2 + b2`.

  *Stages share one float32 buffer.* Each one used to decode the page into an array, do a
  few microseconds of work, clip, round back to `uint8` and re-wrap — about 50 ms per round
  trip, so a thirteen-stage `photo` preset spent two thirds of a second per page converting
  between two representations of the same pixels. `Surface` holds whichever was last written
  and converts only at a boundary. Quantisation now happens once, which also means a shadow
  that drives a pixel below zero and a glare that brings it back no longer lose the
  difference.

- **The capture chain is ordered the way a camera works.** `Downscale` modelled the sensor
  but ran near the end of the `photo` preset, after the blur and before only the noise and
  the compression. It now sits where the sensor is — after the optics, before the noise —
  which is both the physically correct place and the reason the rest of the preset runs on a
  fifth to four fifths of the pixels.

- **The quality gate matches the renderer.** `default_validators(scale)` takes "page" or
  "line". Only one check differs and it was quietly destructive: `NonEmptyText` requires
  eight characters and two words, which on a page means the render failed and on a strip
  means a one-word crop — the normal shape of a form field, a table cell or a caption.
  The page gate rejected 22% of line samples, and not at random: it removed precisely the
  short, context-free strings the line renderer exists to produce and for which a model has
  no language prior. Line rejection fell from 27% to 6%, the remainder being `MinContrast`
  legitimately catching washed-out strips.

- **`doctor` reports how many faces can actually draw the script**, not how many are on
  disk. Without Raqm the renderer is handed Arabic presentation forms rather than base
  letters, and a modern face that joins through `GSUB` carries no glyphs for them — so it is
  installed, readable and unusable. On the bundled pool that is 73 of 101, and the file
  count said 101.

- **`generate` and `doctor` warn when the corpus is too thin to be worth rendering.**
  Leaving `text.source.type` at `inline` and generating a hundred thousand pages produces
  well-laid-out pages, exact labels, a quality gate that passes every one of them, and a
  vocabulary of forty words. Nothing failed, and the only symptom arrived weeks later as a
  model that read its training distribution perfectly and nothing else.

### Fixed

- **A space made every font ineligible.** Introduced while indexing coverage on the font
  pool: `FontPool.supporting` tested the raw probe string against the drawable code points,
  and no face carries an outline for U+0020. Every string containing a space therefore
  matched no font, and the run fell back to the single best-covering face without reporting
  anything — 78% of samples in one 300-sample run were set in Amiri-Bold. Whitespace and the
  zero-width formatting marks are now exempt through one shared `required_codepoints`, and
  the test that would have caught it asserts the regression directly rather than the rule.

- **Handwriting detection matched the opposite of handwriting.** `kufi` was in the
  filename hint list, so `NotoKufiArabic` was the only "handwriting" family the bundled
  pool contained — Kufi is the angular, geometric, architectural Arabic script, as far
  from a written hand as the script gets. Every page the generator set as handwritten was
  therefore set in a monumental display face. `lalezar`, `marhey`, `qahiri` and `shekari`
  were hinted too and are poster faces; `reem` matched Reem Kufi. Detection is now a
  curated list of families that are genuinely hand-derived, plus the Google Fonts
  category where a `fetch-fonts` manifest records one, plus a denylist that no hint can
  override.

- **Every diacritic was deleted from the pixels while the label kept it.**
  `arabic-reshaper` defaults `delete_harakat` to `True`, and OCRSmith constructed it with
  the defaults. On the no-Raqm path — which is the default on a Windows or conda Pillow,
  and is what `doctor` has always reported — the tashkeel was therefore removed before the
  string ever reached the rasteriser. The entire `text.diacritics` section was inert:
  `keep`, `partial` and `strip` rendered identically, and the first two shipped labels
  asserting vowels that were not drawn. A model trained on it learns to hallucinate
  tashkeel, and nothing raises at any point.

  This is the invariant the project is built around, so the fix comes with the end-to-end
  test rather than only the unit one: keeping and stripping must produce measurably
  different ink. Coverage is unaffected — the combining marks are in every Arabic face's
  `cmap`, so the usable pool is the same 69 of 101 either way — and because the probe is
  the shaped string, a face that genuinely lacked them would now be rejected rather than
  silently dropping them.

- **`text.normalization` was ignored for inline text sources.** `load_corpus_texts`
  returned `source.sentences` before reaching the normalisation step, so a corpus
  configured for Arabic-Indic numerals, unified alef or stripped tatweel got none of it as
  soon as it fell back to inline text — including the fallback a failed remote corpus
  lands on. Normalisation now runs on every path.

- **Synthetic numbers in line mode ignored the numeral system.** The corpus is converted
  at load time, but a `numeric` strip's digits are generated in the factory, so a corpus
  set to `arabic_indic` rendered western digits on precisely the samples whose entire
  content is digits.

- **Neighbour bleed pasted whole lines instead of slivers.** `RenderedText.place` clamps a
  negative origin to zero, so a neighbour positioned to hang off the top edge landed fully
  inside the strip: a second, completely legible line of text that the label did not mention.
  The sliver is now cut from the rendered neighbour before compositing, and is confined
  strictly inside the margin.

## [1.3.0] - 2026-08-13

### Changed

- **Degradations are about 1.6x faster on the default preset mix, and produce byte-identical
  output.** Profiling the per-page cost — the question left open by 1.2.1 — showed
  degradation was 75-90% of a page, and that most of it was waste rather than physics.
  Seven degradations built their fields from `np.mgrid[0:height, 0:width]`, two full-page
  int64 arrays (60 MB for A4 at 200 dpi) before a single arithmetic temporary, when every
  one of those fields is either separable or depends on a single axis. `Wrinkles` also
  computed both axes of a gradient and used one.

  Per preset, 12 pages each: `photo` 4.30 → 2.42 s/page, `archive` 3.54 → 2.40,
  `scan` 1.72 → 1.18, `fax` 1.53 → 1.03. The worst individual steps: `Folds` 1.18 → 0.33s,
  `PageCurl` 1.45 → 0.75s, `Wrinkles` 2.17 → 1.16s.

  A speed-up in a data generator is only legitimate if the data does not move, so all 38
  degradation configurations across the four presets were hashed — pixels and serialised
  annotation — before and after, on a fixed page and seed. All 38 identical. The
  equivalence is a test rather than a claim: `_axes` must equal the `np.mgrid` it replaced,
  exactly.

### Fixed

- **A page whose body was washed out shipped with a full transcription if any one block
  survived.** `MinContrast` reduced its per-block measurements with `max()`, so a faded
  archival page that kept its heading dark passed on the heading's 238 levels of contrast
  while every paragraph under it sat at the shade of the paper. The label claimed text no
  reader could recover. It now judges the page by the *fraction* of blocks below the
  threshold, like `LegibleLineHeight` beside it, and rejects above half.

  That alone changed nothing, because the quantity being reduced was also wrong. Contrast
  was a 10-90 percentile spread taken across a block, which measures whatever varies within
  it — and on a folded, stained page what varies is the shading. Text presence barely
  entered: on one page an erased caption scored 60.0 and a perfectly readable heading 61.9.
  Every block cleared the threshold and the page passed with nothing failing.

  Contrast is now measured at stroke scale — how far ink falls below the paper immediately
  around it, taking the local paper level from a dilation by roughly a stroke's width.
  Smooth shading subtracts out by construction. On that same page the readable heading
  moves to 205.2 and the erased caption to 36.0, a gap the old measure inverted. A `clean`
  page scores 214-239 on every block at 110, 200 and 300 dpi, so the window needs no
  scaling.

  The threshold follows from that scale: at 90, `clean`, `scan` and `fax` lose nothing,
  `photo` loses 2.9%, and `archive` loses the 25.7% of its pages that are unreadable. At
  120 it would start discarding a fifth of `photo`, which is dark but perfectly legible.

- **Tables on right-to-left pages were pinned to the left margin.** `_place_table`
  composited at `column.x0` whatever the direction, while text in the same column ranged
  right — so an Arabic page put its table against the left edge with every paragraph beside
  it starting at the right. `direction` was already threaded in for cell content; it now
  also decides where the block starts. Figures centred and text followed direction; tables
  were the only block type that ignored it.

## [1.2.1] - 2026-08-13

Both entries were found by re-running the benchmark after the 1.2.0 fixes, not by a test.
Neither raised anything: one reported success with no output, the other ran a job at a
fraction of the requested parallelism. They are the same kind of defect the rest of this
project exists to prevent — a wrong result that looks like a right one.

### Fixed

- **A parallel run whose every shard died reported success with zero pages.** Worker
  exceptions were collapsed into a bare counter and the exceptions themselves discarded,
  so `run_generation` returned a success-shaped result and the only surviving evidence was
  a number. `GenerationResult.shard_errors` now records `(shard, reason)` for every shard
  that died, the CLI prints each one, and a run where every shard failed and nothing was
  written raises instead of returning. Serial and parallel paths previously disagreed —
  serial propagated, parallel swallowed — and now share one seam.

### Added

- **`generate` says when `output.shard_size` is capping parallelism.** A shard is the unit
  of parallelism, so 400 documents in 250-document shards is two shards and therefore two
  processes however many workers were asked for. Measured: 4 workers over 2 shards gave a
  1.6x speedup; the same work over 8 shards gave 3.2x, and 12 workers over 16 shards gave
  5.3x. The only symptom of the capped case is that the job finishes late, so the advice —
  including the `shard_size` that would use every worker — is printed before the run.
- Scaling table, the `shard_size ≈ num_samples / workers` rule of thumb and the ~210 MB
  per-worker memory budget in the README.
- A test for a claim that had been resting on an argument: that the worker count does not
  change the dataset. Verified against real worker processes, record by record.

## [1.2.0] - 2026-08-13

### Added

- **`ocrsmith.validation`** — the two claims the rest of the project only asserted, made
  measurable.
  - `ocrsmith compare <synthetic> <real>` measures the sim-to-real gap feature by feature
    across ten properties (ink fraction, stroke width, contrast, edge density,
    high-frequency energy, JPEG block noise, illumination range, …). Every feature maps
    onto a generator knob, so the report says **what to change** — "stroke_width is 22%
    lower, widen the weight distribution or raise ink spread" — rather than reporting an
    abstract divergence. Reports both a signed effect size and a distribution overlap,
    because a corpus can have the right mean and the wrong spread and a mean-only measure
    would call that a match.
  - `ocrsmith ablate <name>` builds the corpora that answer "does this feature actually
    help?". Arms differ in exactly one knob and **share the base seed**, so a downstream
    accuracy difference can only be attributed to that knob — if the arms drew different
    text, the experiment would answer nothing. Presets for degradations, fonts, layout and
    diacritics. Training is deliberately left to the caller.

## [1.1.1] - 2026-08-13

### Fixed

- **An Arabic page with more than one column was laid out to be read in the wrong order.**
  With the default `text.direction: auto`, the configured direction is `None`; the text
  still came out right-to-left from the content, but the *page spec* fell back to
  left-to-right, so `column_boxes()` ordered the columns left-first. The result was a page
  whose text ran right-to-left inside columns that were read left-to-right. Nothing
  raised, the pixels looked plausible, and only the reading order was wrong. The page spec
  now follows the content's direction when none is configured.

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

[1.2.0]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.2.0
[1.1.1]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.1.1
[1.1.0]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.1.0
[1.0.2]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.0.2
[1.0.1]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.0.1
[1.0.0]: https://github.com/atlasia-ma/OCRSmith/releases/tag/v1.0.0
