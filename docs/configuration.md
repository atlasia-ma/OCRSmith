# Configuration reference

A config file is a complete corpus specification. Anything not given takes the default,
so the smallest useful file is a few lines.

```bash
ocrsmith show-config -c my_corpus.yaml     # print the fully resolved config
ocrsmith generate -c my_corpus.yaml --set run.workers=16
```

Overrides use dotted keys and JSON values: `--set page.columns='{"2":1}'`,
`--set fonts.size_range='[22,34]'`, `--set seed=7`.

---

## `seed`

Base seed for the whole run. Each sample's seed is *derived* from it and the sample index,
so any sample is reproducible on its own. `null` means non-reproducible.

## `fonts`

| Key | Default | Meaning |
| --- | --- | --- |
| `paths` | `["assets/fonts"]` | Files or directories to search for `.ttf`/`.otf`/`.ttc`. |
| `size_range` | `[18, 30]` | Body text size in pixels; every other role is derived from it. |
| `require_full_coverage` | `true` | Reject a face that cannot draw every character of the document. |
| `include` / `exclude` | `[]` | Filename substring filters, e.g. `include: ["Noto", "Amiri"]`. |

Leaving `require_full_coverage` on is strongly recommended: a missing glyph renders as a
blank or a tofu box while the label still claims the character.

## `text`

### `text.source`

| Key | Meaning |
| --- | --- |
| `type` | `csv`, `parquet`, `huggingface` or `inline`. |
| `path` | File path or Hugging Face dataset id. |
| `column` | Text column. |
| `title_column` | Optional title column. |
| `split`, `name` | Hugging Face split and configuration. |
| `limit` | Stop after this many rows — useful for smoke runs. |
| `sentences` | Used by `inline`, and as the fallback if a source yields nothing. |

### `text.normalization`

Every option here **rewrites the ground truth**, so all default to off except whitespace
collapsing.

| Key | Effect |
| --- | --- |
| `strip_diacritics` | Remove tashkeel and Quranic marks. |
| `strip_tatweel` | Remove kashida elongation. |
| `unify_alef` | `أ إ آ ٱ` → `ا`. |
| `unify_ya` | `ى` → `ي`. |
| `numerals` | `keep`, `western`, `arabic_indic`, `eastern_arabic_indic`. |

Training a model to *drop* diacritics is a legitimate choice; making it accidentally is
not. Decide deliberately.

### `text.direction`

`auto` (read from the text), `rtl` or `ltr`.

## `page`

| Key | Default | Meaning |
| --- | --- | --- |
| `papers` | `{a4: 4, a5: 1, letter: 1}` | Weighted paper sizes. Also `a3`, `a6`, `legal`, `receipt`, `id_card`. |
| `dpi_range` | `[110, 200]` | Scanning resolution. |
| `margin_mm_range` | `[12, 28]` | Page margins in millimetres. |
| `columns` | `{1: 3, 2: 1}` | Weighted column counts. |
| `landscape_probability` | `0.05` | |
| `header_probability` | `0.35` | Chance of a running header band. |
| `footer_probability` | `0.45` | Chance of a footer / page number. |
| `max_pages` | `3` | Ceiling on pages per document. |

### `page.background`

| Key | Default | Meaning |
| --- | --- | --- |
| `kinds` | `{paper: 3, solid: 1, gradient: 0.3, image: 0}` | Weighted background kinds. |
| `image_paths` | `[]` | Required if `image` has a positive weight. |
| `tint_range` | `[[238,234,226],[255,255,255]]` | Paper tint endpoints, interpolated together. |

## `templates.weights`

`article`, `report`, `newspaper`, `letter`, `form`, `invoice`. Set a weight to `0` to
exclude a genre.

## `degradations.presets`

| Preset | What it models |
| --- | --- |
| `clean` | Born-digital render, no capture stage. |
| `scan` | Flatbed: slight skew, paper texture, toner spread, mild compression. |
| `photo` | Phone: perspective, uneven light, glare, motion blur, heavy JPEG. |
| `fax` | Low resolution, high contrast, broken strokes. |
| `archive` | Aged paper: stains, folds, bleed-through, faded contrast. |

Keeping a share of `clean` is useful for curriculum training — start easy, then degrade.

## `quality`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Validate every page before it reaches the dataset. |
| `max_rejection_rate` | `0.5` | Abort a shard rejecting more than this fraction. |

A high rejection rate is a configuration problem, not bad luck. The abort is deliberate:
a shard that is mostly holes would otherwise look like a working dataset.

## `output`

| Key | Default | Meaning |
| --- | --- | --- |
| `dir` | `outputs/dataset` | |
| `format` | `jsonl` | `jsonl`, `parquet`, `webdataset`, `coco`, `paddleocr`, `chat`. |
| `image_format` | `png` | `png`, `jpeg`, `webp`. |
| `image_quality` | `92` | For lossy formats. |
| `shard_size` | `500` | Documents per shard; the unit of parallelism and of resume. |

## `run`

| Key | Default | Meaning |
| --- | --- | --- |
| `num_samples` | `100` | **Documents**, not pages — one document may produce several pages. |
| `workers` | `1` | Worker processes. |
| `start_index` | `0` | Skip earlier indices; lets a run continue where another stopped. |
| `max_consecutive_failures` | `50` | Abort rather than spin when generation keeps failing. |

---

## Recipes

**Dense multi-column newspaper pages, photographed**

```yaml
page:
  papers: { a3: 1.0 }
  columns: { 3: 3.0, 4: 1.0 }
  dpi_range: [150, 220]
templates:
  weights: { newspaper: 1.0 }
degradations:
  presets: { photo: 1.0 }
```

**Receipts and forms for key–value extraction**

```yaml
page:
  papers: { receipt: 2.0, a5: 1.0 }
  columns: { 1: 1.0 }
templates:
  weights: { invoice: 2.0, form: 1.0 }
degradations:
  presets: { photo: 2.0, fax: 1.0 }
```

**A clean, undiacritised recognition corpus**

```yaml
text:
  normalization: { strip_diacritics: true, numerals: western }
degradations:
  presets: { clean: 1.0, scan: 1.0 }
output:
  format: paddleocr
```
