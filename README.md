<div align="center">

# OCRSmith

**A synthetic document forge for training OCR and document-understanding models — Arabic first.**

[![CI](https://github.com/atlasia-ma/OCRSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/atlasia-ma/OCRSmith/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

> **OCRSmith produced the training data behind [AtlasOCR](https://huggingface.co/blog/imomayiz/atlasocr)**, the first
> open-source Moroccan Darija OCR model — 86% of its corpus, ~10.7M words. AtlasOCR beats
> comparable models on [KITAB-Bench](https://arxiv.org/pdf/2502.14949).

OCRSmith generates **whole documents**, not cropped text lines: multi-column pages with
titles, tables, figures, forms and running headers, degraded to look like something a
scanner or a phone actually produced — and it emits the ground truth for every objective
those pages can supervise.

One rendered page gives you, from the same pass:

| Objective | What you get |
| --- | --- |
| Recognition | Line and word crops with logical-order text |
| Detection | Word, line and region boxes; polygons where the page is warped |
| Layout analysis | Typed regions (`title`, `table`, `figure`, `key_value`, …) in reading order |
| Document → markup | The page serialised back to **Markdown** and **HTML** |
| Table structure | Cell grid as HTML *and* **OTSL** |
| Chart → JSON | The series values the chart was drawn from |
| Formula → LaTeX | The expression tree the formula was typeset from |

Everything is reproducible from a seed, streamed rather than materialised, and validated
before it reaches the dataset.

## Why another generator

Most synthetic OCR data is a line of text on a noisy background. That teaches a model to
read a crop; it does not teach it to read a page. And most generators built for Latin
script get Arabic subtly wrong in ways that are invisible until training plateaus:

- **Logical vs visual order.** The label a model must predict is the *logical* string; the
  pixels are in *visual* order. OCRSmith keeps them apart explicitly and never confuses
  them — see [`ocrsmith/text/shaping.py`](src/ocrsmith/text/shaping.py).
- **Missing glyphs.** A font that cannot draw a character renders a blank or a tofu box
  while the label still claims it. OCRSmith checks the font's `cmap` before choosing it.
- **Boxes that do not follow the pixels.** A rotation that moves ink but leaves the
  annotation behind produces a dataset that looks fine and trains a detector to be
  systematically wrong. Here every geometric degradation maps the annotation through the
  same transform.
- **Silent truncation.** A wrapper that drops the tail of a paragraph produces an image
  whose label claims text that was never drawn. OCRSmith's wrapping is lossless, and text
  that does not fit is *reported*, not discarded.
- **Uniform vocalisation.** Real Arabic is *partially* diacritised and the proportion
  varies by genre. OCRSmith varies it per document — and never invents marks, because
  diacritising bare text would make the label assert vowels nobody wrote.

## Install

```bash
pip install -e ".[data,dev]"
ocrsmith fetch-fonts --subset arabic     # 57 open-licensed families, licences included
```

`data` adds pandas/pyarrow/`datasets` for tabular and Hugging Face corpora and for
Parquet output; the core install stays light.

Check the machine can produce correct Arabic:

```bash
ocrsmith doctor
```

Pillow built with Raqm delegates shaping to HarfBuzz; without it OCRSmith falls back to
`arabic-reshaper` + `python-bidi`. Both paths produce the same labels — but Raqm is worth
having: it raises the usable font pool from **85 to 203 faces**, because most modern
Arabic families join via GSUB and carry no presentation-form glyphs.

```bash
conda install -c conda-forge pillow      # usually ships with Raqm; pip wheels usually do not
```

## Quick start

```bash
ocrsmith preview --count 3 --boxes --output outputs/preview
```

```bash
ocrsmith generate --num-samples 10000 --workers 8 --format webdataset -o data/train
```

```bash
ocrsmith stats data/train --markdown data/train/DATASET_CARD.md
ocrsmith validate data/train
```

From Python:

```python
from ocrsmith import load_config, run_generation

config = load_config("configs/darija_scan.yaml")
result = run_generation(config)
print(result.to_dict())
```

Or drive the generator directly and keep the samples in memory:

```python
from ocrsmith import SampleFactory, load_config

factory = SampleFactory(load_config())
for sample in factory.create(index=0):
    sample.image.save(f"{sample.id}.png")
    print(sample.page.to_markdown())
    for word in sample.page.iter_words():
        print(word.text, word.bbox.as_tuple())
```

## What comes out

```
data/train/
├── images/                     # one PNG/JPEG per page
├── annotations-00000.jsonl     # one record per page
└── .shard-00000.done           # completion marker, so a rerun resumes
```

Each record carries the full annotation tree plus both markup serialisations:

```jsonc
{
  "id": "00000042_01",
  "image_path": "images/00000042_01.png",
  "text": "تقرير سنوي\nيحتوي هذا التقرير على جداول وأرقام…",
  "markdown": "# تقرير سنوي\n\nيحتوي هذا التقرير…",
  "html": "<h1>تقرير سنوي</h1>\n<p>يحتوي هذا التقرير…</p>",
  "page": {
    "width": 1240, "height": 1754, "direction": "rtl",
    "regions": [
      {
        "type": "title", "bbox": [...], "reading_order": 0,
        "lines": [{ "text": "تقرير سنوي", "bbox": [...], "direction": "rtl",
                    "words": [{ "text": "تقرير", "bbox": [...] }] }]
      },
      { "type": "table", "bbox": [...], "table": { "rows": 4, "cols": 3, "cells": [...] } }
    ]
  },
  "provenance": {
    "seed": 918273645, "template": "report", "font_path": ".../Amiri-Regular.ttf",
    "background": "paper", "degradations": [{ "name": "PerspectiveWarp", "magnitude": 0.03 }],
    "extra": { "index": 42, "page": 1, "preset": "photo", "dpi": 150, "columns": 2 }
  }
}
```

### Output formats

`--format` selects the writer: `jsonl`, `parquet`, `webdataset`, `coco`, `paddleocr`,
`chat`. See [docs/formats.md](docs/formats.md).

## How it fits together

```
config/         one validated GenerationConfig describing a whole corpus
  ↓
text/           script detection · normalisation policy · bidi + shaping · glyph coverage
core/documents/ content model → typography → flow layout → pages
core/rendering/ text drawn word by word, emitting word boxes in the same pass
core/degradations/ capture conditions; geometric ones move the annotation too
  ↓
domain/         Page → Region → Line → Word (+ Table), immutable and serialisable
  ↓
quality/        validate the page, then count what the corpus contains
datasets/       six writers behind one SampleSink protocol
evaluation/     CER · WER · NED · table similarity · detection P/R/F1
```

Longer version: [docs/architecture.md](docs/architecture.md).

## What it generates

**Ten document genres**, weighted so a corpus is balanced deliberately rather than
accidentally:

| Genre | Why it is there |
| --- | --- |
| `article`, `newspaper`, `letter` | The shapes that dominate real corpora |
| `report` | Tables, charts and captions in context |
| `paper` | Displayed equations interleaved with prose |
| `form`, `invoice` | Label/value pairs, line-item tables, real reference numbers |
| `contents` | Dot leaders — runs of identical glyphs that models miscount |
| `slide` | Very large type, very little of it |
| `notes` | Handwriting-style setting, for the handwriting-heavy Arabic benchmarks |

**Five capture conditions** — `clean`, `scan`, `photo`, `fax`, `archive` — modelling
physical cause rather than visual effect: toner spreading into fibres, text showing
through a thin sheet, a page bending away from the sensor, uneven lighting, wrinkles,
stains, folds, and the resolution loss that destroys diacritics.

## Configuring a corpus

A corpus is defined by its *distributions*, so every per-sample choice is a range or a
weight map rather than a fixed value:

```yaml
page:
  papers: { a4: 4.0, a5: 1.0, letter: 1.0 }
  dpi_range: [110, 200]
  columns: { 1: 3.0, 2: 1.0 }

templates:
  weights: { article: 3.0, report: 2.0, newspaper: 1.5, letter: 1.0, form: 1.0, invoice: 1.0 }

degradations:
  presets: { clean: 1.0, scan: 4.0, photo: 3.0, fax: 0.5, archive: 1.0 }
```

Override anything from the command line:

```bash
ocrsmith generate --set page.columns='{"2":1}' --set degradations.presets='{"photo":1}'
```

Full reference: [docs/configuration.md](docs/configuration.md). Ready-made corpora:
[examples/configs](examples/configs).

## Reproducibility

Every sample's seed is *derived*, never drawn:

```python
sample_seed(index) = f(config.seed, index)
```

So sample 8 412 of a ten-million-page run can be regenerated on its own, months later,
without replaying anything — which is what makes a synthetic dataset debuggable.

```bash
ocrsmith preview --set run.start_index=8412 --count 1 --boxes
```

## Scale

- Generation is a **generator end to end**; a ten-million-page run holds one page in memory.
- Work is **sharded**, and each worker writes its own shard rather than shipping images
  back through a pickle queue.
- A completed shard is marked done, so an interrupted job **resumes** instead of restarting.
- Font metrics are cached per `(font, size, shaper)`, which is most of the cost of laying
  out a page.

```bash
ocrsmith generate -n 1000000 --workers 32 --format webdataset -o /mnt/data/ocr
```

**A shard is the unit of parallelism**, so `output.shard_size` sets the ceiling on workers:
400 documents in 250-document shards is two shards, and therefore two processes however
many you ask for. `generate` says so before the run rather than leaving you to infer it
from the clock. Measured on 400 documents, 6 physical cores:

| workers | shards | elapsed | speedup | peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 1371s | — | 535 MB |
| 4 | 2 | 849s | 1.6x | 826 MB |
| 4 | 8 | 427s | **3.2x** | 1.3 GB |
| 12 | 16 | 259s | **5.3x** | 2.5 GB |

Rule of thumb: `shard_size ≈ num_samples / workers`, and budget ~210 MB per worker.
Every row above produced the same 378 pages and the same 22 rejections: a sample derives
from `(seed, index)`, never from execution order, so the worker count does not change the
dataset.

## Validating the corpus

The sim-to-real gap is usually answered with an assertion. Measure it instead:

```bash
ocrsmith compare data/train/images /path/to/real/scans -m gap.md
```

Ten features, each mapping onto a generator knob, so the report names **what to change**:

| feature | synthetic | real | gap | overlap | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `high_frequency_energy` | 20.6 | 14.3 | +44% | 25% | mismatched |
| `stroke_width` | 5.37 | 6.84 | -22% | 50% | mismatched |
| `illumination_range` | 209 | 208 | +1% | 50% | matched |

> `high_frequency_energy` is synthetic higher (+44%) — Blur, Downscale and GaussianNoise strength
> `stroke_width` is synthetic lower (-22%) — font weight distribution, InkSpread / InkErosion probability

And when you add a feature, measure whether it helps rather than assuming:

```bash
ocrsmith ablate degradations --generate -n 20000
```

Every arm shares the base seed, so the arms differ in exactly the knob under test.
Train the same model on each and compare on a held-out **real** benchmark — a synthetic
test set would only reward the generator's own biases.

## Evaluating a model on it

```bash
ocrsmith generate -n 2000 -o data/bench --set seed=777
# … run your model, write {sample_id: prediction} to predictions.json …
ocrsmith evaluate data/bench predictions.json --ignore-diacritics
```

```python
from ocrsmith.evaluation import evaluate, load_references

report = evaluate(load_references("data/bench", target="markdown"), predictions)
print(report.to_markdown())
print(report.worst(10))          # where to start debugging
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Tests are the specification: run `pytest` from a
fresh clone, no install required.

## Licence and credits

MIT — see [LICENSE](LICENSE). Bundled fonts are distributed under their own licences
(SIL OFL for the Noto, Amiri, Mada, Fustat, Kufam, Mirza and Vazirmatn families).

Built for the [AtlasIA](https://github.com/atlasia-ma) effort to bring Moroccan Darija
into open models.
