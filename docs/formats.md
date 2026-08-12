# Output formats

One rendered page can supervise several different objectives, and each community expects
its own file layout. Rather than pick one and make everyone else convert, the same
`Sample` is serialised through a small `SampleSink` protocol.

All writers are incremental, so a run that is killed leaves valid shards behind rather
than one truncated file.

```bash
ocrsmith generate --format webdataset -o data/train
```

---

## `jsonl` — the default

```
data/train/
├── images/00000042_01.png
└── annotations-00000.jsonl
```

One JSON object per page, carrying the full annotation tree plus `text`, `markdown` and
`html`. This is the format `ocrsmith validate`, `ocrsmith stats` and `ocrsmith evaluate`
read.

```python
import json
for line in open("data/train/annotations-00000.jsonl", encoding="utf-8"):
    record = json.loads(line)
    for region in record["page"]["regions"]:
        for line_ann in region.get("lines", []):
            print(line_ann["text"], line_ann["bbox"])
```

## `parquet`

The same records, columnar, one file per shard. Nested annotations are stored as JSON
strings so the schema stays stable across pages of very different structure. Needs the
`data` extra.

## `webdataset`

Tar shards of `(image, json)` pairs keyed by sample id. Sequential reads off a tar are
what keep data loading ahead of a GPU once the dataset no longer fits on local disk.

```python
import webdataset as wds

dataset = (
    wds.WebDataset("data/train/shard-{00000..00099}.tar")
    .decode("pil")
    .to_tuple("jpg", "json")
)
```

## `coco`

Object-detection layout. Every word, every line and every region becomes an instance;
categories are `word`, `line` and one per region type. Polygons appear in `segmentation`
where a degradation warped the page.

Use it to train a text detector (`word`/`line` categories) or a layout model (region
categories) from the same file.

## `paddleocr`

The two-file layout PaddleOCR expects:

- `det_label-00000.txt` — `image_path \t [{"transcription": ..., "points": [[x,y] × 4]}]`
- `rec_label-00000.txt` — `line_crops/xxx.png \t text`, with the crops written alongside

Detection and recognition training data from one pass.

## `chat`

Instruction-tuning records for a vision-language model:

```jsonc
{
  "id": "00000042_01",
  "image": "images/00000042_01.png",
  "messages": [
    {"role": "user", "content": [{"type": "image"},
                                 {"type": "text", "text": "Transcribe this document to Markdown, preserving its structure."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "# تقرير سنوي\n\n…"}]}
  ]
}
```

The assistant turn is the page's Markdown, which is what document-to-markup models are
trained against.

---

## Writing your own

Implement `SampleSink` and register it:

```python
from ocrsmith.datasets.writers import SampleSink, _SINKS

class MySink(SampleSink):
    def __init__(self, directory, *, shard=0, **kwargs):
        self.path = Path(directory) / f"shard-{shard:05d}.txt"

    def open(self):
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, sample):
        self._handle.write(f"{sample.id}\t{sample.text}\n")

    def close(self):
        self._handle.close()

_SINKS["mine"] = MySink
```
