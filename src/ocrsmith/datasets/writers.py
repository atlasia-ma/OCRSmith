"""Dataset writers.

One rendered page can supervise several different objectives, and each community expects
its own file layout. Rather than pick one and make everyone else convert, the same
`Sample` is serialised through a small `SampleSink` protocol:

* `jsonl` — images on disk plus one JSON object per page; the lingua franca.
* `parquet` — the same records in a columnar file, for large-scale loading.
* `webdataset` — tar shards of (image, json) pairs, the format that keeps GPUs fed when
  the dataset no longer fits on local disk.
* `coco` — object-detection layout with word boxes as instances.
* `paddleocr` — the two-file detection/recognition layout PaddleOCR expects.
* `chat` — image + instruction + markup answer, the shape used to fine-tune a
  vision-language model on document-to-markup.

Sinks are context managers and write incrementally, so a run that is killed halfway leaves
valid shards behind rather than one truncated file.
"""

from __future__ import annotations

import json
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..domain import RegionType, Sample

__all__ = [
    "ChatSink",
    "CocoSink",
    "JsonlSink",
    "PaddleOcrSink",
    "ParquetSink",
    "SampleSink",
    "WebDatasetSink",
    "build_sink",
    "sink_names",
]

_DEFAULT_INSTRUCTION = "Transcribe this document to Markdown, preserving its structure."


class SampleSink(ABC):
    """Writes samples somewhere. Used as a context manager."""

    def __enter__(self) -> SampleSink:
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:  # pragma: no cover - most sinks need no setup
        return None

    @abstractmethod
    def write(self, sample: Sample) -> None: ...

    def close(self) -> None:  # pragma: no cover - most sinks need no teardown
        return None


class _ImageWritingSink(SampleSink):
    """Shared behaviour for sinks that put image files on disk next to a manifest."""

    def __init__(
        self,
        directory: str | Path,
        *,
        shard: int = 0,
        image_format: str = "png",
        image_quality: int = 92,
        images_subdir: str = "images",
    ):
        self.directory = Path(directory)
        self.shard = shard
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.images_dir = self.directory / images_subdir
        self.count = 0

    def open(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _save_image(self, sample: Sample) -> str:
        extension = "jpg" if self.image_format == "jpeg" else self.image_format
        relative = Path(self.images_dir.name) / f"{sample.id}.{extension}"
        path = self.directory / relative
        image = sample.image if sample.image.mode == "RGB" else sample.image.convert("RGB")
        if self.image_format in ("jpeg", "webp"):
            image.save(path, quality=self.image_quality)
        else:
            image.save(path)
        self.count += 1
        return relative.as_posix()


class JsonlSink(_ImageWritingSink):
    """Images on disk plus one JSON record per page."""

    def open(self) -> None:
        super().open()
        self._handle = (self.directory / f"annotations-{self.shard:05d}.jsonl").open("w", encoding="utf-8")

    def write(self, sample: Sample) -> None:
        record = sample.to_dict(image_path=self._save_image(sample))
        record["text"] = sample.text
        record["markdown"] = sample.page.to_markdown()
        record["html"] = sample.page.to_html()
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._handle.close()


class ParquetSink(_ImageWritingSink):
    """The JSONL records, columnar. Requires the `data` extra."""

    def open(self) -> None:
        super().open()
        self._rows: list[dict[str, Any]] = []

    def write(self, sample: Sample) -> None:
        self._rows.append(
            {
                "id": sample.id,
                "image_path": self._save_image(sample),
                "text": sample.text,
                "markdown": sample.page.to_markdown(),
                "html": sample.page.to_html(),
                "width": sample.page.width,
                "height": sample.page.height,
                "annotation": json.dumps(sample.page.to_dict(), ensure_ascii=False),
                "provenance": json.dumps(sample.provenance.to_dict(), ensure_ascii=False),
            }
        )

    def close(self) -> None:
        if not self._rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The parquet writer needs pyarrow. Install it with: pip install 'ocrsmith[data]'"
            ) from exc
        table = pa.Table.from_pylist(self._rows)
        pq.write_table(table, self.directory / f"shard-{self.shard:05d}.parquet")


class WebDatasetSink(SampleSink):
    """Tar shards of `(image, json)` pairs keyed by sample id.

    Sequential reads off a tar are what keep data loading ahead of a GPU once the dataset
    is too large for local disk, which is the regime any serious OCR training run is in.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        shard: int = 0,
        image_format: str = "jpeg",
        image_quality: int = 92,
        **_ignored,
    ):
        self.directory = Path(directory)
        self.shard = shard
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.count = 0

    def open(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # Held open for the sink lifetime by design: the sink *is* the context manager.
        self._tar = tarfile.open(self.directory / f"shard-{self.shard:05d}.tar", "w")  # noqa: SIM115

    def write(self, sample: Sample) -> None:
        import io

        extension = "jpg" if self.image_format == "jpeg" else self.image_format
        buffer = io.BytesIO()
        image = sample.image if sample.image.mode == "RGB" else sample.image.convert("RGB")
        if self.image_format in ("jpeg", "webp"):
            image.save(buffer, format=self.image_format.upper(), quality=self.image_quality)
        else:
            image.save(buffer, format=self.image_format.upper())
        self._add(f"{sample.id}.{extension}", buffer.getvalue())

        record = sample.to_dict()
        record["text"] = sample.text
        record["markdown"] = sample.page.to_markdown()
        record["html"] = sample.page.to_html()
        self._add(f"{sample.id}.json", json.dumps(record, ensure_ascii=False).encode("utf-8"))
        self.count += 1

    def _add(self, name: str, payload: bytes) -> None:
        import io

        info = tarfile.TarInfo(name)
        info.size = len(payload)
        self._tar.addfile(info, io.BytesIO(payload))

    def close(self) -> None:
        self._tar.close()


class CocoSink(_ImageWritingSink):
    """Detection layout, with every word as an instance and regions as a second category set."""

    #: Region types become categories alongside "word" and "line".
    def open(self) -> None:
        super().open()
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._categories = {
            name: index + 1 for index, name in enumerate(["word", "line", *(t.value for t in RegionType)])
        }

    def write(self, sample: Sample) -> None:
        image_id = len(self._images) + 1
        self._images.append(
            {
                "id": image_id,
                "file_name": self._save_image(sample),
                "width": sample.page.width,
                "height": sample.page.height,
            }
        )
        for region in sample.page.ordered_regions():
            self._add_instance(image_id, region.type.value, region.bbox, region.text)
            for line in region.lines:
                self._add_instance(image_id, "line", line.bbox, line.text, line.polygon)
                for word in line.words:
                    self._add_instance(image_id, "word", word.bbox, word.text, word.polygon)

    def _add_instance(self, image_id: int, category: str, box, text: str, polygon=None) -> None:
        self._annotations.append(
            {
                "id": len(self._annotations) + 1,
                "image_id": image_id,
                "category_id": self._categories[category],
                "bbox": [round(value, 2) for value in box.as_xywh()],
                "area": round(box.area, 2),
                "iscrowd": 0,
                "segmentation": [list(polygon.as_flat())] if polygon else [],
                "text": text,
            }
        )

    def close(self) -> None:
        payload = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": [{"id": index, "name": name} for name, index in self._categories.items()],
        }
        path = self.directory / f"instances-{self.shard:05d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class PaddleOcrSink(_ImageWritingSink):
    """PaddleOCR's detection label file plus per-line recognition crops."""

    def open(self) -> None:
        super().open()
        self._det = (self.directory / f"det_label-{self.shard:05d}.txt").open("w", encoding="utf-8")
        self._rec = (self.directory / f"rec_label-{self.shard:05d}.txt").open("w", encoding="utf-8")
        self._crops_dir = self.directory / "line_crops"
        self._crops_dir.mkdir(parents=True, exist_ok=True)

    def write(self, sample: Sample) -> None:
        image_path = self._save_image(sample)
        boxes = [
            {
                "transcription": line.text,
                "points": [[round(x, 1), round(y, 1)] for x, y in self._quad(line)],
            }
            for line in sample.page.iter_lines()
            if line.text.strip()
        ]
        self._det.write(f"{image_path}\t{json.dumps(boxes, ensure_ascii=False)}\n")

        for index, line in enumerate(sample.page.iter_lines()):
            if not line.text.strip():
                continue
            crop_name = f"{sample.id}_{index:04d}.png"
            box = line.bbox.clip(0, 0, sample.page.width, sample.page.height)
            if box.width < 2 or box.height < 2:
                continue
            sample.image.crop(box.as_int()).save(self._crops_dir / crop_name)
            self._rec.write(f"line_crops/{crop_name}\t{line.text}\n")

    @staticmethod
    def _quad(line):
        from ..domain.geometry import Polygon

        return (line.polygon or Polygon.from_bbox(line.bbox)).points

    def close(self) -> None:
        self._det.close()
        self._rec.close()


class ChatSink(_ImageWritingSink):
    """Instruction-tuning records for a vision-language model."""

    def __init__(self, *args, instruction: str = _DEFAULT_INSTRUCTION, **kwargs):
        super().__init__(*args, **kwargs)
        self.instruction = instruction

    def open(self) -> None:
        super().open()
        self._handle = (self.directory / f"chat-{self.shard:05d}.jsonl").open("w", encoding="utf-8")

    def write(self, sample: Sample) -> None:
        record = {
            "id": sample.id,
            "image": self._save_image(sample),
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.instruction}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample.page.to_markdown()}]},
            ],
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._handle.close()


_SINKS: dict[str, type[SampleSink]] = {
    "jsonl": JsonlSink,
    "parquet": ParquetSink,
    "webdataset": WebDatasetSink,
    "coco": CocoSink,
    "paddleocr": PaddleOcrSink,
    "chat": ChatSink,
}


def sink_names() -> tuple[str, ...]:
    return tuple(sorted(_SINKS))


def build_sink(name: str, directory: str | Path, **kwargs) -> SampleSink:
    try:
        return _SINKS[name](directory, **kwargs)
    except KeyError:
        raise ValueError(f"Unknown output format {name!r}. Available: {', '.join(sink_names())}") from None
