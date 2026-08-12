"""Contract for the dataset writers.

Each format is a promise to a downstream consumer, so the tests read the files back and
check they say what the sample said — not merely that a file appeared.
"""

import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from ocrsmith.datasets.writers import build_sink, sink_names
from ocrsmith.domain import (
    BBox,
    Line,
    Page,
    Provenance,
    Region,
    RegionType,
    Sample,
    Table,
    TableCell,
    Word,
)


@pytest.fixture
def sample():
    words = (Word("مرحبا", BBox(10, 10, 60, 30)), Word("بالعالم", BBox(70, 10, 140, 30)))
    line = Line("مرحبا بالعالم", BBox(10, 10, 140, 30), words)
    table = Table(
        1, 2, (TableCell(0, 0, "a", BBox(10, 40, 60, 60)), TableCell(0, 1, "b", BBox(60, 40, 110, 60)))
    )
    page = Page(
        200,
        100,
        (
            Region(RegionType.TITLE, BBox(10, 10, 140, 30), (line,), None, 0),
            Region(RegionType.TABLE, BBox(10, 40, 110, 60), (), table, 1),
        ),
    )
    return Sample(
        id="00000001_01",
        image=Image.new("RGB", (200, 100), (250, 250, 245)),
        page=page,
        provenance=Provenance(seed=5, template="report"),
    )


class TestRegistry:
    def test_every_format_is_named(self):
        assert set(sink_names()) == {"jsonl", "parquet", "webdataset", "coco", "paddleocr", "chat"}

    def test_unknown_format_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown output format"):
            build_sink("hdf5", tmp_path)


class TestJsonlSink:
    def test_writes_an_image_and_a_record(self, sample, tmp_path):
        with build_sink("jsonl", tmp_path) as sink:
            sink.write(sample)

        record = json.loads(next(tmp_path.glob("annotations-*.jsonl")).read_text(encoding="utf-8"))
        assert (tmp_path / record["image_path"]).exists()
        assert record["id"] == sample.id
        assert record["text"] == sample.text

    def test_carries_markup_ground_truth(self, sample, tmp_path):
        with build_sink("jsonl", tmp_path) as sink:
            sink.write(sample)

        record = json.loads(next(tmp_path.glob("annotations-*.jsonl")).read_text(encoding="utf-8"))
        assert "<h1>" in record["html"]
        assert record["markdown"].startswith("# ")

    def test_word_boxes_reach_the_file(self, sample, tmp_path):
        with build_sink("jsonl", tmp_path) as sink:
            sink.write(sample)

        record = json.loads(next(tmp_path.glob("annotations-*.jsonl")).read_text(encoding="utf-8"))
        words = record["page"]["regions"][0]["lines"][0]["words"]
        assert [word["text"] for word in words] == ["مرحبا", "بالعالم"]

    def test_shards_get_distinct_files(self, sample, tmp_path):
        for shard in (0, 1):
            with build_sink("jsonl", tmp_path, shard=shard) as sink:
                sink.write(sample)

        assert len(list(tmp_path.glob("annotations-*.jsonl"))) == 2

    def test_jpeg_output_is_honoured(self, sample, tmp_path):
        with build_sink("jsonl", tmp_path, image_format="jpeg") as sink:
            sink.write(sample)

        assert list((tmp_path / "images").glob("*.jpg"))


class TestWebDatasetSink:
    def test_produces_a_tar_of_image_json_pairs(self, sample, tmp_path):
        with build_sink("webdataset", tmp_path) as sink:
            sink.write(sample)

        archive = next(tmp_path.glob("shard-*.tar"))
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            assert f"{sample.id}.jpg" in names
            assert f"{sample.id}.json" in names
            payload = json.loads(tar.extractfile(f"{sample.id}.json").read().decode("utf-8"))
        assert payload["markdown"]

    def test_the_archive_is_readable_after_a_clean_close(self, sample, tmp_path):
        with build_sink("webdataset", tmp_path) as sink:
            for _ in range(3):
                sink.write(sample)

        with tarfile.open(next(tmp_path.glob("shard-*.tar"))) as tar:
            assert len(tar.getnames()) == 6  # three pairs, deduplicated names notwithstanding


class TestCocoSink:
    def test_words_lines_and_regions_all_become_instances(self, sample, tmp_path):
        with build_sink("coco", tmp_path) as sink:
            sink.write(sample)

        payload = json.loads(next(tmp_path.glob("instances-*.json")).read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in payload["categories"]}
        names = {categories[a["category_id"]] for a in payload["annotations"]}
        assert {"word", "line", "title"} <= names

    def test_boxes_use_coco_xywh(self, sample, tmp_path):
        with build_sink("coco", tmp_path) as sink:
            sink.write(sample)

        payload = json.loads(next(tmp_path.glob("instances-*.json")).read_text(encoding="utf-8"))
        word = next(a for a in payload["annotations"] if a["text"] == "مرحبا")
        assert word["bbox"] == [10, 10, 50, 20]

    def test_images_are_registered(self, sample, tmp_path):
        with build_sink("coco", tmp_path) as sink:
            sink.write(sample)

        payload = json.loads(next(tmp_path.glob("instances-*.json")).read_text(encoding="utf-8"))
        assert payload["images"][0]["width"] == 200
        assert (tmp_path / payload["images"][0]["file_name"]).exists()


class TestPaddleOcrSink:
    def test_detection_labels_list_line_quads(self, sample, tmp_path):
        with build_sink("paddleocr", tmp_path) as sink:
            sink.write(sample)

        line = next(tmp_path.glob("det_label-*.txt")).read_text(encoding="utf-8").strip()
        image_path, payload = line.split("\t")
        boxes = json.loads(payload)
        assert (tmp_path / image_path).exists()
        assert boxes[0]["transcription"] == "مرحبا بالعالم"
        assert len(boxes[0]["points"]) == 4

    def test_recognition_crops_are_written_with_their_text(self, sample, tmp_path):
        with build_sink("paddleocr", tmp_path) as sink:
            sink.write(sample)

        entry = next(tmp_path.glob("rec_label-*.txt")).read_text(encoding="utf-8").strip()
        crop_path, text = entry.split("\t")
        assert (tmp_path / crop_path).exists()
        assert text == "مرحبا بالعالم"


class TestChatSink:
    def test_produces_an_instruction_pair(self, sample, tmp_path):
        with build_sink("chat", tmp_path) as sink:
            sink.write(sample)

        record = json.loads(next(tmp_path.glob("chat-*.jsonl")).read_text(encoding="utf-8"))
        assert record["messages"][0]["role"] == "user"
        assert record["messages"][1]["content"][0]["text"] == sample.page.to_markdown()

    def test_the_instruction_is_configurable(self, sample, tmp_path):
        with build_sink("chat", tmp_path, instruction="Read this page.") as sink:
            sink.write(sample)

        record = json.loads(next(tmp_path.glob("chat-*.jsonl")).read_text(encoding="utf-8"))
        assert record["messages"][0]["content"][1]["text"] == "Read this page."


class TestParquetSink:
    def test_writes_a_readable_table(self, sample, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        with build_sink("parquet", tmp_path) as sink:
            sink.write(sample)

        table = pq.read_table(next(tmp_path.glob("shard-*.parquet")))
        assert table.num_rows == 1
        assert table.column("id")[0].as_py() == sample.id
        assert json.loads(table.column("annotation")[0].as_py())["regions"]

    def test_an_empty_shard_writes_nothing(self, tmp_path):
        with build_sink("parquet", tmp_path):
            pass

        assert list(Path(tmp_path).glob("*.parquet")) == []
