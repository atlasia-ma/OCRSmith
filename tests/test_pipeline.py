"""Contract for configuration, the sample factory and the sharded runner.

The properties that make a large run survivable:

* a sample is reproducible from `(seed, index)` alone, without replaying the run;
* generation is a generator — nothing is materialised;
* shards are independent, and a completed shard is never redone.
"""

import json
from pathlib import Path

import pytest

from ocrsmith.config import GenerationConfig, apply_overrides, load_config
from ocrsmith.pipeline import (
    SampleFactory,
    iter_samples,
    plan_shards,
    run_generation,
    run_shard,
    runner,
)
from ocrsmith.pipeline.runner import ShardPlan

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

pytestmark = pytest.mark.skipif(not FONT_DIR.exists(), reason="bundled fonts unavailable")

SENTENCES = [
    "هذا نص تجريبي لتوليد المستندات الاصطناعية باللغة العربية.",
    "تهدف أطلسيا إلى بناء نماذج تعرف ضوئي عالية الجودة للدارجة المغربية.",
    "يحتوي هذا التقرير على جداول وأرقام ومعلومات إضافية مفيدة.",
    "تم إنشاء هذه الصفحة تلقائيا لأغراض التدريب والتقييم.",
]


def make_config(tmp_path: Path, **overrides) -> GenerationConfig:
    data = {
        "seed": 99,
        "fonts": {"paths": [str(FONT_DIR)], "size_range": [16, 20], "include": ["NotoSansArabic-"]},
        "text": {"source": {"type": "inline", "sentences": SENTENCES}},
        "page": {
            "papers": {"a6": 1.0},
            "dpi_range": [90, 90],
            "columns": {1: 1.0},
            "max_pages": 2,
            "header_probability": 0.0,
            "footer_probability": 0.0,
        },
        "templates": {"weights": {"article": 1.0}},
        "degradations": {"presets": {"clean": 1.0}},
        "output": {"dir": str(tmp_path / "dataset"), "shard_size": 2},
        "run": {"num_samples": 4, "workers": 1},
    }
    for key, value in overrides.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) else value
    return GenerationConfig.model_validate(data)


class TestConfig:
    def test_the_bundled_default_is_valid(self):
        assert load_config().run.num_samples > 0

    def test_missing_config_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_overrides_parse_json_values(self):
        data = apply_overrides({}, ["run.workers=8", "fonts.size_range=[20,24]", "output.dir=out"])

        assert data["run"]["workers"] == 8
        assert data["fonts"]["size_range"] == [20, 24]
        assert data["output"]["dir"] == "out"

    def test_malformed_override_is_rejected(self):
        with pytest.raises(ValueError, match="key.path=value"):
            apply_overrides({}, ["run.workers"])

    def test_seeds_are_derived_not_drawn(self):
        config = GenerationConfig(seed=7)

        assert config.sample_seed(3) == GenerationConfig(seed=7).sample_seed(3)
        assert config.sample_seed(3) != config.sample_seed(4)

    def test_a_different_base_seed_changes_every_sample(self):
        assert GenerationConfig(seed=1).sample_seed(0) != GenerationConfig(seed=2).sample_seed(0)

    def test_size_range_is_ordered(self):
        assert GenerationConfig.model_validate({"fonts": {"size_range": [30, 12]}}).fonts.size_range == (
            12,
            30,
        )

    def test_an_empty_template_weighting_is_rejected(self):
        with pytest.raises(ValueError, match="at least one template"):
            GenerationConfig.model_validate({"templates": {"weights": {"article": 0}}})

    def test_a_file_source_needs_a_path(self):
        with pytest.raises(ValueError, match="needs a path"):
            GenerationConfig.model_validate({"text": {"source": {"type": "csv"}}})

    def test_config_round_trips_through_a_plain_dict(self, tmp_path):
        config = make_config(tmp_path)

        assert GenerationConfig.model_validate(config.model_dump()) == config


class TestSampleFactory:
    def test_produces_at_least_one_page_per_document(self, tmp_path):
        factory = SampleFactory(make_config(tmp_path))

        samples = list(factory.create(0))

        assert samples
        assert samples[0].page.regions

    def test_pages_are_numbered_within_a_document(self, tmp_path):
        factory = SampleFactory(make_config(tmp_path, page={"max_pages": 3}))

        samples = list(factory.create(1))

        pages = [sample.provenance.extra["page"] for sample in samples]
        assert pages == list(range(1, len(samples) + 1))

    def test_identical_indices_produce_identical_samples(self, tmp_path):
        config = make_config(tmp_path)

        first = list(SampleFactory(config).create(2))
        second = list(SampleFactory(config).create(2))

        assert [s.id for s in first] == [s.id for s in second]
        assert [s.page.to_dict() for s in first] == [s.page.to_dict() for s in second]

    def test_different_indices_produce_different_samples(self, tmp_path):
        factory = SampleFactory(make_config(tmp_path))

        first = next(iter(factory.create(0)))
        second = next(iter(factory.create(1)))

        assert first.page.to_dict() != second.page.to_dict()

    def test_provenance_explains_the_sample(self, tmp_path):
        sample = next(iter(SampleFactory(make_config(tmp_path)).create(0)))

        provenance = sample.provenance.to_dict()
        assert provenance["seed"] == make_config(tmp_path).sample_seed(0)
        assert provenance["template"] == "article"
        assert provenance["font_path"].endswith(".ttf")
        assert provenance["extra"]["preset"] == "clean"

    def test_samples_carry_markup_ground_truth(self, tmp_path):
        sample = next(iter(SampleFactory(make_config(tmp_path)).create(0)))

        assert sample.page.to_html().startswith("<")
        assert sample.page.to_markdown().strip()
        assert sample.text.strip()

    def test_word_boxes_survive_the_whole_pipeline(self, tmp_path):
        sample = next(iter(SampleFactory(make_config(tmp_path)).create(0)))

        words = list(sample.page.iter_words())
        assert words
        for word in words:
            assert sample.page.bbox.contains(word.bbox)

    def test_degradation_preset_is_honoured(self, tmp_path):
        config = make_config(tmp_path, degradations={"presets": {"photo": 1.0}})

        sample = next(iter(SampleFactory(config).create(0)))

        assert sample.provenance.extra["preset"] == "photo"
        assert sample.provenance.degradations


class TestIterSamples:
    def test_streams_without_materialising(self, tmp_path):
        stream = iter_samples(make_config(tmp_path))

        assert next(stream).id.endswith("_01")

    def test_covers_the_configured_range(self, tmp_path):
        config = make_config(tmp_path, run={"num_samples": 3, "workers": 1})

        indices = {sample.provenance.extra["index"] for sample in iter_samples(config)}

        assert indices == {0, 1, 2}

    def test_start_index_shifts_the_range(self, tmp_path):
        config = make_config(tmp_path, run={"num_samples": 2, "workers": 1, "start_index": 10})

        indices = {sample.provenance.extra["index"] for sample in iter_samples(config)}

        assert indices == {10, 11}

    def test_a_wall_of_failures_aborts_rather_than_spinning(self, tmp_path):
        config = make_config(tmp_path, run={"num_samples": 20, "workers": 1, "max_consecutive_failures": 3})

        class Exploding(SampleFactory):
            def create(self, index):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="consecutive failures"):
            list(iter_samples(config, factory=Exploding(config)))


class TestSharding:
    def test_shards_tile_the_requested_range(self, tmp_path):
        config = make_config(tmp_path, run={"num_samples": 7, "workers": 1}, output={"shard_size": 3})

        plans = plan_shards(config)

        assert [(p.start, p.stop) for p in plans] == [(0, 3), (3, 6), (6, 7)]

    def test_shards_respect_the_start_index(self, tmp_path):
        config = make_config(tmp_path, run={"num_samples": 4, "start_index": 10}, output={"shard_size": 2})

        assert [(p.start, p.stop) for p in plan_shards(config)] == [(10, 12), (12, 14)]

    def test_a_completed_shard_is_not_regenerated(self, tmp_path):
        config = make_config(tmp_path)
        plan = ShardPlan(index=0, start=0, stop=1)

        first = run_shard(config, plan)
        second = run_shard(config, plan)

        assert first["skipped"] is False
        assert second["skipped"] is True


class TestRunGeneration:
    def test_writes_images_and_annotations(self, tmp_path):
        config = make_config(tmp_path)

        result = run_generation(config)

        directory = Path(config.output.dir)
        assert result.pages > 0
        assert result.documents == 4
        assert list((directory / "images").glob("*.png"))
        assert list(directory.glob("annotations-*.jsonl"))

    def test_annotations_describe_the_images_that_exist(self, tmp_path):
        config = make_config(tmp_path)

        run_generation(config)

        directory = Path(config.output.dir)
        for line in (next(directory.glob("annotations-*.jsonl"))).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert (directory / record["image_path"]).exists()
            assert record["page"]["regions"]
            assert record["markdown"]

    def test_progress_is_reported_per_shard(self, tmp_path):
        config = make_config(tmp_path)
        seen = []

        run_generation(config, progress=seen.append)

        assert len(seen) == len(plan_shards(config))

    def test_rerunning_skips_completed_shards(self, tmp_path):
        config = make_config(tmp_path)

        run_generation(config)
        second = run_generation(config)

        assert second.pages == 0
        assert len(second.skipped_shards) == len(plan_shards(config))

    def test_the_same_seed_gives_the_same_dataset(self, tmp_path):
        first = run_generation(make_config(tmp_path / "a"))
        second = run_generation(make_config(tmp_path / "b"))

        def records(result):
            path = next(Path(result.output_dir).glob("annotations-*.jsonl"))
            return [json.loads(line)["page"] for line in path.read_text(encoding="utf-8").splitlines()]

        assert records(first) == records(second)


class TestShardFailures:
    """A shard that dies must say why, and a run that produces nothing must not look fine.

    Found by a benchmark, not by a test: a parallel run reported success with zero pages
    and no reason, because every worker exception was swallowed into a bare counter. The
    cause was in the caller, but the pipeline destroyed the only evidence that would have
    identified it in seconds rather than in an hour.
    """

    def test_a_run_that_produces_nothing_raises(self, tmp_path, monkeypatch):
        config = make_config(tmp_path, run={"num_samples": 4, "workers": 1}, output={"shard_size": 2})

        def explode(_config, _plan):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(runner, "_run_one", explode)

        with pytest.raises(RuntimeError, match="disk on fire"):
            run_generation(config)

    def test_a_partial_failure_keeps_the_good_shards_and_the_reason(self, tmp_path, monkeypatch):
        config = make_config(tmp_path, run={"num_samples": 4, "workers": 1}, output={"shard_size": 2})
        real = runner._run_one

        def flaky(config_, plan):
            if plan.index == 1:
                raise RuntimeError("disk on fire")
            return real(config_, plan)

        monkeypatch.setattr(runner, "_run_one", flaky)

        result = run_generation(config)

        assert result.pages > 0, "a healthy shard must still be written"
        assert result.failures == 1
        assert "disk on fire" in dict(result.shard_errors)[1]

    def test_a_clean_run_reports_no_errors(self, tmp_path):
        result = run_generation(make_config(tmp_path))

        assert result.failures == 0
        assert result.shard_errors == ()

    def test_the_errors_serialise(self, tmp_path, monkeypatch):
        config = make_config(tmp_path, run={"num_samples": 4, "workers": 1}, output={"shard_size": 2})
        real = runner._run_one
        monkeypatch.setattr(
            runner,
            "_run_one",
            lambda c, p: (_ for _ in ()).throw(RuntimeError("nope")) if p.index else real(c, p),
        )

        payload = json.loads(json.dumps(run_generation(config).to_dict()))

        assert payload["shard_errors"][0][0] == 1


class TestReadingDirection:
    """The page must read in the same direction as its text.

    With `text.direction: auto` the configured direction is None, and defaulting the
    *page spec* to left-to-right ordered the columns left-first while the text inside them
    ran right-to-left — so an Arabic newspaper was laid out to be read in the wrong order.
    Nothing raised; the pixels looked plausible and the reading order was simply wrong.
    """

    ARABIC = [
        "المغرب بلد يقع في شمال إفريقيا ويطل على المحيط الأطلسي والبحر الأبيض المتوسط.",
        "تهدف أطلسيا إلى بناء نماذج ذكاء اصطناعي مفتوحة المصدر للغة الدارجة المغربية.",
        "يحتوي هذا التقرير على جداول وأرقام ومعلومات إضافية مفيدة للقارئ المهتم جدا.",
    ]

    def _multi_column(self, tmp_path, direction="auto"):
        return GenerationConfig.model_validate(
            {
                "seed": 20260813,
                "fonts": {"paths": [str(FONT_DIR)], "include": ["NotoSansArabic-"], "size_range": [15, 16]},
                "text": {"source": {"type": "inline", "sentences": self.ARABIC}, "direction": direction},
                "page": {
                    "papers": {"a4": 1.0},
                    "dpi_range": [110, 110],
                    "columns": {2: 1.0},
                    "max_pages": 1,
                    "header_probability": 0.0,
                    "footer_probability": 0.0,
                },
                "templates": {"weights": {"newspaper": 1.0}},
                "degradations": {"presets": {"clean": 1.0}},
                "output": {"dir": str(tmp_path)},
            }
        )

    def _first_body_region(self, sample):
        from ocrsmith.domain import RegionType

        furniture = {RegionType.HEADER, RegionType.FOOTER, RegionType.PAGE_NUMBER}
        return next(r for r in sample.page.ordered_regions() if r.type not in furniture)

    def test_auto_direction_reads_arabic_columns_right_first(self, tmp_path):
        config = self._multi_column(tmp_path, direction="auto")

        sample = next(iter(SampleFactory(config).create(0)))

        assert sample.page.direction.value == "rtl"
        first = self._first_body_region(sample)
        assert first.bbox.center[0] > sample.page.width / 2, "an Arabic page must start on the right"

    def test_explicit_rtl_agrees_with_auto(self, tmp_path):
        auto = next(iter(SampleFactory(self._multi_column(tmp_path, "auto")).create(0)))
        explicit = next(iter(SampleFactory(self._multi_column(tmp_path, "rtl")).create(0)))

        assert self._first_body_region(auto).bbox.center[0] == pytest.approx(
            self._first_body_region(explicit).bbox.center[0]
        )

    def test_forcing_ltr_reads_left_first(self, tmp_path):
        config = self._multi_column(tmp_path, direction="ltr")

        sample = next(iter(SampleFactory(config).create(0)))

        first = self._first_body_region(sample)
        assert first.bbox.center[0] < sample.page.width / 2

    def test_both_columns_are_used(self, tmp_path):
        config = self._multi_column(tmp_path, direction="auto")

        sample = next(iter(SampleFactory(config).create(0)))

        centres = [r.bbox.center[0] for r in sample.page.regions]
        midpoint = sample.page.width / 2
        assert any(c > midpoint for c in centres) and any(c < midpoint for c in centres)
