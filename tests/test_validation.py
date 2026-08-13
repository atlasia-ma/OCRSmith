"""Contract for corpus validation.

Two claims the rest of the project only asserts, made measurable here:

* that generated pages resemble real ones — measured feature by feature, with each gap
  named against the generator knob behind it;
* that a feature helps — by building corpora that differ in exactly one knob, so a
  downstream difference can only be attributed to that knob.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from ocrsmith.config import GenerationConfig
from ocrsmith.validation import (
    FEATURE_NAMES,
    PRESET_ABLATIONS,
    build_ablation,
    compare_features,
    extract_features,
    iter_image_features,
)


def page(
    *,
    ink=(15, 15, 20),
    paper=(248, 246, 240),
    stroke=3,
    spacing=3.0,
    blur=0.0,
    size=(400, 520),
    seed=0,
) -> Image.Image:
    """A crude text-like page, parameterised on the properties the features measure.

    Drawn as vertical stems rather than solid bars, because that is what a scanline
    through real text actually crosses — and it is the horizontal run length of those
    stems that `stroke_width` measures.
    """
    image = Image.new("RGB", size, paper)
    draw = ImageDraw.Draw(image)
    rng = random.Random(seed)
    line_height = stroke * 4
    y = 30
    while y < size[1] - 30:
        x = 30
        limit = rng.randint(int(size[0] * 0.5), int(size[0] * 0.9))
        while x < limit:
            draw.rectangle((x, y, x + stroke - 1, y + line_height), fill=ink)
            x += max(stroke + 1, int(stroke * spacing))
        y += line_height + stroke * 2
    return image.filter(ImageFilter.GaussianBlur(blur)) if blur else image


class TestFeatures:
    def test_every_named_feature_is_produced(self):
        features = extract_features(page())

        vector = features.as_vector()
        assert len(vector) == len(FEATURE_NAMES)
        assert not np.isnan(vector).any()

    def test_darker_ink_is_measured_as_darker(self):
        light = extract_features(page(ink=(140, 140, 140)))
        dark = extract_features(page(ink=(10, 10, 10)))

        assert dark.ink_darkness < light.ink_darkness

    def test_thicker_strokes_measure_wider(self):
        thin = extract_features(page(stroke=2))
        thick = extract_features(page(stroke=8))

        assert thick.stroke_width > thin.stroke_width

    def test_blur_removes_high_frequency_energy(self):
        crisp = extract_features(page())
        blurred = extract_features(page(blur=2.5))

        assert blurred.high_frequency_energy < crisp.high_frequency_energy
        assert blurred.edge_density < crisp.edge_density

    def test_a_denser_page_has_a_higher_ink_fraction(self):
        # Density is spacing, not stroke width: scaling stroke and gap together leaves the
        # ink fraction unchanged, which is exactly what a scale-invariant measure should do.
        sparse = extract_features(page(spacing=6.0))
        dense = extract_features(page(spacing=2.0))

        assert dense.ink_fraction > sparse.ink_fraction

    def test_contrast_separates_ink_from_its_own_paper(self):
        strong = extract_features(page(ink=(10, 10, 10), paper=(250, 250, 250)))
        weak = extract_features(page(ink=(120, 120, 120), paper=(180, 180, 180)))

        assert strong.contrast > weak.contrast

    def test_resolution_scaling_is_damped(self):
        small = extract_features(page(size=(700, 910), stroke=3))
        large = extract_features(page(size=(1400, 1820), stroke=6))

        # The long edge is capped before measurement, so doubling the capture resolution
        # must not double the measured stroke width. It damps the effect rather than
        # removing it, and the test says so instead of pretending otherwise.
        ratio = large.stroke_width / max(1e-6, small.stroke_width)
        assert 1.0 <= ratio < 1.8, f"expected damping below the 2.0x raw scale, got {ratio:.2f}"

    def test_serialisation_is_rounded_and_complete(self):
        data = extract_features(page()).to_dict()

        assert set(data) == set(FEATURE_NAMES)

    def test_unreadable_files_are_skipped(self, tmp_path):
        (tmp_path / "broken.png").write_bytes(b"not an image")
        page().save(tmp_path / "good.png")

        assert len(list(iter_image_features(tmp_path))) == 1

    def test_the_limit_is_honoured(self, tmp_path):
        for index in range(5):
            page(seed=index).save(tmp_path / f"{index}.png")

        assert len(list(iter_image_features(tmp_path, limit=2))) == 2


class TestComparison:
    def test_a_corpus_matches_itself(self):
        features = [extract_features(page(seed=seed)) for seed in range(6)]

        report = compare_features(features, features)

        assert report.mean_overlap == pytest.approx(1.0)
        assert report.mismatched == []

    def test_a_known_difference_is_detected_and_named(self):
        crisp = [extract_features(page(seed=s)) for s in range(6)]
        blurred = [extract_features(page(seed=s, blur=3.0)) for s in range(6)]

        report = compare_features(crisp, blurred)

        mismatched = {item.feature for item in report.mismatched}
        assert "high_frequency_energy" in mismatched
        assert report.mean_overlap < 0.9

    def test_the_direction_of_a_gap_is_reported(self):
        thin = [extract_features(page(seed=s, stroke=2)) for s in range(6)]
        thick = [extract_features(page(seed=s, stroke=8)) for s in range(6)]

        report = compare_features(thin, thick)
        stroke = next(c for c in report.comparisons if c.feature == "stroke_width")

        assert stroke.cohens_d < 0
        assert stroke.direction == "synthetic lower"
        assert stroke.relative_difference < 0

    def test_overlap_catches_a_spread_mismatch_a_mean_would_miss(self):
        # Same centre, different spread: a mean-only measure calls this a match.
        narrow = [extract_features(page(seed=s, stroke=4)) for s in range(8)]
        wide = [extract_features(page(seed=s, stroke=2 + (s % 4) * 2)) for s in range(8)]

        report = compare_features(narrow, wide)
        stroke = next(c for c in report.comparisons if c.feature == "stroke_width")

        assert stroke.overlap < 1.0

    def test_the_report_names_a_knob_to_change(self):
        crisp = [extract_features(page(seed=s)) for s in range(6)]
        blurred = [extract_features(page(seed=s, blur=3.0)) for s in range(6)]

        markdown = compare_features(crisp, blurred).to_markdown()

        assert "What to change" in markdown
        assert "Blur" in markdown, "a gap must point at a generator knob, not an abstraction"

    def test_an_empty_corpus_is_an_error(self):
        with pytest.raises(ValueError, match="at least one readable image"):
            compare_features([], [extract_features(page())])

    def test_the_report_serialises(self):
        features = [extract_features(page(seed=s)) for s in range(4)]

        data = compare_features(features, features).to_dict()

        assert json.loads(json.dumps(data))["synthetic_images"] == 4


class TestAblation:
    @pytest.fixture
    def base(self, tmp_path):
        font_dir = Path(__file__).resolve().parents[1] / "assets" / "fonts"
        if not font_dir.exists():
            pytest.skip("bundled fonts unavailable")
        return GenerationConfig.model_validate(
            {
                "seed": 4242,
                "fonts": {"paths": [str(font_dir)], "include": ["NotoSansArabic-"]},
                "output": {"dir": str(tmp_path / "base")},
                "run": {"num_samples": 3},
            }
        )

    def test_every_preset_builds(self, base, tmp_path):
        for name in PRESET_ABLATIONS:
            plan = build_ablation(name, base, tmp_path)
            assert len(plan.variants) >= 2

    def test_an_unknown_ablation_is_rejected(self, base, tmp_path):
        with pytest.raises(ValueError, match="Unknown ablation"):
            build_ablation("telepathy", base, tmp_path)

    def test_arms_share_the_base_seed(self, base, tmp_path):
        # If the arms drew different text, a downstream difference could come from either
        # the feature or the sample, and the experiment would answer nothing.
        plan = build_ablation("degradations", base, tmp_path)

        assert {variant.config.seed for variant in plan.variants} == {base.seed}

    def test_arms_write_to_separate_directories(self, base, tmp_path):
        plan = build_ablation("layout", base, tmp_path)

        directories = [variant.config.output.dir for variant in plan.variants]
        assert len(set(directories)) == len(directories)

    def test_each_arm_differs_only_in_its_named_knob(self, base, tmp_path):
        plan = build_ablation("degradations", base, tmp_path)
        baseline, clean = plan.variants[0], plan.variants[1]

        assert clean.config.degradations.presets != baseline.config.degradations.presets
        assert clean.config.page == baseline.config.page
        assert clean.config.templates == baseline.config.templates
        assert clean.config.fonts == baseline.config.fonts

    def test_sample_count_can_be_overridden_for_every_arm(self, base, tmp_path):
        plan = build_ablation("fonts", base, tmp_path, num_samples=7)

        assert {variant.config.run.num_samples for variant in plan.variants} == {7}

    def test_the_manifest_records_the_plan(self, base, tmp_path):
        plan = build_ablation("diacritics", base, tmp_path)

        manifest = json.loads(plan.write_manifest(tmp_path).read_text(encoding="utf-8"))

        assert manifest["ablation"] == "diacritics"
        assert len(manifest["variants"]) == 3

    def test_the_markdown_warns_against_a_synthetic_test_set(self, base, tmp_path):
        markdown = build_ablation("layout", base, tmp_path).to_markdown()

        assert "real" in markdown.lower()
