"""Ablation: generating the corpora that answer "does this feature help?"

Every feature in this repository was added because the literature or a benchmark said it
should help. That is a reasonable prior and a poor substitute for evidence. An ablation
turns it into a measurement: generate the same corpus with a feature on and off, train the
same model on each, and compare on a held-out real benchmark.

This module builds the *corpora*, not the model. Training is deliberately out of scope — a
data generator that also owned a training loop would be two projects, and the loop you
want depends on the architecture you are testing. What is provided is the part that is
easy to get subtly wrong: variants that differ in exactly one knob, share a seed so their
content is otherwise identical, and record what they changed.

Sharing the seed matters more than it looks. If the "with" and "without" corpora draw
different text, a difference in downstream accuracy could come from either the feature or
the sample, and the experiment answers nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config.loader import apply_overrides
from ..config.schema import GenerationConfig

__all__ = ["AblationPlan", "AblationVariant", "PRESET_ABLATIONS", "build_ablation"]


@dataclass(frozen=True, slots=True)
class AblationVariant:
    """One arm of an ablation: a config, and what makes it different."""

    name: str
    config: GenerationConfig
    overrides: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "overrides": list(self.overrides),
            "output_dir": self.config.output.dir,
        }


@dataclass
class AblationPlan:
    """A baseline plus one variant per feature under test."""

    name: str
    variants: list[AblationVariant] = field(default_factory=list)

    @property
    def baseline(self) -> AblationVariant:
        return self.variants[0]

    def to_dict(self) -> dict:
        return {
            "ablation": self.name,
            "baseline": self.baseline.name,
            "variants": [variant.to_dict() for variant in self.variants],
        }

    def write_manifest(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        manifest = path / f"ablation-{self.name}.json"
        manifest.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    def to_markdown(self) -> str:
        lines = [
            f"## Ablation: {self.name}",
            "",
            "Every arm shares the same seed, so the arms differ in exactly the knob named",
            "and in nothing else. Train the same model on each and compare on a held-out",
            "**real** benchmark — a synthetic test set would reward the generator's own biases.",
            "",
            "| arm | changes | corpus |",
            "| --- | --- | --- |",
        ]
        for variant in self.variants:
            changes = ", ".join(f"`{o}`" for o in variant.overrides) or "_(baseline)_"
            lines.append(f"| `{variant.name}` | {changes} | `{variant.config.output.dir}` |")
        return "\n".join(lines) + "\n"


#: The questions worth asking first, each phrased as "turn this off and see".
#: Every one corresponds to a claim made in the CHANGELOG that is currently unmeasured.
PRESET_ABLATIONS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "degradations": [
        ("all", "every capture condition", ()),
        ("clean_only", "no degradation at all", ('degradations.presets={"clean":1}',)),
        (
            "no_physical",
            "no page curl, wrinkles or illumination fields",
            ('degradations.presets={"clean":1,"scan":4,"fax":1}',),
        ),
    ],
    "fonts": [
        ("all", "the full font pool", ()),
        ("single_family", "one typeface only", ('fonts.include=["NotoSansArabic-"]',)),
    ],
    "layout": [
        ("all", "every genre", ()),
        (
            "prose_only",
            "articles only - no tables, forms, charts or equations",
            ('templates.weights={"article":1}',),
        ),
        ("single_column", "never more than one column", ('page.columns={"1":1}',)),
    ],
    "diacritics": [
        ("mixed", "vocalisation varies per document", ('text.diacritics.mode="mixed"',)),
        ("stripped", "no diacritics anywhere", ('text.diacritics.mode="strip"',)),
        ("kept", "source vocalisation untouched", ('text.diacritics.mode="keep"',)),
    ],
}


def build_ablation(
    name: str,
    base_config: GenerationConfig,
    output_root: str | Path,
    *,
    arms: Sequence[tuple[str, str, Sequence[str]]] | None = None,
    num_samples: int | None = None,
) -> AblationPlan:
    """Build the corpora for an ablation.

    Each arm inherits the base config, applies its own overrides, and writes to its own
    directory — but keeps the base seed, so the arms differ only in the knob under test.
    """
    if arms is None:
        try:
            arms = PRESET_ABLATIONS[name]
        except KeyError:
            raise ValueError(
                f"Unknown ablation {name!r}. Available: {', '.join(sorted(PRESET_ABLATIONS))}"
            ) from None

    root = Path(output_root)
    payload = base_config.model_dump(mode="json")
    variants: list[AblationVariant] = []

    for arm_name, description, overrides in arms:
        arm_overrides = [*overrides, f'output.dir="{(root / name / arm_name).as_posix()}"']
        if num_samples:
            arm_overrides.append(f"run.num_samples={int(num_samples)}")
        config = GenerationConfig.model_validate(apply_overrides(payload, arm_overrides))
        variants.append(
            AblationVariant(
                name=arm_name,
                config=config,
                overrides=tuple(overrides),
                description=description,
            )
        )

    return AblationPlan(name=name, variants=variants)
