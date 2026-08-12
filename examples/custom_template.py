"""Add a document genre of your own.

A template describes *what a document says*; page size, fonts and degradation are decided
elsewhere. That separation is what lets one template appear as a crisp A4 print and a
creased phone photo with identical markup ground truth.

    python examples/custom_template.py --output outputs/id_cards
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from ocrsmith import load_config
from ocrsmith.core.documents import DocumentBuilder, TextProvider, default_registry
from ocrsmith.pipeline import SampleFactory
from ocrsmith.text import Direction


@dataclass(frozen=True, slots=True)
class IdCardTemplate:
    """A compact identity document: a title, a portrait slot, and label/value rows."""

    name: str = "id_card"

    def build(self, source: TextProvider, rng: random.Random, **options):
        builder = DocumentBuilder(options.get("direction") or Direction.RTL, template=self.name)
        builder.title(source.phrase(rng, 3))
        builder.figure(width=rng.randint(120, 180), height=rng.randint(150, 210))
        builder.key_values(
            [
                (source.phrase(rng, 1), source.phrase(rng, 2)),
                (source.phrase(rng, 1), source.phrase(rng, 2)),
                (source.phrase(rng, 2), f"{rng.randint(10, 28)}/{rng.randint(1, 12)}/19{rng.randint(60, 99)}"),
                (source.phrase(rng, 1), f"{rng.randint(100000, 999999)}"),
            ]
        )
        builder.separator()
        builder.paragraph(source.phrase(rng, 6))
        return builder.build()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/id_cards"))
    args = parser.parse_args()

    # Register the new genre and make it the only one sampled.
    registry = default_registry().register(IdCardTemplate(), weight=1.0)
    config = load_config().model_copy(
        update={
            "templates": type(load_config().templates)(weights={"id_card": 1.0}),
        }
    )
    config.page.papers = {"id_card": 1.0}
    config.page.max_pages = 1
    config.text.source.type = "inline"

    factory = SampleFactory(config, registry=registry)
    args.output.mkdir(parents=True, exist_ok=True)

    for index in range(args.count):
        for sample in factory.create(index):
            sample.image.save(args.output / f"{sample.id}.png")
            (args.output / f"{sample.id}.md").write_text(
                sample.page.to_markdown(), encoding="utf-8"
            )
    print(f"Wrote {args.count} cards to {args.output}")


if __name__ == "__main__":
    main()
