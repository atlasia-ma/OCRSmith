"""Build a line-recognition corpus from full pages.

Line crops cut from a real page are better training data than lines rendered in isolation:
they carry the page's actual background, degradation and neighbouring-line bleed, which is
what a recogniser sees when it is fed a detector's output.

    python examples/generate_line_crops.py --count 200 --output outputs/lines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrsmith import SampleFactory, load_config

#: Below this height a crop carries no recoverable stroke detail.
MIN_CROP_HEIGHT = 12
#: Grown slightly so ascenders and descenders are not clipped by a tight box.
CROP_PADDING = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--count", type=int, default=50, help="Documents to render.")
    parser.add_argument("--output", type=Path, default=Path("outputs/lines"))
    args = parser.parse_args()

    config = load_config(args.config)
    factory = SampleFactory(config)

    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "labels.jsonl"

    written = 0
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(args.count):
            for sample in factory.create(index):
                page = sample.page
                for position, line in enumerate(page.iter_lines()):
                    if not line.text.strip():
                        continue
                    box = line.bbox.pad(CROP_PADDING).clip(0, 0, page.width, page.height)
                    if box.height < MIN_CROP_HEIGHT or box.width < MIN_CROP_HEIGHT:
                        continue
                    name = f"{sample.id}_{position:04d}.png"
                    sample.image.crop(box.as_int()).save(images_dir / name)
                    handle.write(
                        json.dumps(
                            {
                                "image": f"images/{name}",
                                "text": line.text,
                                "direction": line.direction.value,
                                "source_page": sample.id,
                                "words": [word.text for word in line.words],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

    print(f"Wrote {written} line crops to {args.output}")


if __name__ == "__main__":
    main()
