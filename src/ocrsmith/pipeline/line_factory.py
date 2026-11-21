"""Turning a seed into an annotated recognition strip.

The line counterpart to `SampleFactory`. Same contract — `create(index)` yields `Sample`s
and everything random comes from one seed derived from `(config.seed, index)` — so the
runner, the writers, the quality gate and the CLI do not know which of the two produced
what they are holding.

What a strip contains, and why it is not just a sentence
--------------------------------------------------------
A recogniser is not fed sentences. It is fed whatever a detector cut out, and that
distribution is much stranger than a corpus is: single words from a form field, a number
with no sentence around it, three unrelated words from a table row, a word sliced in half
by a box edge. A generator that only ever emits well-formed sentences teaches the model to
use a language prior it will not have, and the failure shows up as confident misreadings
of short strings — exactly the strings that carry the information on an invoice or an ID.

So the content of a strip is sampled from a distribution of *crop shapes*, not of
sentences, and `_CONTENT` below is that distribution.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from functools import lru_cache

from PIL import Image

from ..config.schema import GenerationConfig
from ..core.backgrounds import BackgroundSampler, adapt_to_background
from ..core.degradations import build_preset
from ..core.documents.text_source import CorpusTextProvider
from ..core.fonts import FontPool, load_font
from ..core.lines import LineRenderer, LineStrip
from ..core.rendering.style import TextStyle
from ..domain import Provenance, Sample
from ..text.diacritics import DiacriticsMode, DiacriticsPolicy, apply_diacritics
from ..text.normalization import NumeralSystem, to_numeral_system
from ..text.script import Direction
from .factory import build_text_provider

__all__ = ["LineSampleFactory"]

_DIRECTIONS = {"rtl": Direction.RTL, "ltr": Direction.LTR}

#: Named inks, as (r, g, b). Not a palette so much as a list of the things documents are
#: actually written with: toner, worn toner, pencil and a biro.
_INKS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "near_black": (28, 28, 32),
    "grey": (84, 86, 92),
    "blue": (26, 42, 112),
}

#: Shapes a crop can take, and their weights. See the module docstring: this is the
#: distribution that matters, and it is not the distribution of sentences in a corpus.
_CONTENT: dict[str, float] = {
    "sentence": 4.0,
    "window": 3.0,
    "phrase": 2.0,
    "contextless": 1.0,
    "numeric": 1.0,
    "fragment": 0.6,
}

#: One background texture is generated per this many strips and then cut up, because
#: generating paper grain is a fixed cost per *image* and a strip is a thousandth of a
#: page. Each strip still takes its patch from a random offset, so they do not repeat.
_TEXTURE_REUSE = 48
_TEXTURE_SIZE = (900, 600)


@lru_cache(maxsize=512)
def _height_per_point(font_path: str) -> float:
    """Line-box height per pixel of nominal font size, for this face.

    Nominal size is not a height: an Arabic face with tall vertical metrics can set an
    ascender-to-descender box twice its em size, and a Latin one around 1.2x. A strip is
    specified by the height its text must come out at, so the size has to be solved for
    rather than sampled — otherwise `text_height_range` would mean a different thing for
    every font in the pool, which is the sort of silent inconsistency that makes a
    recognition corpus quietly biased towards whichever faces happen to run small.
    """
    probe = load_font(font_path, 100)
    ascent, descent = probe.getmetrics()
    return max(0.5, (ascent + descent) / 100.0)


def _weighted_choice(weights: dict, rng: random.Random):
    usable = {key: value for key, value in weights.items() if value > 0}
    if not usable:
        raise ValueError("no option has a positive weight")
    keys = list(usable)
    return rng.choices(keys, weights=[usable[key] for key in keys], k=1)[0]


class LineSampleFactory:
    """Builds annotated recognition strips from a `GenerationConfig`.

    Constructed once per process; `create` is cheap and stateless apart from its own rng
    and a background texture it reuses for a while.
    """

    def __init__(
        self,
        config: GenerationConfig,
        *,
        text_provider: CorpusTextProvider | None = None,
    ):
        self.config = config
        self.fonts = FontPool(
            config.fonts.paths,
            include=config.fonts.include,
            exclude=config.fonts.exclude,
            require_full_coverage=config.fonts.require_full_coverage,
        )
        self.text = text_provider or build_text_provider(config)
        self.backgrounds = BackgroundSampler(
            config.page.background.kinds,
            image_paths=config.page.background.image_paths,
            tint_range=config.page.background.tint_range,
        )
        self.renderer = LineRenderer()
        self._texture: Image.Image | None = None
        self._texture_kind = "solid"
        self._texture_ink: tuple[int, int, int] = (0, 0, 0)
        self._texture_age = 0

    # -- generation --------------------------------------------------------

    def create(self, index: int) -> Iterator[Sample]:
        """Yield the strip generated for `index`. One sample, always."""
        seed = self.config.sample_seed(index)
        rng = random.Random(seed)
        spec = self.config.line

        kind, text = self._content(rng)
        text, diacritics_kept = self._diacritics(text, rng)
        if not text.strip():
            return

        direction = _DIRECTIONS.get(self.config.text.direction) or None
        font_path = self.fonts.choose(text, rng)
        text_height = rng.randint(*sorted(spec.text_height_range))
        size = max(6, round(text_height / _height_per_point(str(font_path))))
        font = load_font(font_path, size)

        background, background_kind, paper_ink = self._background(rng)
        style = self._style(rng, paper_ink)

        strip = LineStrip(
            text_height=text_height,
            # Padding scales with the text, so a 64 px line is not framed like a 24 px
            # one. Sampled per edge: a real crop is never symmetric.
            pad_left=self._pad(rng, spec.pad_x_range, text_height),
            pad_right=self._pad(rng, spec.pad_x_range, text_height),
            pad_top=self._pad(rng, spec.pad_y_range, text_height),
            pad_bottom=self._pad(rng, spec.pad_y_range, text_height),
        )

        neighbours = None
        if rng.random() < spec.neighbour_probability:
            neighbours = (self.text.phrase(rng, 6), self.text.phrase(rng, 6))

        rendered = self.renderer.render(
            text,
            font,
            background,
            strip,
            style=style,
            direction=direction,
            neighbours=neighbours,
            rng=rng,
        )
        if not rendered.line.text:
            return

        preset_name = _weighted_choice(self.config.degradations.presets, rng)
        pipeline = build_preset(preset_name, scale="line")
        image, page, records = pipeline.apply(rendered.image, rendered.as_page(), rng)

        yield Sample(
            id=f"{index:08d}",
            image=image,
            page=page,
            provenance=Provenance(
                seed=seed,
                font_path=str(font_path),
                font_size=size,
                renderer="line",
                background=background_kind,
                shaper=type(self.renderer.text_renderer.shaper).__name__,
                source=self.config.text.source.type,
                template=kind,
                degradations=tuple(record.to_dict() for record in records),
                extra={
                    "index": index,
                    "diacritics_kept": round(diacritics_kept, 3),
                    "preset": preset_name,
                    "content": kind,
                    "text_height": text_height,
                    "direction": page.direction.value,
                },
            ),
        )

    # -- sampling helpers --------------------------------------------------

    def _content(self, rng: random.Random) -> tuple[str, str]:
        """One crop's worth of text, and the name of the shape it was drawn as."""
        low, high = sorted(self.config.line.words_range)
        kind = _weighted_choice(_CONTENT, rng)
        if kind == "sentence":
            words = self.text.sentence(rng).split()
            return kind, " ".join(words[: max(low, min(high, len(words)))])
        if kind == "window":
            words = self.text.sentence(rng).split()
            length = max(1, min(rng.randint(low, high), len(words)))
            start = rng.randint(0, max(0, len(words) - length))
            return kind, " ".join(words[start : start + length])
        if kind == "phrase":
            return kind, self.text.phrase(rng, max(1, min(3, high)))
        if kind == "contextless":
            return kind, self.text.contextless(rng, max(1, min(4, high)))
        if kind == "numeric":
            # Run through the configured numeral system like any other text. The corpus
            # gets this at load time, but these digits are invented here and so would
            # otherwise be western on every strip of a corpus configured for
            # Arabic-Indic — on exactly the samples whose whole content is digits.
            return kind, to_numeral_system(
                self._numeric(rng), NumeralSystem(self.config.text.normalization.numerals)
            )
        return kind, self.text.fragment(rng)

    @staticmethod
    def _numeric(rng: random.Random) -> str:
        """A number, date, reference or amount, with no sentence around it.

        These are both the hardest strings for a recogniser — no language prior helps —
        and the ones that carry the value on an invoice, a receipt or an ID. A corpus of
        prose contains almost none of them.
        """
        shape = rng.choice(["amount", "date", "reference", "integer", "percent", "phone"])
        if shape == "amount":
            return f"{rng.randrange(1, 999_999):,}.{rng.randrange(0, 100):02d}"
        if shape == "date":
            separator = rng.choice(["/", "-", "."])
            return separator.join(
                [f"{rng.randint(1, 28):02d}", f"{rng.randint(1, 12):02d}", str(rng.randint(1960, 2030))]
            )
        if shape == "reference":
            letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(rng.randint(2, 4)))
            return f"{letters}{rng.choice(['-', '/', ''])}{rng.randrange(1000, 9_999_999)}"
        if shape == "percent":
            return f"{rng.randint(0, 100)}{rng.choice(['%', ' %'])}"
        if shape == "phone":
            return f"0{rng.randint(5, 7)}{rng.randrange(10**7, 10**8)}"
        return str(rng.randrange(1, 10 ** rng.randint(1, 9)))

    def _diacritics(self, text: str, rng: random.Random) -> tuple[str, float]:
        policy = self.config.text.diacritics
        return apply_diacritics(
            text,
            DiacriticsPolicy(
                mode=DiacriticsMode(policy.mode),
                keep_range=tuple(policy.keep_range),
                mixed_weights=tuple(policy.mixed_weights),
            ),
            rng,
        )

    def _background(self, rng: random.Random):
        """A texture to cut this strip's patch from, generated once per `_TEXTURE_REUSE`.

        Paper grain costs the same whether the image is a page or a strip, so generating
        one per strip would make the background the most expensive part of a line — more
        than the text, the degradation and the encoding together.
        """
        if self._texture is None or self._texture_age >= _TEXTURE_REUSE:
            factory, kind = self.backgrounds.sample(rng)
            factory, ink = adapt_to_background(factory, rng, _TEXTURE_SIZE)
            self._texture = factory(*_TEXTURE_SIZE)
            self._texture_kind = kind
            self._texture_ink = ink
            self._texture_age = 0
        self._texture_age += 1
        return self._texture, self._texture_kind, self._texture_ink

    def _style(self, rng: random.Random, paper_ink: tuple[int, int, int]) -> TextStyle:
        """How this strip is set.

        `paper_ink` is what `adapt_to_background` decided is readable on this stock — on a
        photograph of dark slate that is a light colour — and it wins over the configured
        ink palette, because an unreadable strip with a confident label is the worst thing
        this project can emit.
        """
        if sum(paper_ink) / 3 > 128:
            color = paper_ink
        else:
            color = _INKS[_weighted_choice(self.config.line.inks, rng)]
        return TextStyle(
            color=color,
            synthetic_bold=rng.choice([0.0, 0.0, 0.0, 0.0, 0.5, 1.0]),
            synthetic_italic=rng.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.12]),
            underline=rng.random() < 0.03,
            word_spacing=rng.uniform(0.85, 1.35),
            # Sub-pixel wobble. Typesetting is exact and capture is not, and a model
            # trained on exact baselines leans on them.
            baseline_jitter=rng.uniform(0.0, 0.6),
            word_spacing_jitter=rng.uniform(0.0, 1.5),
        )

    @staticmethod
    def _pad(rng: random.Random, pad_range: tuple[int, int], text_height: int) -> int:
        low, high = sorted(pad_range)
        scaled = text_height / 40.0
        # Never below the renderer's own bleed, or a descender gets clipped by the edge
        # of the strip and the annotation would claim ink that was cut off.
        return max(5, int(round(rng.randint(low, high) * scaled)))
