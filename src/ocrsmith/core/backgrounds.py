"""Page backgrounds.

Backgrounds are produced by a factory taking `(width, height)`, so the layout engine never
needs to know whether a page is plain white, tinted stock, a gradient or a photograph of
real paper. Textures are generated at a quarter scale and upsampled: paper grain is
low-frequency, and generating it per-pixel at full page size is pure waste.
"""

from __future__ import annotations

import colorsys
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = [
    "BackgroundFactory",
    "BackgroundSampler",
    "adapt_to_background",
    "gradient_background",
    "image_background",
    "paper_background",
    "solid_background",
]

BackgroundFactory = Callable[[int, int], Image.Image]
Color = tuple[int, int, int]

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def solid_background(color: Color) -> BackgroundFactory:
    def factory(width: int, height: int) -> Image.Image:
        return Image.new("RGB", (width, height), color)

    return factory


def paper_background(color: Color, grain: float, rng: random.Random) -> BackgroundFactory:
    """Tinted stock with low-frequency fibre texture."""
    seed = rng.getrandbits(32)

    def factory(width: int, height: int) -> Image.Image:
        generator = np.random.default_rng(seed)
        small_h, small_w = max(1, height // 4), max(1, width // 4)
        noise = generator.standard_normal((small_h, small_w), dtype=np.float32) * np.float32(grain)
        fibres = Image.fromarray(np.clip(noise + 128.0, 0, 255).astype(np.uint8), "L").resize(
            (width, height), Image.Resampling.BILINEAR
        )
        # The tint is applied per channel with a lookup table rather than by building a
        # float page and adding to it. Both produce the same bytes; this one touches
        # `width * height` bytes three times instead of `width * height * 3` floats five
        # times, and a background is generated once per page whatever else happens to it.
        return Image.merge(
            "RGB",
            [
                fibres.point([min(255, max(0, value - 128 + channel)) for value in range(256)])
                for channel in color
            ],
        )

    return factory


def gradient_background(start: Color, end: Color, direction: str) -> BackgroundFactory:
    def factory(width: int, height: int) -> Image.Image:
        ys, xs = np.mgrid[0:height, 0:width]
        if direction == "horizontal":
            ratio = xs / max(1, width - 1)
        elif direction == "vertical":
            ratio = ys / max(1, height - 1)
        else:
            ratio = (xs + ys) / max(1, width + height - 2)
        ratio = ratio[..., None].astype(np.float32)
        array = (
            np.array(start, dtype=np.float32)
            + (np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32)) * ratio
        )
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")

    return factory


def image_background(path: str | Path, opacity: float = 1.0) -> BackgroundFactory:
    """Centre-crop a photograph to the page aspect ratio and scale it to fit.

    Drawn at full strength by default. A surface that is dark, or light, or textured is
    not a problem to be washed out - it is the surface, and `adapt_to_background` handles
    it by choosing ink that can be read against it. Only a background that no single ink
    colour can survive is toned back, and only as far as it has to be.
    """

    def factory(width: int, height: int) -> Image.Image:
        source = Image.open(path).convert("RGB")
        target_ratio = width / height
        source_ratio = source.width / source.height
        if source_ratio > target_ratio:
            crop_width = int(source.height * target_ratio)
            left = (source.width - crop_width) // 2
            source = source.crop((left, 0, left + crop_width, source.height))
        else:
            crop_height = int(source.width / target_ratio)
            top = (source.height - crop_height) // 2
            source = source.crop((0, top, source.width, top + crop_height))
        scaled = source.resize((width, height), Image.Resampling.LANCZOS)
        if opacity >= 1.0:
            return scaled
        paper = Image.new("RGB", (width, height), (252, 250, 246))
        return Image.blend(paper, scaled, max(0.0, min(1.0, opacity)))

    return factory


#: Ink is judged against the *worst* part of the surface it has to cross, not the average
#: one. A page whose mean luminance is comfortable can still swallow a word where a bright
#: streak or a dark seam happens to fall, and it is the word that gets lost, not the mean -
#: so the probe reads the 4th and 96th percentiles rather than something gentler. Further
#: out than that and it starts chasing single-pixel outliers no line of text will ever sit
#: on, which bleaches good surfaces to protect against a patch nothing lands in.
#:
#: Separation below this is not reliably readable once a capture condition has been through
#: it. Enforced on *luminance* specifically, and not relaxed for a strong hue difference,
#: because most OCR pipelines convert to greyscale before they see anything: two colours
#: that look vividly different to a reader can arrive at the model identical.
_MIN_SEPARATION = 70.0

#: How far *dark* ink sits from the surface's own hue, as a fraction of the colour wheel.
#: Directly opposite reads as deliberate - warm brown pigment on pale blue stock is a
#: choice a designer would make - and keeps the ink distinguishable to a colour model even
#: where luminance alone is thin.
#:
#: Light ink is not treated the same way, and the asymmetry is the point. Dark ink is
#: pigment sitting on the surface, and a complement gives it richness; light ink is closer
#: to the surface being left unprinted, and tinting it against the field reads as a stain
#: rather than as ink. Cream lettering on saturated blue looks like water damage. Real
#: work puts a clean near-white on a coloured field, carrying at most a trace of the
#: field's own hue, which is what the low saturation below produces.
_COMPLEMENT = 0.5
_LIGHT_INK_SATURATION = (0.03, 0.09)

#: Light ink is placed as bright as the band allows rather than as close to the floor as it
#: can get. There is no hue to keep visible here, so the only thing trading brightness away
#: would buy is a dingier white and less contrast.
_LIGHT_INK_FLOOR = 228.0

#: Ink luminance bands. Wide, and deliberately so: a hue is only visible if the ink has
#: room to carry it. At luminance 30 a saturated orange renders as (44, 25, 18), which is
#: black to any reader - the complement was being computed correctly and then thrown away
#: by placing it somewhere no colour survives. Where the surface leaves contrast to spare,
#: the ink is placed at the light or dark end of what still clears the floor rather than at
#: the extreme, so the colour actually shows.
_DARK_INK_LUMA = (22.0, 95.0)
_LIGHT_INK_LUMA = (196.0, 250.0)

#: How much of its own band the ink gives up to keep some variety between documents.
_INK_LATITUDE = 40.0

#: A surface whose hue wanders this far has no single complement to oppose - a row of
#: shipping containers is red and blue and yellow at once, and ink chosen against the mean
#: of those is chosen against a colour that appears nowhere on the page. Measured as
#: circular variance, and only trusted when the surface is colourful enough for hue to mean
#: anything: newsprint has a wandering hue too, but it is grey, and its hue is just noise.
_MAX_HUE_VARIANCE = 0.25
_MIN_COLOURFULNESS = 30.0

#: Outside this band an ink reads as flat black or flat white whatever hue it was given.
#: A bright saturated surface - a swimming pool at noon - leaves no room between its own
#: luminance and the separation floor, so ink against it is forced to an extreme and the
#: complement is computed and then lost. On a colourful surface that is a reason to lay
#: paper over it: toning opens the gap the ink needs to show its colour. On a grey one it
#: is not, because there was no colour to show.
_HUE_VISIBLE = (45.0, 240.0)

#: Ink for a page whose surface has been toned. Plain near-black, no complement: once
#: paper is over the photograph the page reads as a document on textured stock, and that is
#: printed in black. Chasing a hue here fights the wash that was just applied to make the
#: text legible in the first place.
_TONED_INKS = ((18, 18, 18), (26, 26, 30), (34, 34, 38))

#: How much of a photograph may be given up to make black ink work on it. Documents are
#: overwhelmingly dark-on-light, so black is tried first and a light wash is preferred to
#: reversing the page out: a blue canvas reads as a document printed on blue stock, which is
#: black ink on a lightened field, not white ink on a dark one. Light ink is kept for the
#: surface too dark to lighten without destroying it.
_MIN_KEEP_FOR_DARK = 0.55

#: The most of the photograph a toned surface may keep. Solving purely for legibility let a
#: busy surface off with a wash so slight it was invisible - black ink already cleared the
#: containers photograph on paper, so the solver asked for no tone at all and the page came
#: back as unreadable text over shipping crates. A layer that is applied must be seen.
_MAX_TONED_KEEP = 0.6


def _luminance(color: tuple[float, float, float]) -> float:
    red, green, blue = color
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _hue_profile(image: Image.Image) -> tuple[float, float, float, float]:
    """The surface's hue, saturation, how much the hue wanders, and how colourful it is.

    The hue is a saturation-weighted *circular* mean, because hue wraps: averaging 0.02 and
    0.98 arithmetically gives cyan when both are red. The same computation yields the
    resultant length, and one minus that is the circular variance - near zero when the
    surface is essentially one colour, near one when it is many.

    Colourfulness is carried alongside because hue variance alone cannot tell a row of
    painted containers from a sheet of newsprint: both wander, but on grey paper the hue is
    numerical noise with nothing to oppose.
    """
    array = np.asarray(image, dtype=np.float32)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    weight = (saturation * value).ravel()
    total = float(weight.sum())
    if total < 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    angle = hue.ravel() * 2 * np.pi
    sin_sum = float((np.sin(angle) * weight).sum())
    cos_sum = float((np.cos(angle) * weight).sum())
    mean_hue = (np.arctan2(sin_sum, cos_sum) / (2 * np.pi)) % 1.0
    resultant = float(np.hypot(sin_sum, cos_sum)) / total

    # Hasler-Susstrunk: how far the pixels spread along the red-green and yellow-blue axes.
    red_green = array[..., 0] - array[..., 1]
    yellow_blue = 0.5 * (array[..., 0] + array[..., 1]) - array[..., 2]
    colourfulness = float(
        np.hypot(red_green.std(), yellow_blue.std())
        + 0.3 * np.hypot(red_green.mean(), yellow_blue.mean())
    )
    return mean_hue, float(saturation.mean()), 1.0 - resultant, colourfulness


def _ink_at(hue: float, target_luma: float, saturation: float) -> tuple[int, int, int]:
    """An ink of the given hue placed at a target luminance."""
    base = colorsys.hsv_to_rgb(hue, saturation, 1.0)
    base255 = tuple(channel * 255.0 for channel in base)
    base_luma = _luminance(base255)

    if target_luma <= base_luma:
        # Shade towards black until it lands on the target.
        scale = target_luma / max(1e-6, base_luma)
        rgb = tuple(channel * scale for channel in base255)
    else:
        # Tint towards white instead; the hue survives as a wash over the letterforms.
        mix = (255.0 - target_luma) / max(1e-6, 255.0 - base_luma)
        rgb = tuple(255.0 * (1.0 - mix) + channel * mix for channel in base255)
    return tuple(int(round(max(0.0, min(255.0, channel)))) for channel in rgb)


def adapt_to_background(
    factory: BackgroundFactory,
    rng: random.Random,
    size: tuple[int, int] = (96, 128),
) -> tuple[BackgroundFactory, tuple[int, int, int]]:
    """Pick ink that can be read against this surface, toning the surface only if it must be.

    Two decisions, in order. *Which end* - a dark surface takes light ink and a light one
    takes dark, chosen by which has more room against the surface's worst patch, so a
    photograph is never bleached merely because black was tried first. Then *which colour* -
    the complement of the surface's own hue rather than flat black or white, because ink
    opposite the paper on the colour wheel is what a designer would reach for, and a page
    of black text on bright blue water looks like a mistake even when it is legible.

    `size` should be the page's own proportions: the factory centre-crops to whatever aspect
    it is handed, so probing a square when the page is a tall receipt measures a region the
    page will never show.

    Returns the factory to draw with and the ink to draw in.
    """
    longest = max(size)
    probe_size = (max(8, round(size[0] * 96 / longest)), max(8, round(size[1] * 96 / longest)))
    probe = factory(*probe_size).convert("RGB")
    array = np.asarray(probe, dtype=np.float32)
    luma = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    darkest, brightest = float(np.percentile(luma, 4)), float(np.percentile(luma, 96))

    hue, surface_saturation, hue_variance, colourfulness = _hue_profile(probe)
    complement = (hue + _COMPLEMENT) % 1.0
    many_hued = hue_variance > _MAX_HUE_VARIANCE and colourfulness > _MIN_COLOURFULNESS
    # A grey surface has no hue to oppose; tinting ink against it invents a colour cast the
    # page has no reason to carry, so those stay neutral.
    ink_saturation = 0.0 if surface_saturation < 0.12 else rng.uniform(0.35, 0.7)

    # Ink has latitude - anywhere in its luminance band will do - so the band is searched
    # rather than sampled and then tested. Sampling first and checking afterwards made a
    # borderline surface like brickwork a coin flip: the same wall kept its photograph or
    # got bleached depending on which shade of white came up.
    darkest_ink, lightest_ink = _DARK_INK_LUMA[0], _LIGHT_INK_LUMA[1]
    dark_margin = darkest - darkest_ink
    light_margin = lightest_ink - brightest

    colourful = colourfulness >= _MIN_COLOURFULNESS

    # Black first. If a wash no heavier than `_MIN_KEEP_FOR_DARK` buys dark ink the room it
    # needs, that is the page - a lightened field printed in black, which is what a document
    # on coloured stock actually looks like.
    if not many_hued and dark_margin < _MIN_SEPARATION:
        trial = rng.choice(_TONED_INKS)
        paper_level = _luminance((250, 248, 244))
        needed = (paper_level - _luminance(trial) - _MIN_SEPARATION) / (paper_level - darkest)
        if _MIN_KEEP_FOR_DARK <= needed < 1.0:
            keep = min(_MAX_TONED_KEEP, needed)

            def lightened(width: int, height: int) -> Image.Image:
                base = factory(width, height).convert("RGB")
                paper = Image.new("RGB", (width, height), (250, 248, 244))
                return Image.blend(paper, base, keep)

            return lightened, trial

    if max(dark_margin, light_margin) >= _MIN_SEPARATION and not many_hued:
        if light_margin > dark_margin:
            # Every shade at least this light clears the floor; take the brightest end of
            # it, tinted only faintly and with the surface's own hue rather than against it.
            floor = max(_LIGHT_INK_LUMA[0], brightest + _MIN_SEPARATION)
            luma = rng.uniform(max(floor, _LIGHT_INK_FLOOR), _LIGHT_INK_LUMA[1])
            return factory, _ink_at(hue, luma, rng.uniform(*_LIGHT_INK_SATURATION))
        else:
            ceiling = min(_DARK_INK_LUMA[1], darkest - _MIN_SEPARATION)
            luma = rng.uniform(max(_DARK_INK_LUMA[0], ceiling - _INK_LATITUDE), ceiling)
            if not colourful or luma >= _HUE_VISIBLE[0]:
                return factory, _ink_at(complement, luma, ink_saturation)


    # Either no ink clears the surface, or it has too many hues for one complement to
    # oppose, or it is bright enough that ink against it could never show its colour. All
    # three are the same fix: lay paper over it and print in black.
    dark = rng.choice(_TONED_INKS)
    paper_color = (250, 248, 244)
    paper_level = _luminance(paper_color)
    headroom = paper_level - _luminance(dark) - _MIN_SEPARATION
    solved = headroom / (paper_level - darkest) if paper_level > darkest else 0.0
    # Whichever is the stronger wash: what legibility demands, or what makes the layer
    # visible as a layer.
    keep = max(0.15, min(_MAX_TONED_KEEP, solved))

    def toned(width: int, height: int) -> Image.Image:
        base = factory(width, height).convert("RGB")
        paper = Image.new("RGB", (width, height), paper_color)
        return Image.blend(paper, base, keep)

    return toned, dark


class BackgroundSampler:
    """Samples one background factory per document from weighted kinds."""

    def __init__(
        self,
        kinds: dict[str, float],
        *,
        image_paths: Sequence[str | Path] = (),
        tint_range: tuple[Color, Color] = ((238, 234, 226), (255, 255, 255)),
    ):
        self.kinds = {name: weight for name, weight in kinds.items() if weight > 0}
        if not self.kinds:
            raise ValueError("BackgroundSampler needs at least one kind with a positive weight")
        self.images = self._collect_images(image_paths)
        if "image" in self.kinds and not self.images:
            raise ValueError("background kind 'image' was requested but no images were found")
        self.tint_range = tint_range

    @staticmethod
    def _collect_images(paths: Sequence[str | Path]) -> tuple[Path, ...]:
        found: list[Path] = []
        for raw in paths:
            root = Path(raw)
            if root.is_file() and root.suffix.lower() in _IMAGE_EXTENSIONS:
                found.append(root)
            elif root.is_dir():
                found.extend(
                    path
                    for path in sorted(root.rglob("*"))
                    if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
                )
        return tuple(found)

    def _tint(self, rng: random.Random) -> Color:
        """Interpolate between the two endpoint colours with a single factor.

        Sampling each channel independently would produce hues no paper stock has — a
        pink or green page — because the channels of real paper move together.
        """
        low, high = self.tint_range
        factor = rng.random()
        return tuple(  # type: ignore[return-value]
            int(round(a + (b - a) * factor)) for a, b in zip(low, high, strict=True)
        )

    def sample(self, rng: random.Random) -> tuple[BackgroundFactory, str]:
        """Return a factory and the name of the kind it came from, for provenance."""
        names = list(self.kinds)
        kind = rng.choices(names, weights=[self.kinds[name] for name in names], k=1)[0]
        if kind == "solid":
            return solid_background(self._tint(rng)), kind
        if kind == "gradient":
            return (
                gradient_background(
                    self._tint(rng),
                    self._tint(rng),
                    rng.choice(["horizontal", "vertical", "diagonal"]),
                ),
                kind,
            )
        if kind == "image":
            return image_background(rng.choice(self.images)), kind
        return paper_background(self._tint(rng), rng.uniform(2.0, 9.0), rng), "paper"
