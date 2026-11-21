"""Named degradation pipelines.

A corpus is only as good as its coverage of *capture conditions*. These presets describe
recognisable ones, so a dataset can be composed as "40% flatbed scan, 40% phone photo, 20%
clean" rather than as an undifferentiated cloud of noise. Ordering within each preset
follows the physical chain: the page is distorted by the optics first, then shaded, then
sampled and compressed.
"""

from __future__ import annotations

from PIL import Image

from .base import DegradationPipeline
from .geometric import PlanarCapture
from .photometric import (
    Bleedthrough,
    Blur,
    Brightness,
    Contrast,
    Downscale,
    Folds,
    GaussianNoise,
    Glare,
    InkErosion,
    InkSpread,
    JpegArtifacts,
    MotionBlur,
    PaperGrain,
    Shadow,
    Stains,
    Vignette,
)
from .physical import IlluminationField, PageCurl

__all__ = ["LINE_PRESETS", "PRESETS", "build_preset", "preset_names"]


def _clean() -> DegradationPipeline:
    """A born-digital render: no capture stage at all."""
    return DegradationPipeline()


def _scan() -> DegradationPipeline:
    """Flatbed scan: slight skew, paper texture, toner spread, mild compression.

    A sheet on glass is flat, so this is pure in-plane skew with no perspective, and it
    is resampled bicubically: a scan is the final image, with nothing downstream to hide
    a soft edge.
    """
    return DegradationPipeline(
        [
            PlanarCapture(max_angle=1.5, magnitude=0.0, probability=0.7),
            PaperGrain(strength=(4.0, 12.0), probability=0.8),
            Bleedthrough(strength=(0.03, 0.12), probability=0.25),
            InkSpread(size=3, probability=0.2),
            Blur(radius=(0.2, 0.7), probability=0.5),
            GaussianNoise(sigma=(2.0, 8.0), probability=0.6),
            Brightness(factor=(0.92, 1.08), probability=0.5),
            JpegArtifacts(quality=(60, 92), probability=0.6),
        ]
    )


def _photo() -> DegradationPipeline:
    """Phone photograph: perspective, uneven light, glare, motion, heavy compression.

    Ordered along the optical chain, and that ordering is load-bearing in both
    directions. The scene reaches the lens, the lens projects and defocuses it, the
    sensor samples it at its own resolution, and only then is there noise and
    compression — so `Downscale` sits *before* the sensor stages rather than near the
    end, which is both the physically right place for it and the reason this preset is
    fast: everything after it runs on a fifth to four fifths of the pixels.
    """
    return DegradationPipeline(
        [
            # The page is skewed and seen off-axis at once: one homography, one resample.
            # Bilinear, because six further stages of blur, noise and JPEG are about to
            # run over the result and nothing would survive to show the difference.
            # The area the warp exposes is the surface the page is lying on, so it is
            # filled with a neutral desk grey rather than white.
            PlanarCapture(
                max_angle=4.0,
                magnitude=(0.01, 0.06),
                fill=(118, 116, 112),
                resample=Image.BILINEAR,
                probability=0.9,
            ),
            # The sheet is not flat and the light is not even; both are what separates a
            # photographed page from a scanned one.
            PageCurl(strength=(0.02, 0.09), probability=0.35),
            # `Wrinkles` is deliberately absent from every preset. It works, it carries the
            # annotation through its displacement field correctly, and it stays in
            # `ocrsmith.core.degradations.physical` for anyone who wants it - but its
            # local ripple reads as a wobble in the text rather than as a crease in the
            # paper, and a corpus full of it looks synthetic in exactly the way this
            # project exists to avoid. Add it back explicitly if you want it.
            IlluminationField(strength=(0.12, 0.4), probability=0.7),
            Shadow(strength=(0.1, 0.45), probability=0.7),
            Glare(strength=(0.1, 0.4), probability=0.35),
            Vignette(strength=(0.1, 0.4), probability=0.5),
            MotionBlur(length=(3, 9), probability=0.3),
            Blur(radius=(0.3, 1.4), probability=0.5),
            # -- the sensor samples here -------------------------------------------
            Downscale(scale=(0.45, 0.9), probability=0.5),
            GaussianNoise(sigma=(4.0, 16.0), probability=0.7),
            JpegArtifacts(quality=(25, 70), probability=0.9),
        ]
    )


def _fax() -> DegradationPipeline:
    """Low-resolution, high-contrast transmission with broken strokes."""
    return DegradationPipeline(
        [
            PlanarCapture(max_angle=2.0, magnitude=0.0, resample=Image.BILINEAR, probability=0.5),
            Downscale(scale=(0.3, 0.55), probability=0.9),
            Contrast(factor=(1.4, 2.2), probability=0.9),
            InkErosion(size=3, probability=0.5),
            GaussianNoise(sigma=(6.0, 20.0), probability=0.7),
            JpegArtifacts(quality=(20, 55), probability=0.7),
        ]
    )


def _archive() -> DegradationPipeline:
    """An aged document: stains, folds, bleed-through and a faded, low-contrast image."""
    return DegradationPipeline(
        [
            # Nothing in this preset may attack the ink itself. Ageing marks and deforms
            # the *paper* - stains, folds, ridges, show-through from the far side - and
            # the page stays readable, which is why archives are worth scanning at all.
            # Erosion, a contrast fade and heavy grain used to stack on top of each other
            # here and left pages whose text sat within ~40 grey levels of the paper: no
            # reader could recover a word, while the label still claimed every one of
            # them. Degradations may obscure text; they may not erase it.
            PlanarCapture(max_angle=2.5, magnitude=0.0, probability=0.6),
            PaperGrain(strength=(4.0, 12.0), probability=0.9),
            Stains(count=(1, 4), strength=(0.05, 0.25), probability=0.8),
            Folds(count=(1, 3), strength=(0.08, 0.3), probability=0.6),
            Bleedthrough(strength=(0.08, 0.22), probability=0.6),
            Brightness(factor=(0.94, 1.05), probability=0.6),
            GaussianNoise(sigma=(3.0, 10.0), probability=0.6),
            JpegArtifacts(quality=(45, 85), probability=0.5),
        ]
    )


#: Preset name to factory. Factories rather than instances, so each caller gets a
#: pipeline it can mutate without affecting anyone else.
PRESETS = {
    "clean": _clean,
    "scan": _scan,
    "photo": _photo,
    "fax": _fax,
    "archive": _archive,
}


def preset_names() -> tuple[str, ...]:
    return tuple(sorted(PRESETS))


def build_preset(name: str, scale: str = "page") -> DegradationPipeline:
    """Instantiate the named preset pipeline at the given physical scale.

    `scale` is "page" for a whole sheet or "line" for a recognition strip. The names are
    the same at both scales so that a corpus specification does not have to be rewritten
    when it switches from document to line output — see the note above `LINE_PRESETS` for
    why the same name has to mean a different pipeline.
    """
    table = {"page": PRESETS, "line": LINE_PRESETS}.get(scale)
    if table is None:
        raise ValueError(f"Unknown degradation scale {scale!r}. Available: line, page")
    try:
        return table[name]()
    except KeyError:
        raise ValueError(
            f"Unknown degradation preset {name!r}. Available: {', '.join(preset_names())}"
        ) from None


# -- line-scale presets ----------------------------------------------------------------
#
# A degradation preset encodes a *physical scale* as well as a capture condition, and the
# page presets above are written for a sheet of paper. Running them over a 40-pixel strip
# does not produce a degraded line, it produces nonsense: a vignette across a strip is a
# horizontal gradient, a coffee stain is the whole image, a page curl bends a line that
# has no page to bend with, and a 4-degree skew throws most of the text off a canvas whose
# height is one line box.
#
# The line presets below describe the same capture conditions at the scale of a crop:
# the skew is what survives a deskew pass, the perspective is the residue of an
# uncorrected one, and the noise, blur and compression are the same because they are
# properties of the sensor and the codec, not of the sheet.


def _line_clean() -> DegradationPipeline:
    """A crop of a born-digital render."""
    return DegradationPipeline()


def _line_scan() -> DegradationPipeline:
    """A line cut from a flatbed scan."""
    return DegradationPipeline(
        [
            PlanarCapture(max_angle=0.8, magnitude=0.0, probability=0.6),
            PaperGrain(strength=(3.0, 10.0), probability=0.8),
            InkSpread(size=3, probability=0.15),
            Blur(radius=(0.2, 0.6), probability=0.5),
            GaussianNoise(sigma=(2.0, 7.0), probability=0.6),
            Brightness(factor=(0.92, 1.08), probability=0.5),
            JpegArtifacts(quality=(60, 92), probability=0.5),
        ]
    )


def _line_photo() -> DegradationPipeline:
    """A line cut from a photograph, after whatever deskewing the cropper managed.

    The skew and perspective here are small on purpose. A crop comes out of a detector,
    and a detector that returned a box 4 degrees off its line would have cut the
    ascenders off. What reaches a recogniser is the residue.
    """
    return DegradationPipeline(
        [
            PlanarCapture(
                max_angle=1.5,
                magnitude=(0.005, 0.02),
                resample=Image.BILINEAR,
                probability=0.8,
            ),
            IlluminationField(strength=(0.08, 0.3), scale=3, probability=0.6),
            Shadow(strength=(0.05, 0.3), probability=0.5),
            MotionBlur(length=(2, 6), probability=0.25),
            Blur(radius=(0.3, 1.1), probability=0.5),
            # The sensor samples here, as in `_photo`.
            Downscale(scale=(0.5, 0.95), probability=0.5),
            GaussianNoise(sigma=(3.0, 14.0), probability=0.7),
            JpegArtifacts(quality=(30, 75), probability=0.85),
        ]
    )


def _line_fax() -> DegradationPipeline:
    """A line from a low-resolution, high-contrast transmission."""
    return DegradationPipeline(
        [
            PlanarCapture(max_angle=1.0, magnitude=0.0, resample=Image.BILINEAR, probability=0.4),
            Downscale(scale=(0.35, 0.6), probability=0.9),
            Contrast(factor=(1.4, 2.2), probability=0.9),
            InkErosion(size=3, probability=0.4),
            GaussianNoise(sigma=(5.0, 18.0), probability=0.7),
            JpegArtifacts(quality=(20, 55), probability=0.7),
        ]
    )


def _line_archive() -> DegradationPipeline:
    """A line from an aged document: textured stock and show-through, text intact.

    Same rule as `_archive`: ageing marks the paper, it does not erase the ink.
    """
    return DegradationPipeline(
        [
            PlanarCapture(max_angle=1.2, magnitude=0.0, probability=0.5),
            PaperGrain(strength=(4.0, 12.0), probability=0.9),
            Bleedthrough(strength=(0.06, 0.18), probability=0.5),
            Brightness(factor=(0.94, 1.05), probability=0.6),
            GaussianNoise(sigma=(3.0, 9.0), probability=0.6),
            JpegArtifacts(quality=(45, 85), probability=0.5),
        ]
    )


#: Preset name to factory, at line scale. The names match `PRESETS` so a config written
#: for one scale is valid at the other and a corpus can be described once.
LINE_PRESETS = {
    "clean": _line_clean,
    "scan": _line_scan,
    "photo": _line_photo,
    "fax": _line_fax,
    "archive": _line_archive,
}
