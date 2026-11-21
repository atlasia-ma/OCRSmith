"""The working surface a degradation pipeline passes between its stages.

A capture pipeline is a dozen stages long, and most of them are arithmetic on the pixels:
add grain, multiply by a shading field, add a glare blob, add sensor noise. Written the
obvious way, each one decodes the PIL image into a float array, does a few microseconds of
real work, clips, rounds back to `uint8` and re-wraps it. On an A4 page at 200 dpi that
round trip costs about 50 ms, so a thirteen-stage `photo` preset spends roughly two thirds
of a second per page converting between two representations of the same pixels — more than
it spends on the optics, the noise and the compression put together.

`Surface` holds the page in whichever representation was last written and converts only
when the other one is asked for. A run of array stages therefore shares one float32 buffer,
and the quantisation to 8 bits happens once, at the end, instead of after every stage.

Two consequences worth stating, because they are the reason this is a `Surface` and not
just "keep a numpy array around":

* **Intermediate values are not clipped.** A shadow that takes a pixel to -3 and a glare
  that brings it back to 240 now round-trips cleanly, where the old code clamped it to 0
  and lost the information. Clipping happens once, where the pixels become bytes.
* **It is a value, not a view.** `as_array` hands back a buffer the caller may write into
  in place; `from_array` adopts it. Nothing else holds a reference, so no stage can
  corrupt another's input.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["Surface"]


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """An RGB view of `image`, compositing transparency onto white rather than dropping it."""
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.split()[-1])
        return flat
    return image.convert("RGB")


class Surface:
    """A page held as a PIL image, a float32 array, or both.

    Stages take and return a `Surface`; the conversion between representations happens at
    most once per boundary between a PIL-native stage (a resample, a JPEG round trip, a
    morphological filter) and a run of arithmetic ones.
    """

    __slots__ = ("_image", "_array", "_size")

    def __init__(self, image: Image.Image | None = None, array: np.ndarray | None = None):
        if (image is None) == (array is None):
            raise ValueError("a Surface is built from exactly one of an image or an array")
        if image is not None:
            image = _flatten_to_rgb(image)
            self._size = image.size
        else:
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"expected an (H, W, 3) array, got {array.shape}")
            self._size = (array.shape[1], array.shape[0])
        self._image = image
        self._array = array

    # -- construction ------------------------------------------------------

    @classmethod
    def of(cls, image: Image.Image) -> Surface:
        return cls(image=image)

    @classmethod
    def from_array(cls, array: np.ndarray) -> Surface:
        """Adopt `array` as the surface. Values may sit outside 0–255; see the module docstring."""
        return cls(array=np.asarray(array, dtype=np.float32))

    # -- representations ---------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        """`(width, height)`, available without forcing a conversion."""
        return self._size

    def as_array(self) -> np.ndarray:
        """The pixels as a writable `(H, W, 3)` float32 array."""
        if self._array is None:
            self._array = np.asarray(self._image, dtype=np.float32)
            # Decoding a PIL image yields a read-only view onto its buffer; stages are
            # allowed to work in place, so hand them something they may actually write to.
            if not self._array.flags.writeable:
                self._array = self._array.copy()
            self._image = None  # the array is now the live copy
        return self._array

    def as_image(self) -> Image.Image:
        """The pixels as an RGB `Image`, clipping and rounding if they are still floats."""
        if self._image is None:
            # Clipped in place. `np.clip(a, 0, 255)` allocates a second copy of the page
            # — 21 MB for an A4 at 200 dpi — purely to throw it away one line later.
            np.clip(self._array, 0.0, 255.0, out=self._array)
            self._image = Image.fromarray(self._array.astype(np.uint8), "RGB")
            self._array = None  # the image is now the live copy
        return self._image

    # -- transitions -------------------------------------------------------

    def with_array(self, array: np.ndarray) -> Surface:
        """A surface holding `array`, which a stage has just produced."""
        return Surface.from_array(array)

    def with_image(self, image: Image.Image) -> Surface:
        """A surface holding `image`, which a PIL-native stage has just produced."""
        return Surface.of(image)
