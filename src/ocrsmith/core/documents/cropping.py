"""Partial page captures.

Real scans are often not whole pages. A sheet fed crooked loses its foot, a phone photo
catches the top two thirds, a book scanner clips the gutter edge. A model trained only on
complete pages reads those badly, because it has never seen a page that simply stops.

The rule that makes a crop honest is about where it *starts*. A capture of the bottom of a
page must read as a continuation - text already under way, no title at the top - because a
title at the top of a fragment tells the model it is looking at the start of a document
when it is not. So a crop boundary is never allowed to land just above a heading: it is
moved down past it, and the fragment opens mid-section the way the real thing does.

The annotation is carried through the same transform: regions fully outside are dropped,
the rest are translated into the crop's coordinates and clipped to it, so the label
describes the fragment rather than the page it came from.
"""

from __future__ import annotations

import random

from PIL import Image

from ...domain import Page, RegionType

__all__ = ["CropMode", "crop_page"]

CropMode = str

#: Region types that announce the start of something. A fragment must not begin with one.
_OPENERS = (RegionType.TITLE, RegionType.HEADING)

#: A region must keep at least this much of its height to stay in the annotation. Half a
#: line of text at the cut edge is not readable and must not be labelled as though it is.
_MIN_VISIBLE = 0.6


def _avoid_opening_on_a_heading(page: Page, top: float, bottom: float) -> float:
    """Push `top` below any title or heading that straddles or just follows it."""
    for region in sorted(page.regions, key=lambda r: r.bbox.y0):
        if region.type not in _OPENERS:
            continue
        # A heading within the first tenth of the fragment would read as its title.
        window = top + (bottom - top) * 0.1
        if top - 1 <= region.bbox.y0 <= window:
            top = min(region.bbox.y1 + 1, bottom - 1)
    return top


def crop_page(
    image: Image.Image,
    page: Page,
    rng: random.Random,
    *,
    mode: CropMode = "bottom",
    keep: float = 0.6,
) -> tuple[Image.Image, Page]:
    """Return a fragment of `image` and the annotation that describes only that fragment."""
    height = page.height
    band = max(1.0, min(1.0, keep) * height)

    if mode == "top":
        top, bottom = 0.0, band
    elif mode == "middle":
        top = rng.uniform(0.15, 0.85 - keep) * height if keep < 0.7 else (height - band) / 2
        top = max(0.0, top)
        bottom = min(float(height), top + band)
    else:  # bottom
        top, bottom = float(height) - band, float(height)

    if mode != "top":
        # Only a top capture may open on a title: it is the top of the page.
        top = _avoid_opening_on_a_heading(page, top, bottom)

    top_i, bottom_i = int(round(top)), int(round(bottom))
    if bottom_i - top_i < 8:
        return image, page

    cropped = image.crop((0, top_i, image.width, bottom_i))

    regions = []
    for region in page.regions:
        box = region.bbox
        visible = min(box.y1, bottom_i) - max(box.y0, top_i)
        span = max(1e-6, box.y1 - box.y0)
        if visible <= 0 or visible / span < _MIN_VISIBLE:
            continue
        moved = region.translate(0, -top_i)
        lines = tuple(
            line
            for line in moved.lines
            if (min(line.bbox.y1, bottom_i - top_i) - max(line.bbox.y0, 0))
            / max(1e-6, line.bbox.y1 - line.bbox.y0)
            >= _MIN_VISIBLE
        )
        if not lines and moved.lines:
            continue
        regions.append(
            type(moved)(
                moved.type,
                moved.bbox,
                lines,
                moved.table,
                moved.reading_order,
                moved.attributes,
            )
        )

    fragment = Page(
        width=page.width,
        height=bottom_i - top_i,
        regions=tuple(regions),
        direction=page.direction,
        attributes={**page.attributes, "capture": mode},
    ).clipped()
    return cropped, fragment
