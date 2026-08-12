"""Geometric degradations: everything that moves the page under the lens.

These are the dangerous ones. Whatever transform is applied to the pixels must be applied
to every box and polygon in the annotation, or the dataset silently teaches a detector to
predict boxes that are systematically offset. Each class here derives an explicit forward
point mapping and passes the whole annotation through it.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image

from ...domain.annotations import Page
from .base import Degradation, map_page

__all__ = ["PerspectiveWarp", "Rotation", "perspective_coefficients"]


def perspective_coefficients(
    source: list[tuple[float, float]], target: list[tuple[float, float]]
) -> tuple[float, ...]:
    """Coefficients of the homography mapping `source` corners onto `target` corners.

    Pillow's `Image.transform(..., PERSPECTIVE, coeffs)` samples the *input* for each
    output pixel, so it needs the inverse of the visual mapping. Solving the system here
    once lets the caller keep thinking in terms of "where does this corner end up".
    """
    matrix = []
    for (sx, sy), (tx, ty) in zip(source, target, strict=True):
        matrix.append([sx, sy, 1, 0, 0, 0, -tx * sx, -tx * sy])
        matrix.append([0, 0, 0, sx, sy, 1, -ty * sx, -ty * sy])
    a = np.array(matrix, dtype=np.float64)
    b = np.array([value for point in target for value in point], dtype=np.float64)
    solution = np.linalg.solve(a, b)
    return tuple(float(value) for value in solution)


class Rotation(Degradation):
    """Page skew, as produced by a sheet fed slightly crooked into a scanner."""

    def __init__(
        self,
        max_angle: float = 4.0,
        fill: tuple[int, int, int] = (255, 255, 255),
        probability: float = 1.0,
    ):
        self.max_angle = max_angle
        self.fill = fill
        self.probability = probability

    def apply(self, image: Image.Image, page: Page, rng: random.Random):
        angle = rng.uniform(-self.max_angle, self.max_angle)
        if abs(angle) < 1e-6:
            return image, page, {"angle": 0.0}

        width, height = image.size
        rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=self.fill)
        new_width, new_height = rotated.size

        # Pillow rotates anticlockwise about the centre, then expands the canvas; the
        # forward mapping therefore rotates by -angle in screen coordinates and recentres.
        radians = math.radians(-angle)
        cos, sin = math.cos(radians), math.sin(radians)
        cx, cy = width / 2, height / 2
        ncx, ncy = new_width / 2, new_height / 2

        def mapper(x: float, y: float) -> tuple[float, float]:
            dx, dy = x - cx, y - cy
            return (ncx + dx * cos - dy * sin, ncy + dx * sin + dy * cos)

        return rotated, map_page(page, mapper, new_width, new_height), {"angle": angle}


class PerspectiveWarp(Degradation):
    """A page photographed at an angle rather than scanned flat.

    Each corner is displaced independently by up to `magnitude` of the page size, which is
    what separates a phone snapshot from a flatbed scan and is the transform most
    synthetic corpora omit.
    """

    def __init__(
        self,
        magnitude: float | tuple[float, float] = (0.01, 0.06),
        fill: tuple[int, int, int] = (255, 255, 255),
        probability: float = 1.0,
    ):
        self.magnitude = magnitude
        self.fill = fill
        self.probability = probability

    def apply(self, image: Image.Image, page: Page, rng: random.Random):
        if isinstance(self.magnitude, (tuple, list)):
            magnitude = rng.uniform(float(self.magnitude[0]), float(self.magnitude[1]))
        else:
            magnitude = float(self.magnitude)

        width, height = image.size
        source = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        jitter_x, jitter_y = magnitude * width, magnitude * height
        target = [
            (x + rng.uniform(-jitter_x, jitter_x), y + rng.uniform(-jitter_y, jitter_y)) for x, y in source
        ]

        # Shift so the warped page starts at the origin, and grow the canvas to hold it.
        min_x = min(x for x, _ in target)
        min_y = min(y for _, y in target)
        target = [(x - min_x, y - min_y) for x, y in target]
        new_width = max(1, int(math.ceil(max(x for x, _ in target))))
        new_height = max(1, int(math.ceil(max(y for _, y in target))))

        inverse = perspective_coefficients(target, source)
        warped = image.convert("RGB").transform(
            (new_width, new_height),
            Image.PERSPECTIVE,
            inverse,
            resample=Image.BICUBIC,
            fillcolor=self.fill,
        )

        forward = perspective_coefficients(source, target)
        a, b, c, d, e, f, g, h = forward

        def mapper(x: float, y: float) -> tuple[float, float]:
            denominator = g * x + h * y + 1.0
            if abs(denominator) < 1e-9:
                denominator = 1e-9
            return ((a * x + b * y + c) / denominator, (d * x + e * y + f) / denominator)

        return (
            warped,
            map_page(page, mapper, new_width, new_height),
            {
                "magnitude": magnitude,
                "corners": [(round(x, 2), round(y, 2)) for x, y in target],
            },
        )
