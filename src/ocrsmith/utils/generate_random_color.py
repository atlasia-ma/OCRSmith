# src/ocrsmith/utils/generate_random_color.py

from typing import Tuple
from PIL import ImageColor
import random as rnd

def generate_random_color(color_range: str) -> Tuple[int, int, int]:
    colors = [ImageColor.getrgb(c) for c in color_range.split(",")]
    c1, c2 = colors[0], colors[-1]
    return (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )
    