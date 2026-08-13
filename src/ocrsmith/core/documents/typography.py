"""Typography: which font and style each kind of block is drawn in.

Documents are typographically *coherent* — a page does not pick a random typeface per
paragraph. Sampling one family per document and varying only weight and size reproduces
that, and it is what makes a synthetic corpus look like documents rather than like a font
catalogue. The variation that does matter for OCR (size, spacing, colour, alignment) is
sampled per document instead.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL.ImageFont import FreeTypeFont

from ...domain.annotations import RegionType
from ..fonts import font_variations, load_font
from ..rendering.style import Alignment, TextStyle

__all__ = [
    "Face",
    "FontFamily",
    "RoleTypography",
    "Typography",
    "TypographySampler",
    "expand_faces",
    "group_font_families",
]

#: Weight keywords ordered from lightest to heaviest, used to rank faces within a family.
_WEIGHT_ORDER = (
    "thin",
    "extralight",
    "ultralight",
    "light",
    "regular",
    "book",
    "medium",
    "semibold",
    "demibold",
    "bold",
    "extrabold",
    "black",
    "heavy",
)


@dataclass(frozen=True, slots=True)
class Face:
    """One drawable face: a file, plus a named instance when the file is variable."""

    path: Path
    variation: str | None = None

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True, slots=True)
class FontFamily:
    """The faces of one typeface, ranked from lightest to heaviest.

    A variable font is expanded into its named instances. Without that, `light`, `regular`
    and `bold` of a variable family all resolve to the same default instance — and about
    half of the Arabic families on Google Fonts are variable, so the loss is substantial.
    """

    name: str
    faces: tuple[Face, ...]

    @property
    def regular(self) -> Face:
        return self.faces[len(self.faces) // 2] if len(self.faces) > 2 else self.faces[0]

    @property
    def bold(self) -> Face:
        return self.faces[-1]

    @property
    def light(self) -> Face:
        return self.faces[0]


def _weight_rank(face: Face) -> int:
    label = (face.variation or face.stem).lower().replace("-", "").replace("_", "").replace(" ", "")
    for rank, keyword in enumerate(_WEIGHT_ORDER):
        if keyword in label:
            return rank
    return _WEIGHT_ORDER.index("regular")


def _family_name(path: Path) -> str:
    """The family part of a filename, ignoring weight and variable-axis suffixes.

    Google Fonts names variable files `Alexandria[wght].ttf`, so the axis list has to be
    stripped as well as the `-Bold` style suffix.
    """
    return path.stem.split("[")[0].split("-")[0].split("_")[0]


def expand_faces(path: Path) -> tuple[Face, ...]:
    """A static font is one face; a variable font is one face per named instance."""
    variations = font_variations(path)
    if not variations:
        return (Face(path),)
    return tuple(Face(path, name) for name in variations)


def group_font_families(paths: Sequence[Path | str]) -> tuple[FontFamily, ...]:
    """Group font files into families, expanding variable fonts into their instances.

    `Amiri-Bold.ttf` and `Amiri-Regular.ttf` are two faces of one family; treating them as
    unrelated is what produces documents whose heading and body look like different eras.
    """
    families: dict[str, list[Face]] = {}
    for raw in paths:
        path = Path(raw)
        families.setdefault(_family_name(path), []).extend(expand_faces(path))
    return tuple(
        FontFamily(name, tuple(sorted(faces, key=_weight_rank)))
        for name, faces in sorted(families.items())
        if faces
    )


@dataclass(frozen=True, slots=True)
class RoleTypography:
    """How one kind of block is drawn, and how much air surrounds it."""

    font: FreeTypeFont
    style: TextStyle
    space_before: float = 0.0
    space_after: float = 0.0


@dataclass(frozen=True, slots=True)
class Typography:
    """A per-role lookup with a body-text fallback."""

    body: RoleTypography
    roles: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.roles is None:
            object.__setattr__(self, "roles", {})

    def for_(self, region_type: RegionType) -> RoleTypography:
        return self.roles.get(region_type, self.body)

    def with_role(self, region_type: RegionType, role: RoleTypography) -> Typography:
        return Typography(self.body, {**self.roles, region_type: role})


class TypographySampler:
    """Samples a coherent `Typography` per document from a pool of font files."""

    def __init__(
        self,
        font_paths: Sequence[Path | str],
        *,
        body_size_range: tuple[int, int] = (18, 30),
        families: Sequence[FontFamily] | None = None,
    ):
        self.families = tuple(families) if families else group_font_families(font_paths)
        if not self.families:
            raise ValueError("TypographySampler needs at least one font file")
        self.body_size_range = body_size_range

    def sample(self, rng: random.Random | None = None, *, direction=None) -> Typography:
        rng = rng or random.Random()
        family = rng.choice(self.families)
        body_size = rng.randint(*self.body_size_range)
        align = Alignment.NATURAL
        # Relative to ascender-to-descender height, not to the em size: Arabic faces have
        # tall metrics, so the range that reads as "normal leading" sits close to 1.0.
        line_spacing = rng.uniform(0.9, 1.15)
        ink = rng.choice([(0, 0, 0), (16, 16, 16), (32, 32, 40), (10, 24, 48)])

        def role(
            face: Face,
            size: float,
            *,
            spacing: float | None = None,
            before: float = 0.0,
            after: float = 0.0,
            **style_kwargs,
        ) -> RoleTypography:
            font = load_font(face.path, max(6, int(round(size))), face.variation)
            style = TextStyle(
                color=ink,
                line_spacing=spacing if spacing is not None else line_spacing,
                **{"align": align, **style_kwargs},
            )
            return RoleTypography(font, style, space_before=before, space_after=after)

        body = role(family.regular, body_size, before=0, after=body_size * 0.6)
        roles = {
            RegionType.TITLE: role(
                family.bold, body_size * rng.uniform(1.8, 2.4), spacing=0.95, after=body_size
            ),
            RegionType.HEADING: role(
                family.bold,
                body_size * rng.uniform(1.25, 1.6),
                spacing=0.95,
                before=body_size * 0.8,
                after=body_size * 0.4,
            ),
            RegionType.CAPTION: role(
                family.light,
                body_size * 0.85,
                after=body_size * 0.7,
                align=Alignment.CENTER,
            ),
            RegionType.HEADER: role(family.light, body_size * 0.85),
            RegionType.FOOTER: role(family.light, body_size * 0.8),
            RegionType.PAGE_NUMBER: role(family.regular, body_size * 0.85),
            RegionType.QUOTE: role(family.light, body_size, before=body_size * 0.5, after=body_size * 0.5),
            RegionType.LIST: role(family.regular, body_size, after=body_size * 0.6),
            RegionType.KEY_VALUE: role(family.regular, body_size * 0.95, after=body_size * 0.25),
            RegionType.TABLE: role(family.regular, body_size * 0.9, after=body_size * 0.8),
            RegionType.CODE: role(family.regular, body_size * 0.9, spacing=1.15, after=body_size * 0.6),
            RegionType.FORMULA: role(family.regular, body_size * 1.05, after=body_size * 0.6),
        }
        return Typography(body, roles)
