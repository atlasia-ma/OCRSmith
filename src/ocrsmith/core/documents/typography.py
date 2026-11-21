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
    "handwriting_families_from_manifest",
    "tint",
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


#: Second colours real documents are actually printed in. Every one is dark enough to
#: read as text on light stock, because an accent that needs a lighter weight to work is
#: a design choice this generator has no way to make. Sampled as a family so a corpus
#: contains recognisable house styles rather than a uniform spread over the colour wheel.
_ACCENTS: tuple[tuple[int, int, int], ...] = (
    (21, 54, 112),  # corporate navy
    (10, 74, 110),  # petrol
    (112, 26, 32),  # maroon, as on letterheads and certificates
    (24, 84, 56),  # forest, as on official forms
    (128, 70, 10),  # ochre
    (74, 32, 102),  # aubergine
    (16, 16, 16),  # no accent at all, just a heavier black
)


def tint(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """`color` mixed towards white by `amount`, for a pale rule or a header band."""
    return tuple(int(round(channel + (255 - channel) * amount)) for channel in color)  # type: ignore[return-value]


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


#: Families whose letterforms come from a written hand rather than from a drawn type
#: design. Curated, because the obvious signals do not work.
#:
#: Google Fonts' own `category` is authoritative where it says "Handwriting", and
#: `handwriting_families_from_manifest` reads it — but across the entire Arabic subset it
#: says that of exactly *one* family, Playpen Sans Arabic. Meanwhile the styles that
#: actually matter for Arabic handwriting recognition are filed under Serif and Display:
#: ruqʿah is what Arabic speakers write by hand every day, and Aref Ruqaa is categorised
#: Serif. A corpus built on the category alone would contain one hand.
#:
#: Each entry below is a script style a person writes in, not a poster face.
_HANDWRITTEN_FAMILIES = frozenset(
    {
        "playpensansarabic",  # informal handwriting, the one GF labels as such
        "arefruqaa",  # ruqʿah: the everyday Arabic hand
        "arefruqaaink",
        "notonastaliqurdu",  # nastaliq: the hanging Perso-Arabic hand
        "gulzar",  # nastaliq
        "mirza",  # nastaliq-influenced
        "katibeh",  # "katib", the scribe; a calligraphic hand
        "lateef",  # a flowing naskh, used for handwritten-style setting
        "jomhuria",  # heavy calligraphic brush
        "rakkas",  # calligraphic display with a brush origin
        "scheherazade",  # naskh with a strong calligraphic descent
        "scheherazadenew",
    }
)

#: Fragments that are diagnostic of a hand in a filename, for a pool assembled from an
#: arbitrary directory where the family is not one we know.
_HANDWRITING_HINTS = (
    "hand",
    "script",
    "ruqaa",
    "ruqa",
    "thuluth",
    "diwani",
    "diwan",
    "nastaliq",
    "nastaleeq",
    "calligra",
    "khatt",
    "cursive",
)

#: Fragments that must never count as a hand, whatever else matches. Kufi is the angular,
#: geometric, architectural Arabic script — the furthest thing from handwriting there is,
#: and `kufi` used to be in the hint list, which is why `NotoKufiArabic` was the only
#: "handwriting" family in the bundled pool and why handwritten pages were set in it.
#: `handjet` is here because it matches the "hand" hint and is a modular pixel face.
_NOT_HANDWRITING = ("kufi", "kufam", "mono", "code", "handjet")


def _normalise_family(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _looks_handwritten(name: str) -> bool:
    """Whether a family name denotes a face derived from a written hand.

    A name is a weak signal and this is a heuristic, but it is a heuristic that has to
    fail *closed*: setting a document in a display face because its name looked cursive
    produces an odd-looking page, while the reverse — a corpus that claims handwriting
    and contains none — is a dataset that teaches a model nothing about handwriting.
    """
    lowered = name.lower()
    if any(fragment in lowered for fragment in _NOT_HANDWRITING):
        return False
    if _normalise_family(name) in _HANDWRITTEN_FAMILIES:
        return True
    return any(hint in lowered for hint in _HANDWRITING_HINTS)


def handwriting_families_from_manifest(paths: Sequence[Path | str]) -> frozenset[str]:
    """Family names that `ocrsmith fetch-fonts` recorded as Google Fonts "Handwriting".

    The manifest is written beside the fonts it fetched, so when a pool came from there
    its categories are known rather than guessed. Returns an empty set when there is no
    manifest, which is the normal case for a hand-assembled directory.
    """
    from ...assets.fonts import load_manifest

    found: set[str] = set()
    seen: set[Path] = set()
    candidates: list[Path] = []
    for raw in paths:
        directory = Path(raw)
        directory = directory if directory.is_dir() else directory.parent
        # `fetch-fonts` writes one manifest at the root it was given and puts each family
        # in its own subdirectory, so a pool built from the font *files* is two levels
        # below it. Walk up a little rather than only looking where we were pointed.
        for level in (directory, *directory.parents[:2]):
            if level not in seen:
                seen.add(level)
                candidates.append(level)
    for directory in candidates:
        for family in load_manifest(directory).get("families", ()):
            if str(family.get("category", "")).strip().lower() == "handwriting":
                found.add(_normalise_family(str(family.get("family", ""))))
    return frozenset(found)


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
    #: The document's accent colour, or None for an all-black document. Headings are set
    #: in it and the table rules pick it up, so the whole sheet stays coherent — which is
    #: the point: a real document's second colour appears in several places or in none.
    accent: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.roles is None:
            object.__setattr__(self, "roles", {})

    def for_(self, region_type: RegionType) -> RoleTypography:
        return self.roles.get(region_type, self.body)

    def with_role(self, region_type: RegionType, role: RoleTypography) -> Typography:
        return Typography(self.body, {**self.roles, region_type: role}, self.accent)


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
        # The manifest's categories where the pool came from `fetch-fonts`, the curated
        # list and the name hints otherwise. See `_HANDWRITTEN_FAMILIES` for why the
        # category alone is not enough.
        from_manifest = handwriting_families_from_manifest(list(font_paths)) if font_paths else frozenset()
        self.handwriting_families = tuple(
            family
            for family in self.families
            if _looks_handwritten(family.name) or _normalise_family(family.name) in from_manifest
        )

    def sample(
        self,
        rng: random.Random | None = None,
        *,
        direction=None,
        handwritten: bool = False,
        ink: tuple[int, int, int] | None = None,
        heavy: bool = False,
        accent_probability: float = 0.4,
    ) -> Typography:
        """Sample a coherent typography.

        `handwritten` biases towards a handwriting face and loosens the setting - a hand
        does not hold a constant baseline or an even word gap, and reproducing that
        wobble is most of what separates a handwritten page from a printed one, given
        that the letterforms themselves still come from a font.
        """
        rng = rng or random.Random()
        families = self.handwriting_families if handwritten and self.handwriting_families else self.families
        family = rng.choice(families)
        body_size = rng.randint(*self.body_size_range)
        align = Alignment.NATURAL
        # Relative to ascender-to-descender height, not to the em size: Arabic faces have
        # tall metrics, so the range that reads as "normal leading" sits close to 1.0.
        line_spacing = rng.uniform(1.05, 1.45) if handwritten else rng.uniform(0.9, 1.15)
        # An explicit ink comes from the background: a dark surface needs light ink, and
        # that decision cannot be made here because typography never sees the page it will
        # be drawn on. Absent one, the usual near-blacks.
        sampled_ink = (
            rng.choice([(18, 26, 84), (12, 40, 96), (30, 30, 34)])
            if handwritten
            else rng.choice([(0, 0, 0), (16, 16, 16), (32, 32, 40), (10, 24, 48)])
        )
        ink = ink if ink is not None else sampled_ink
        jitter = (
            {"baseline_jitter": rng.uniform(1.0, 3.0), "word_spacing_jitter": rng.uniform(1.5, 5.0)}
            if handwritten
            else {}
        )

        # A second colour, for the documents that have one. Only on dark ink: the ink is
        # light because the stock is dark, and a navy heading on a photograph of slate is
        # unreadable in a way no amount of realism justifies. Handwriting has no accent
        # either — a hand writes in one colour.
        accent = None
        if not handwritten and sum(ink) / 3 <= 128 and rng.random() < accent_probability:
            accent = rng.choice(_ACCENTS)

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
                line_spacing=spacing if spacing is not None else line_spacing,
                **{"color": ink, "align": align, **jitter, **style_kwargs},
            )
            return RoleTypography(font, style, space_before=before, space_after=after)

        # Body text over a photograph is set in the bold face. A regular weight lays a
        # one- or two-pixel stem over paper that is itself textured, and the texture wins:
        # the strokes compete with grain, fibre and print noise instead of sitting on top of
        # them. Setting text over an image heavier is what a designer does for the same
        # reason, and it is the difference between a caption that survives a capture
        # condition and one that dissolves into it.
        body = role(
            family.bold if heavy else family.regular, body_size, before=0, after=body_size * 0.6
        )
        # The accent reaches the title always and the headings usually. A document whose
        # title is coloured and whose headings are black is a real layout; one where every
        # level is coloured is also real; one where a random subset of headings is
        # coloured is not, so the choice is made once per document rather than per block.
        heading_colour = accent if accent and rng.random() < 0.75 else ink
        roles = {
            RegionType.TITLE: role(
                family.bold,
                body_size * rng.uniform(1.8, 2.4),
                spacing=0.95,
                after=body_size,
                color=accent or ink,
            ),
            RegionType.HEADING: role(
                family.bold,
                body_size * rng.uniform(1.25, 1.6),
                spacing=0.95,
                before=body_size * 0.8,
                after=body_size * 0.4,
                color=heading_colour,
            ),
            # Regular, not light. These roles are already set below body size, and a light
            # weight on top of that gave strokes thin enough to dissolve in the capture
            # chain: captions, running heads and footers measured a median 116-146 levels
            # of ink against the paper where body text measured 217, and the worst of them
            # reached 39 - invisible, with the label still claiming every word. Secondary
            # text is distinguished by size and placement; it does not also need to be
            # frail.
            RegionType.CAPTION: role(
                family.regular,
                body_size * 0.85,
                after=body_size * 0.7,
                align=Alignment.CENTER,
            ),
            RegionType.HEADER: role(family.regular, body_size * 0.85),
            RegionType.FOOTER: role(family.regular, body_size * 0.8),
            RegionType.PAGE_NUMBER: role(family.regular, body_size * 0.85),
            RegionType.QUOTE: role(family.light, body_size, before=body_size * 0.5, after=body_size * 0.5),
            RegionType.LIST: role(family.regular, body_size, after=body_size * 0.6),
            RegionType.KEY_VALUE: role(family.regular, body_size * 0.95, after=body_size * 0.25),
            RegionType.TABLE: role(family.regular, body_size * 0.9, after=body_size * 0.8),
            RegionType.CODE: role(family.regular, body_size * 0.9, spacing=1.15, after=body_size * 0.6),
            RegionType.FORMULA: role(family.regular, body_size * 1.05, after=body_size * 0.6),
        }
        return Typography(body, roles, accent)
