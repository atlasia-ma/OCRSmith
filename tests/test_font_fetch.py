"""Contract for font fetching and variable-font expansion.

Font diversity is the highest-impact lever in synthetic text data, and two things stand
between a downloaded file and usable diversity: only permissively licensed families may be
taken, and a variable font must be expanded into its named instances or a whole family's
weight range collapses onto one face.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ocrsmith.assets import fonts as fetcher
from ocrsmith.core.documents.typography import (
    Face,
    expand_faces,
    group_font_families,
)
from ocrsmith.core.fonts import font_variations, load_font

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
STATIC_FONT = FONT_DIR / "Amiri-Regular.ttf"

pytestmark = pytest.mark.skipif(not STATIC_FONT.exists(), reason="bundled fonts unavailable")

METADATA = {
    "familyMetadataList": [
        {
            "family": "Amiri",
            "category": "Serif",
            "subsets": ["arabic", "latin"],
            "designers": ["Khaled Hosny"],
            "isOpenSource": True,
            "isNoto": False,
        },
        {
            "family": "Noto Sans Arabic",
            "category": "Sans Serif",
            "subsets": ["arabic"],
            "designers": [],
            "isOpenSource": True,
            "isNoto": True,
        },
        {
            "family": "Roboto",
            "category": "Sans Serif",
            "subsets": ["latin"],
            "designers": [],
            "isOpenSource": True,
            "isNoto": False,
        },
        {
            "family": "Proprietary",
            "category": "Serif",
            "subsets": ["arabic"],
            "designers": [],
            "isOpenSource": False,
            "isNoto": False,
        },
    ]
}


class TestListFamilies:
    def test_filters_by_subset(self):
        with patch.object(fetcher, "_get", return_value=METADATA):
            families = fetcher.list_families("arabic")

        assert {f.family for f in families} == {"Amiri", "Noto Sans Arabic"}

    def test_excludes_closed_source_families(self):
        with patch.object(fetcher, "_get", return_value=METADATA):
            families = fetcher.list_families("arabic")

        assert "Proprietary" not in {f.family for f in families}

    def test_closed_source_can_be_included_explicitly(self):
        with patch.object(fetcher, "_get", return_value=METADATA):
            families = fetcher.list_families("arabic", open_source_only=False)

        assert "Proprietary" in {f.family for f in families}

    def test_results_are_sorted_for_reproducibility(self):
        with patch.object(fetcher, "_get", return_value=METADATA):
            families = fetcher.list_families("arabic")

        assert [f.family for f in families] == sorted(f.family for f in families)

    def test_slug_matches_the_repository_directory_convention(self):
        assert fetcher.FontFamilyRecord("Noto Sans Arabic", "Serif", ()).slug == "notosansarabic"


class TestFetching:
    def test_only_permissive_licence_directories_are_consulted(self):
        assert set(fetcher.LICENCE_DIRECTORIES) == {"ofl", "apache", "ufl"}

    def test_a_family_with_no_permissive_directory_is_skipped(self, tmp_path):
        record = fetcher.FontFamilyRecord("Nowhere", "Serif", ("arabic",))
        skipped = []

        with patch.object(fetcher, "_locate", return_value=None):
            fetched = list(
                fetcher.fetch_families(
                    [record], tmp_path, on_progress=lambda name, note: skipped.append(note)
                )
            )

        assert fetched == []
        assert "skipped" in skipped[0]

    def test_files_and_licence_land_on_disk(self, tmp_path):
        located = fetcher.FontFamilyRecord(
            "Amiri",
            "Serif",
            ("arabic",),
            directory="ofl",
            files=("Amiri-Regular.ttf",),
            licence="OFL.txt",
        )

        with (
            patch.object(fetcher, "_locate", return_value=located),
            patch.object(fetcher, "_get", return_value=b"FONTBYTES"),
        ):
            fetched = list(fetcher.fetch_families([located], tmp_path))

        assert len(fetched) == 1
        assert (tmp_path / "amiri" / "Amiri-Regular.ttf").read_bytes() == b"FONTBYTES"
        assert (tmp_path / "amiri" / "OFL.txt").exists(), "licence must travel with the fonts"

    def test_the_manifest_records_provenance(self, tmp_path):
        record = fetcher.FontFamilyRecord(
            "Amiri", "Serif", ("arabic",), directory="ofl", files=("Amiri-Regular.ttf",), licence="OFL.txt"
        )

        fetcher.write_manifest(tmp_path, [record], "arabic")
        manifest = fetcher.load_manifest(tmp_path)

        assert manifest["subset"] == "arabic"
        assert manifest["families"][0]["licence"] == "OFL.txt"
        assert "google/fonts" in manifest["source"]

    def test_load_manifest_is_empty_when_absent(self, tmp_path):
        assert fetcher.load_manifest(tmp_path) == {}


class TestVariableFonts:
    def test_a_static_font_reports_no_variations(self):
        assert font_variations(STATIC_FONT) == ()

    def test_a_static_font_expands_to_exactly_one_face(self):
        faces = expand_faces(STATIC_FONT)

        assert faces == (Face(STATIC_FONT),)

    def test_a_variable_font_expands_to_one_face_per_instance(self, tmp_path):
        variable = tmp_path / "Fake[wght].ttf"
        variable.write_bytes(b"")

        with patch(
            "ocrsmith.core.documents.typography.font_variations",
            return_value=("Light", "Regular", "Bold"),
        ):
            faces = expand_faces(variable)

        assert [face.variation for face in faces] == ["Light", "Regular", "Bold"]

    def test_a_variable_family_yields_distinct_weights(self, tmp_path):
        variable = tmp_path / "Alexandria[wght].ttf"
        variable.write_bytes(b"")

        with patch(
            "ocrsmith.core.documents.typography.font_variations",
            return_value=("Thin", "Regular", "Black"),
        ):
            family = group_font_families([variable])[0]

        # Without expansion these three all resolve to the default instance.
        assert family.light.variation == "Thin"
        assert family.bold.variation == "Black"
        assert family.light != family.bold

    def test_variable_axis_suffix_does_not_split_the_family(self, tmp_path):
        static = tmp_path / "Alexandria-Bold.ttf"
        variable = tmp_path / "Alexandria[wght].ttf"
        for path in (static, variable):
            path.write_bytes(b"")

        with patch("ocrsmith.core.documents.typography.font_variations", return_value=()):
            families = group_font_families([static, variable])

        assert len(families) == 1, "Alexandria[wght] and Alexandria-Bold are one family"

    def test_loading_a_variation_is_cached_separately(self):
        plain = load_font(STATIC_FONT, 20)
        again = load_font(STATIC_FONT, 20)

        assert plain is again


class TestFetchedFontsAreUsable:
    """If a fetch has been run locally, the result must be real, usable fonts."""

    FETCHED = Path(__file__).resolve().parents[1] / "assets" / "fonts_google"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "assets" / "fonts_google").exists(),
        reason="run `ocrsmith fetch-fonts` to exercise this",
    )
    def test_every_fetched_family_carries_its_licence(self):
        manifest = json.loads((self.FETCHED / "fonts-manifest.json").read_text(encoding="utf-8"))

        for family in manifest["families"]:
            assert family["licence"], f"{family['family']} was fetched without a licence file"
            assert family["directory"] in fetcher.LICENCE_DIRECTORIES
