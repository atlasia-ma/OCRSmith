"""Contract for line breaking.

Measurement is stubbed with a fixed character width so the expectations describe the
algorithm rather than the quirks of whichever FreeType build is installed.
"""

import pytest

from ocrsmith.core.rendering import break_long_word, fit_lines, wrap_paragraph, wrap_text

CHAR = 10.0


def measure(text: str) -> float:
    return len(text) * CHAR


class TestWrapParagraph:
    def test_short_text_stays_on_one_line(self):
        assert wrap_paragraph("ab cd", measure, 100) == ["ab cd"]

    def test_breaks_at_word_boundaries(self):
        assert wrap_paragraph("aaa bbb ccc", measure, 70) == ["aaa bbb", "ccc"]

    def test_never_loses_a_word(self):
        text = "one two three four five six seven"

        lines = wrap_paragraph(text, measure, 60)

        assert " ".join(lines).split() == text.split()

    def test_no_line_exceeds_the_column(self):
        lines = wrap_paragraph("alpha beta gamma delta", measure, 120)

        assert all(measure(line) <= 120 for line in lines)

    def test_none_width_means_do_not_wrap(self):
        assert wrap_paragraph("a b c d e f g", measure, None) == ["a b c d e f g"]

    def test_collapses_runs_of_whitespace(self):
        assert wrap_paragraph("a   b", measure, 1000) == ["a b"]

    def test_empty_paragraph_yields_no_lines(self):
        assert wrap_paragraph("   ", measure, 100) == []


class TestBreakLongWord:
    def test_word_that_fits_is_untouched(self):
        assert break_long_word("abc", measure, 100) == ["abc"]

    def test_oversized_word_is_split_into_fitting_pieces(self):
        pieces = break_long_word("abcdefgh", measure, 30)

        assert pieces == ["abc", "def", "gh"]
        assert "".join(pieces) == "abcdefgh"

    def test_oversized_word_inside_a_paragraph_is_broken_not_overflowed(self):
        lines = wrap_paragraph("hi abcdefgh", measure, 30)

        assert all(measure(line) <= 30 for line in lines)
        assert "".join(lines).replace(" ", "") == "hiabcdefgh"

    def test_zero_width_column_does_not_loop_forever(self):
        assert break_long_word("abc", measure, 0) == ["abc"]


class TestWrapText:
    def test_paragraphs_are_separated_by_a_blank_line(self):
        lines = list(wrap_text(["aa", "bb"], measure, 100))

        assert lines == ["aa", "", "bb"]

    def test_single_paragraph_has_no_leading_separator(self):
        assert list(wrap_text(["aa"], measure, 100)) == ["aa"]


class TestFitLines:
    def test_everything_fits_when_there_is_room(self):
        assert fit_lines(["a", "b"], line_height=10, max_height=100) == (["a", "b"], 0)

    def test_overflowing_lines_are_reported_not_hidden(self):
        kept, dropped = fit_lines(["a", "b", "c"], line_height=10, max_height=25)

        assert kept == ["a", "b"]
        assert dropped == 1

    def test_at_least_one_line_survives_a_tiny_budget(self):
        kept, dropped = fit_lines(["a", "b"], line_height=10, max_height=1)

        assert kept == ["a"]
        assert dropped == 1

    def test_no_height_budget_keeps_everything(self):
        assert fit_lines(["a", "b"], line_height=10, max_height=None) == (["a", "b"], 0)


@pytest.mark.parametrize("width", [30, 55, 80, 200])
def test_wrapping_is_lossless_at_any_column_width(width):
    text = "sed ut perspiciatis unde omnis iste natus error sit voluptatem"

    lines = wrap_paragraph(text, measure, width)

    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
