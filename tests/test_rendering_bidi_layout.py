"""Contract for word-level bidirectional ordering.

The indices returned describe *drawing* order, left to right on screen. The annotation
keeps logical order, so these two orders must be able to disagree without either being
wrong.
"""

from ocrsmith.core.rendering import visual_word_order
from ocrsmith.text import Direction


class TestLeftToRight:
    def test_latin_keeps_its_order(self):
        assert visual_word_order(["the", "quick", "fox"], Direction.LTR) == (0, 1, 2)

    def test_embedded_arabic_run_is_reversed_in_place(self):
        words = ["say", "مرحبا", "بالعالم", "now"]

        assert visual_word_order(words, Direction.LTR) == (0, 2, 1, 3)


class TestRightToLeft:
    def test_arabic_is_drawn_last_word_first(self):
        assert visual_word_order(["مرحبا", "بالعالم"], Direction.RTL) == (1, 0)

    def test_embedded_latin_run_keeps_internal_order(self):
        # "سنة 2024 OCR Smith" — the Latin phrase must not come out backwards.
        words = ["سنة", "OCR", "Smith"]

        order = visual_word_order(words, Direction.RTL)

        assert order.index(1) < order.index(2)
        assert order[-1] == 0  # the Arabic word sits rightmost

    def test_neutral_words_follow_their_neighbours(self):
        words = ["مرحبا", "2024", "بالعالم"]

        assert visual_word_order(words, Direction.RTL) == (2, 1, 0)


class TestEdgeCases:
    def test_empty_input(self):
        assert visual_word_order([], Direction.RTL) == ()

    def test_single_word(self):
        assert visual_word_order(["مرحبا"], Direction.RTL) == (0,)

    def test_every_index_appears_exactly_once(self):
        words = ["مرحبا", "OCR", "2024", "بالعالم", "Smith"]

        for base in (Direction.LTR, Direction.RTL):
            order = visual_word_order(words, base)
            assert sorted(order) == list(range(len(words)))

    def test_all_neutral_words_keep_source_order_when_ltr(self):
        assert visual_word_order(["12", "34"], Direction.LTR) == (0, 1)
