"""Contract for the evaluation metrics and harness.

The numbers here are checked against hand-computable cases, because a benchmark whose
metric implementation is itself unverified measures nothing.
"""

import pytest

from ocrsmith.domain import BBox, Table, TableCell
from ocrsmith.evaluation import (
    cer,
    detection_score,
    evaluate,
    levenshtein,
    normalized_edit_similarity,
    table_similarity,
    wer,
)


class TestLevenshtein:
    @pytest.mark.parametrize(
        ("left", "right", "distance"),
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("abc", "abd", 1),
            ("abc", "ab", 1),
            ("ab", "abc", 1),
            ("kitten", "sitting", 3),
        ],
    )
    def test_known_distances(self, left, right, distance):
        assert levenshtein(left, right) == distance

    def test_is_symmetric(self):
        assert levenshtein("abcd", "bcda") == levenshtein("bcda", "abcd")

    def test_works_on_token_sequences(self):
        assert levenshtein(["a", "b"], ["a", "c"]) == 1


class TestCer:
    def test_a_perfect_transcription_scores_zero(self):
        assert cer("مرحبا بالعالم", "مرحبا بالعالم") == 0.0

    def test_one_wrong_character_in_ten(self):
        assert cer("abcdefghij", "abcdefghix") == pytest.approx(0.1)

    def test_an_empty_prediction_scores_one(self):
        assert cer("abcd", "") == 1.0

    def test_two_empty_strings_score_zero(self):
        assert cer("", "") == 0.0

    def test_predicting_text_for_an_empty_reference_scores_one(self):
        assert cer("", "abcd") == 1.0

    def test_whitespace_differences_are_normalised_away(self):
        assert cer("a  b\nc", "a b c") == 0.0

    def test_diacritics_can_be_ignored(self):
        reference = "بِسْمِ اللَّهِ"
        hypothesis = "بسم الله"

        assert cer(reference, hypothesis) > 0
        assert cer(reference, hypothesis, ignore_diacritics=True) == 0.0

    def test_case_can_be_ignored(self):
        assert cer("Hello", "hello", ignore_case=True) == 0.0


class TestWer:
    def test_a_perfect_transcription_scores_zero(self):
        assert wer("one two three", "one two three") == 0.0

    def test_one_wrong_word_in_three(self):
        assert wer("one two three", "one two four") == pytest.approx(1 / 3)

    def test_a_missing_word_counts(self):
        assert wer("one two three", "one two") == pytest.approx(1 / 3)

    def test_an_empty_reference_with_output_scores_one(self):
        assert wer("", "spurious") == 1.0


class TestSimilarity:
    def test_identical_strings_score_one(self):
        assert normalized_edit_similarity("abc", "abc") == 1.0

    def test_unrelated_strings_score_low(self):
        assert normalized_edit_similarity("abcdef", "zyxwvu") < 0.2

    def test_two_empty_strings_score_one(self):
        assert normalized_edit_similarity("", "") == 1.0

    def test_the_score_never_goes_negative(self):
        assert normalized_edit_similarity("a", "b" * 100) >= 0.0


class TestTableSimilarity:
    def _table(self, values, rows=2, cols=2):
        cells = tuple(TableCell(row, col, values[row][col]) for row in range(rows) for col in range(cols))
        return Table(rows, cols, cells, True)

    def test_an_identical_table_scores_one(self):
        table = self._table([["a", "b"], ["c", "d"]])

        assert table_similarity(table, table) == 1.0

    def test_wrong_content_with_the_right_grid_loses_half(self):
        reference = self._table([["a", "b"], ["c", "d"]])
        hypothesis = self._table([["x", "y"], ["z", "w"]])

        assert 0.4 <= table_similarity(reference, hypothesis) <= 0.6

    def test_the_right_content_in_the_wrong_grid_is_penalised(self):
        reference = self._table([["a", "b"], ["c", "d"]])
        flat = Table(1, 4, tuple(TableCell(0, i, t) for i, t in enumerate("abcd")), True)

        assert table_similarity(reference, flat) < 1.0

    def test_a_missing_prediction_scores_zero(self):
        assert table_similarity(self._table([["a", "b"], ["c", "d"]]), None) == 0.0


class TestDetectionScore:
    def test_perfect_prediction(self):
        boxes = [BBox(0, 0, 10, 10), BBox(20, 0, 30, 10)]

        score = detection_score(boxes, boxes)

        assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)

    def test_a_spurious_box_costs_precision(self):
        reference = [BBox(0, 0, 10, 10)]
        predicted = [BBox(0, 0, 10, 10), BBox(50, 50, 60, 60)]

        score = detection_score(reference, predicted)

        assert score.recall == 1.0
        assert score.precision == pytest.approx(0.5)

    def test_a_missed_box_costs_recall(self):
        reference = [BBox(0, 0, 10, 10), BBox(20, 0, 30, 10)]

        score = detection_score(reference, [BBox(0, 0, 10, 10)])

        assert score.precision == 1.0
        assert score.recall == pytest.approx(0.5)

    def test_a_loose_box_fails_the_threshold(self):
        reference = [BBox(0, 0, 10, 10)]
        predicted = [BBox(0, 0, 30, 30)]

        assert detection_score(reference, predicted, iou_threshold=0.5).f1 == 0.0
        assert detection_score(reference, predicted, iou_threshold=0.1).f1 == 1.0

    def test_one_prediction_cannot_match_two_references(self):
        reference = [BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)]

        score = detection_score(reference, [BBox(0, 0, 10, 10)])

        assert score.true_positives == 1
        assert score.false_negatives == 1

    def test_scores_serialise(self):
        data = detection_score([BBox(0, 0, 10, 10)], [BBox(0, 0, 10, 10)]).to_dict()

        assert data["f1"] == 1.0
        assert data["iou_threshold"] == 0.5


class TestHarness:
    @pytest.fixture
    def references(self):
        return {"a": "one two three", "b": "four five six"}

    def test_a_perfect_model_scores_zero_error(self, references):
        report = evaluate(references, dict(references))

        assert report.macro_cer == 0.0
        assert report.mean_similarity == 1.0
        assert report.sample_count == 2

    def test_a_missing_prediction_counts_as_a_total_miss(self, references):
        report = evaluate(references, {"a": "one two three"})

        assert report.missing_predictions == ["b"]
        assert report.macro_cer > 0

    def test_predictions_may_be_dicts(self, references):
        report = evaluate(references, {"a": {"text": "one two three"}, "b": {"text": "four five six"}})

        assert report.macro_cer == 0.0

    def test_micro_and_macro_differ_when_lengths_differ(self):
        references = {"short": "ab", "long": "x" * 200}
        predictions = {"short": "zz", "long": "x" * 200}

        report = evaluate(references, predictions)

        assert report.macro_cer > report.micro_cer

    def test_diacritics_can_be_ignored_across_the_split(self):
        references = {"a": "بِسْمِ اللَّهِ"}

        assert evaluate(references, {"a": "بسم الله"}).macro_cer > 0
        assert evaluate(references, {"a": "بسم الله"}, ignore_diacritics=True).macro_cer == 0.0

    def test_worst_samples_come_first(self):
        report = evaluate({"a": "abcd", "b": "abcd"}, {"a": "abcd", "b": "zzzz"})

        assert report.worst(1)[0].sample_id == "b"

    def test_the_report_serialises_and_renders(self, references):
        report = evaluate(references, dict(references))

        assert report.to_dict()["samples"] == 2
        assert "## Evaluation" in report.to_markdown()
