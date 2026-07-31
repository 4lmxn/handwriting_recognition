from pathlib import Path

from training.confusion_matrix import (
    ConfusionMatrix,
    align_strings,
    analyze_ambiguous_classes,
    hard_negative_classes,
)


def test_align_strings_identical():
    assert align_strings("cat", "cat") == [("c", "c"), ("a", "a"), ("t", "t")]


def test_align_strings_pure_substitution():
    assert align_strings("cat", "bat") == [("c", "b"), ("a", "a"), ("t", "t")]


def test_align_strings_pure_insertion():
    assert align_strings("cat", "cats") == [("c", "c"), ("a", "a"), ("t", "t"), (None, "s")]


def test_align_strings_pure_deletion():
    assert align_strings("cats", "cat") == [("c", "c"), ("a", "a"), ("t", "t"), ("s", None)]


def test_align_strings_mixed():
    # "ab" -> "b": deletion of 'a', match of 'b'
    assert align_strings("ab", "b") == [("a", None), ("b", "b")]


def test_align_strings_empty_reference():
    assert align_strings("", "ab") == [(None, "a"), (None, "b")]


def test_align_strings_empty_hypothesis():
    assert align_strings("ab", "") == [("a", None), ("b", None)]


def test_align_strings_both_empty():
    assert align_strings("", "") == []


def test_confusion_matrix_record_and_most_confused():
    matrix = ConfusionMatrix()
    matrix.record("0000", "OOOO")  # 4x 0->O
    matrix.record("11", "ll")  # 2x 1->l
    matrix.record("cat", "cat")  # no confusions

    ranking = matrix.most_confused(top_n=2)
    assert ranking[0] == (("0", "O"), 4)
    assert ranking[1] == (("1", "l"), 2)


def test_confusion_matrix_record_ignores_insertions_and_deletions():
    matrix = ConfusionMatrix()
    matrix.record("cat", "cats")  # pure insertion
    matrix.record("cats", "cat")  # pure deletion

    assert dict(matrix.counts) == {}


def test_confusion_matrix_to_dict_and_json_round_trip(tmp_path: Path):
    matrix = ConfusionMatrix()
    matrix.record("0", "O")
    matrix.record("0", "O")
    matrix.record("5", "S")

    as_dict = matrix.to_dict()
    assert as_dict["total_substitutions"] == 3
    assert as_dict["confusions"]["0->O"] == 2
    assert as_dict["confusions"]["5->S"] == 1

    path = tmp_path / "confusions.json"
    matrix.save_json(path)
    loaded = ConfusionMatrix.load_json(path)

    assert loaded.counts == matrix.counts
    assert loaded.to_dict() == matrix.to_dict()


def test_analyze_ambiguous_classes_counts_both_directions():
    matrix = ConfusionMatrix()
    matrix.record("0", "O")
    matrix.record("0", "O")
    matrix.record("O", "0")  # opposite direction, same pair
    matrix.record("1", "l")
    matrix.record("1", "I")

    report = analyze_ambiguous_classes(matrix)

    assert report["0/O"] == 3
    assert report["1/l"] == 1
    assert report["1/I"] == 1
    # unrelated ambiguous classes should be present with zero count
    assert report["5/S"] == 0
    assert report["O/Q"] == 0
    assert report["C/G"] == 0
    assert set(report.keys()) == {
        "0/O",
        "1/l",
        "1/I",
        "5/S",
        "2/Z",
        "8/B",
        "6/G",
        "9/g",
        "O/Q",
        "C/G",
    }


def test_hard_negative_classes_threshold_boundary():
    matrix = ConfusionMatrix()
    for _ in range(3):
        matrix.record("0", "O")  # count 3, meets default min_count=3
    for _ in range(2):
        matrix.record("5", "S")  # count 2, below default min_count=3

    result = hard_negative_classes(matrix, min_count=3)
    assert result == {"0", "O"}

    result_lower_threshold = hard_negative_classes(matrix, min_count=2)
    assert result_lower_threshold == {"0", "O", "5", "S"}

    result_higher_threshold = hard_negative_classes(matrix, min_count=4)
    assert result_higher_threshold == set()
