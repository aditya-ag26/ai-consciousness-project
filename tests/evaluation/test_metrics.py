import pytest

from src.evaluation.metrics import (
    classify,
    format_table,
    is_hit,
    mean,
    percentile,
    reciprocal_rank,
)


def test_classify_counts_every_quadrant():
    metrics = classify(
        answered=[True, False, True, False],
        should_answer=[True, True, False, False],
    )

    assert (metrics.true_positives, metrics.false_negatives) == (1, 1)
    assert (metrics.false_positives, metrics.true_negatives) == (1, 1)
    assert metrics.total == 4


def test_perfect_classifier_scores_one():
    metrics = classify(answered=[True, False], should_answer=[True, False])

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_classifier_that_answers_everything_has_full_recall_but_poor_precision():
    metrics = classify(answered=[True, True], should_answer=[True, False])

    assert metrics.recall == 1.0
    assert metrics.precision == 0.5


def test_f1_is_zero_when_nothing_is_answered():
    metrics = classify(answered=[False, False], should_answer=[True, False])

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_classify_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        classify(answered=[True], should_answer=[True, False])


def test_is_hit_matches_a_substring_case_insensitively():
    titles = ["An algebraic theory to discriminate Qualia in the brain"]

    assert is_hit(titles, ["qualia"]) is True
    assert is_hit(titles, ["global workspace"]) is False


def test_is_hit_accepts_any_expected_source():
    assert is_hit(["Detecting Qualia"], ["nothing", "qualia"]) is True


def test_reciprocal_rank_rewards_earlier_positions():
    titles = ["Irrelevant paper", "Qualia and meaning", "Another"]

    assert reciprocal_rank(titles, ["qualia"]) == pytest.approx(0.5)
    assert reciprocal_rank(["Qualia first"], ["qualia"]) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_matches():
    assert reciprocal_rank(["a", "b"], ["qualia"]) == 0.0


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_percentile_returns_observed_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 5.0
    assert percentile([], 0.5) == 0.0


def test_format_table_aligns_columns():
    table = format_table(["a", "bb"], [["1", "2"]])

    assert table.splitlines()[0] == "a | bb"
