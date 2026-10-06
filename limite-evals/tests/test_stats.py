"""The bootstrap, and the reason it resamples problems rather than only rollouts."""

from __future__ import annotations

from limite_evals_core.stats import (
    bootstrap_interval,
    mean_of_group,
    mean_of_groups,
    pass_all_k,
    pass_any_k,
)


def test_avg_at_k_averages_problems_not_rollouts() -> None:
    """One problem with many rollouts must not outweigh one with few."""
    groups = [[True] * 32, [False]]
    assert mean_of_groups(groups) == 0.5


def test_interval_brackets_the_point_estimate() -> None:
    groups = [[True, False, True, False]] * 30
    value, low, high = bootstrap_interval(groups, resamples=500)
    assert low <= value <= high


def test_a_unanimous_result_has_a_degenerate_interval() -> None:
    value, low, high = bootstrap_interval([[True]] * 30, resamples=500)
    assert (value, low, high) == (1.0, 1.0, 1.0)


def test_thirty_problems_gives_a_wide_interval() -> None:
    """The AIME case that motivates making intervals mandatory.

    Thirty problems at avg@32 with a genuinely mixed model leaves an interval
    several points wide, which is why a bare pre-to-post AIME delta cannot be
    told apart from noise.
    """
    groups = [[True] * 32 if i % 3 == 0 else [False] * 32 for i in range(30)]
    _, low, high = bootstrap_interval(groups, resamples=2000)
    assert high - low > 0.2


def test_problem_resampling_widens_the_interval_over_rollouts_alone() -> None:
    """Between-problem variance dominates; ignoring it understates the interval."""
    groups = [[True] * 8 if i % 2 == 0 else [False] * 8 for i in range(20)]
    _, low, high = bootstrap_interval(groups, resamples=2000)
    # Every group is unanimous, so resampling rollouts alone would give a zero-width
    # interval. The hierarchical version must not.
    assert high - low > 0.1


def test_the_interval_is_reproducible() -> None:
    groups = [[True, False]] * 20
    assert bootstrap_interval(groups, resamples=300) == bootstrap_interval(groups, resamples=300)


def test_a_single_problem_cannot_be_bootstrapped() -> None:
    assert bootstrap_interval([[True, False]]) == (0.5, 0.5, 0.5)


def test_empty_groups_are_dropped_not_counted_as_zero() -> None:
    assert mean_of_groups([[True], [], [True]]) == 1.0


# --- the k estimators --------------------------------------------------------


def test_pass_at_k_is_the_unbiased_estimator_not_the_observed_share() -> None:
    """One correct rollout in four: a draw of two holds it in three cases of six.

    `1 - C(3,2)/C(4,2) = 1 - 3/6`. The observed share of correct rollouts is a
    quarter, so a column reporting that would be answering a different question.
    """
    assert pass_any_k(2)([True, False, False, False]) == 0.5


def test_pass_hat_k_is_the_chance_every_draw_is_right() -> None:
    """Two correct in four: exactly one of the six pairs is correct throughout."""
    assert pass_all_k(2)([True, True, False, False]) == 1 / 6


def test_nothing_correct_scores_zero_at_both_estimators() -> None:
    """The c=0 boundary: `C(n-0,k) = C(n,k)`, and no draw of k is all-correct."""
    assert pass_any_k(2)([False] * 4) == 0.0
    assert pass_all_k(2)([False] * 4) == 0.0


def test_everything_correct_scores_one_at_both_estimators() -> None:
    """The c=n boundary: `C(0,k) = 0`, and every draw of k is all-correct."""
    assert pass_any_k(2)([True] * 4) == 1.0
    assert pass_all_k(2)([True] * 4) == 1.0


def test_at_k_equal_to_n_there_is_one_draw_and_it_is_the_whole_group() -> None:
    """The k=n boundary, where the two estimators pull apart the furthest.

    One draw exists, so `pass@n` asks whether the group holds a correct rollout
    at all and `pass^n` whether every one of them is correct.
    """
    assert pass_any_k(4)([True, False, False, False]) == 1.0
    assert pass_all_k(4)([True, False, False, False]) == 0.0


def test_pass_at_one_is_the_avg_at_k_term_to_the_last_bit() -> None:
    """The property RES-396 verifies, at the level it is produced.

    Not `approx`: `1 - (n-c)/n` and `c/n` differ in the last place at n=3, and a
    report carrying both would show two headline figures that disagree with
    nothing to say why.
    """
    for group in ([True, False, False], [True] * 4, [False] * 4, [1.0, 0.0, 1.0, 0.0]):
        assert pass_any_k(1)(group) == mean_of_group(group)


def test_a_group_too_small_for_k_answers_nothing_rather_than_zero() -> None:
    """D-107 one level down: pass@4 over a group of two is not a low number.

    `None` is what keeps it out of the mean at both levels. Zero would report a
    checkpoint that failed a question nobody asked it.
    """
    assert pass_any_k(4)([True, True]) is None
    assert pass_all_k(4)([True, True]) is None
    assert mean_of_group([]) is None


def test_a_problem_that_cannot_answer_is_skipped_by_the_whole_bootstrap() -> None:
    """The point estimate is over the problems that drew enough rollouts.

    The ragged problem here would drag `pass@4` to a half if it were counted as
    a failure, and it says nothing about a draw of four either way.
    """
    groups = [[True, True], [True] * 4, [True] * 4]
    assert bootstrap_interval(groups, statistic=pass_any_k(4), resamples=300)[0] == 1.0


def test_naming_the_default_statistic_changes_no_interval() -> None:
    """The statistic is a parameter; the resampling scheme it is handed is not."""
    groups = [[True, False, True, False]] * 12
    assert bootstrap_interval(groups, resamples=300) == bootstrap_interval(
        groups, statistic=mean_of_group, resamples=300
    )


def test_the_k_columns_are_bootstrapped_by_the_same_two_level_draw() -> None:
    """pass@1 must reproduce avg@k's whole interval, not only its point estimate.

    Same seed, same groups, same resample sequence, and the same float out of
    each problem: what a draw is reduced to is all that differs, and at k=1 that
    is the same number.
    """
    groups = [[True, False, False], [True] * 4, [False] * 4, [True, False, True, False]]
    assert bootstrap_interval(groups, statistic=pass_any_k(1), resamples=300) == (
        bootstrap_interval(groups, resamples=300)
    )
