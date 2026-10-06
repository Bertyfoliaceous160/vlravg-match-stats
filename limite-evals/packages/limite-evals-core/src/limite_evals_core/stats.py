"""Hierarchical bootstrap confidence intervals for the numbers a run reports.

Two levels of sampling produced the number, so two levels resample it: problems
with replacement, then that problem's rollouts with replacement. Resampling only
rollouts understates the interval badly on the small sets that matter most --
AIME24 and AIME25 have 30 problems each, where problem sampling dominates the
variance and a 32-rollout group does almost nothing to reduce it.

This is why the contract makes intervals mandatory rather than optional: a
pre-to-post AIME delta quoted without one cannot be told apart from noise.

**What is resampled and what is read off one problem are two questions, and only
the second is a parameter.** `bootstrap_interval` draws the two levels itself
and asks a `statistic` what each resampled problem says -- the mean of its
rollouts, which is avg@k, unless another is named. The mean over problems is the
bootstrap's own and is not reachable from outside. The other way to add pass@k
would have been a second bootstrap function per estimator, which is how a
resampling scheme comes to differ between two columns of one table without
anybody deciding that it should.

A statistic answering per problem rather than per resample is what makes the
addition free of a re-baseline. The outer mean is still accumulated exactly as
it always was -- `total += ...` over the problems of a draw, which is not
bit-for-bit what `sum()` does, since CPython compensates a float sum and a `+=`
loop does not -- so an interval already reported recomputes to the same float.
Every estimator here is a per-problem quantity anyway: pass@k and pass^k are
statements about one problem's rollouts, averaged over problems exactly as
avg@k is.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from math import comb

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95

#: What one problem's rollouts say, or `None` where they say nothing this
#: statistic can use. Verdicts arrive as bools or as the floats a declared metric
#: records, and every estimator here reads both, since `True` and `1.0` are the
#: same rollout written down twice.
#:
#: `None` is how a problem leaves a column without leaving the run, and it is the
#: same rule at both levels: an empty group has no mean, and a group of four has
#: no pass@32. Neither is a zero, and the bootstrap skips them rather than
#: averaging an absence in (D-107).
Statistic = Callable[[Sequence[float]], float | None]


def mean_of_group(group: Sequence[float]) -> float | None:
    """One problem's share of correct rollouts: avg@k's per-problem term."""
    return sum(group) / len(group) if group else None


def mean_of_groups(groups: Sequence[Sequence[bool]]) -> float:
    """Mean over problems of the mean over each problem's rollouts (avg@k)."""
    return _over_problems(groups, mean_of_group)


def pass_any_k(k: int) -> Statistic:
    """pass@k: the chance that a draw of `k` of a problem's rollouts holds a right one.

    The unbiased estimator rather than the empirical one. With `n` rollouts of
    which `c` are correct, drawing `k` without replacement misses every correct
    rollout in `C(n-c, k)` of the `C(n, k)` draws, so `1 - C(n-c, k) / C(n, k)`
    is what a group of `n` says about a group of `k`. Sampling `k` rollouts and
    looking at them instead would estimate the same quantity from a fraction of
    the evidence, and on the 30-problem competition sets that is the whole of the
    variance the intervals here exist to report.

    Written as one division of two integers -- `(C(n,k) - C(n-c,k)) / C(n,k)` --
    and that is not a tidying. At `k = 1` it reduces to exactly `c / n`, the same
    float `mean_of_group` reaches on the same rollouts, so `pass@1` **is** the
    avg@k column rather than a number that rounds to it. `1 - (n-c)/n` is not:
    at `n = 3, c = 1` it is 0.33333333333333337 against 0.3333333333333333, and
    a report carrying both would show two headline figures differing in the last
    place with nothing to say why.

    A problem with fewer than `k` rollouts answers `None` rather than zero:
    `pass@32` over a group of 4 is not a low number, it is not a number. Which
    columns exist at all is decided one level up, from the group size, so that
    the absence arrives with its reason.
    """

    def statistic(group: Sequence[float]) -> float | None:
        if len(group) < k:
            return None
        n, correct = len(group), _correct(group)
        return (comb(n, k) - comb(n - correct, k)) / comb(n, k)

    return statistic


def pass_all_k(k: int) -> Statistic:
    """pass^k: the chance that a draw of `k` of a problem's rollouts is right throughout.

    `C(c, k) / C(n, k)`, the empirical probability that every one of `k` rollouts
    drawn without replacement is correct. It is the quantity pass@k cannot show:
    a checkpoint that answers a problem once in four has a `pass@32` of 1 and a
    `pass^4` of nothing, and the distance between the two columns is how much of
    a benchmark number a single sample would actually reproduce.

    The same rule for a group too small to ask, and for the same reason.
    """

    def statistic(group: Sequence[float]) -> float | None:
        if len(group) < k:
            return None
        return comb(_correct(group), k) / comb(len(group), k)

    return statistic


def _correct(group: Sequence[float]) -> int:
    """How many of a problem's rollouts were right, counting `1.0` as `True`."""
    return sum(1 for value in group if value)


def _over_problems(groups: Sequence[Sequence[float]], statistic: Statistic) -> float:
    """The point estimate: `statistic` over the problems that can answer it.

    Compensated by `sum`, exactly as `mean_of_groups` always was. The draw loop
    accumulates the same mean with `+=` and the two disagree in the last bit on a
    ragged group whose size is not a power of two; both are left as they were,
    because the object here is to add columns and move nothing.
    """
    values = [value for group in groups if (value := statistic(group)) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def bootstrap_interval(
    groups: Sequence[Sequence[bool]],
    *,
    statistic: Statistic = mean_of_group,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float, float]:
    """`(value, low, high)` for `statistic` over `groups`, one group per problem.

    The seed is fixed by default so a summary is reproducible from its samples;
    an interval that moves when you recompute it from the same data is not
    evidence of anything.

    `statistic` reads one problem's resampled rollouts and nothing else. The
    two-level draw and the mean over problems are the bootstrap's own, and the
    random numbers are consumed in the order they always were, so naming a
    statistic cannot move an interval that was already reported.

    A problem the statistic cannot answer is skipped in a draw exactly as it is
    in the point estimate, so a `pass@32` column over a suite with one ragged
    problem is a mean over the problems that drew 32 rollouts at both levels.
    """
    point = _over_problems(groups, statistic)
    scored = [list(group) for group in groups if group]
    if len(scored) < 2:
        return point, point, point

    rng = random.Random(seed)
    n = len(scored)
    draws = []
    for _ in range(resamples):
        total = 0.0
        counted = 0
        for _ in range(n):
            group = scored[rng.randrange(n)]
            k = len(group)
            value = statistic([group[rng.randrange(k)] for _ in range(k)])
            if value is None:
                continue
            total += value
            counted += 1
        draws.append(total / counted if counted else 0.0)

    draws.sort()
    tail = (1.0 - confidence) / 2.0
    low = draws[min(int(tail * resamples), resamples - 1)]
    high = draws[min(int((1.0 - tail) * resamples), resamples - 1)]
    return point, low, high
