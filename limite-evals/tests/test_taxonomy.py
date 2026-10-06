"""The A--T taxonomy: one fixture per case, checked against the real rungs.

Each fixture asserts two things, and the second is the one that matters. That a
completion classifies to the case it was written for only says the classifier
reads its own rules. That `ladder.score` then does to it what the frozen table
said it would is the claim the tracker actually makes -- the table is a
prediction about the graders, and every row of it is executed here against the
graders themselves rather than against a description of them.

The fixtures are written so the answer they carry is **right** wherever the case
name does not say otherwise, because the table's "yes" is what a case scores
when the answer it holds is right. `N` is the exception the table itself names.
`S` carries no answer at all, which is what `S` is.

Every fixture is scored `boxed`, deliberately: the table's strict column is
about where a `\\boxed{}` answer sits and predicts nothing on a taskset whose
reference reads something else. That scoping is asserted separately, on a
hashed taskset, at the bottom of this file.
"""

from __future__ import annotations

import pytest

from limite_evals import taxonomy
from limite_evals.aggregate import case_counts, case_reconciliation, failure_case, summarize_task
from limite_evals.report import _expected
from limite_evals_core.ladder import score
from limite_evals_core.schema import (
    PROTOCOL_METRICS,
    SCORER_VERSION,
    ProtocolSample,
    SampleResult,
)

GOLD = "42"

#: One completion per case: `(case, completion, gold, truncated)`. Ordered as the
#: frozen table is -- the cases strict can accept, then the ones only permissive
#: can, then the one neither can.
FIXTURES: list[tuple[taxonomy.Case, str, str, bool]] = [
    ("A", "<think>work</think>\nSo the answer is \\boxed{42}.", GOLD, False),
    ("M", "<think>work</think>\nMaybe \\boxed{7}, no: \\boxed{42}.", GOLD, False),
    # The gold is the bare number and the box writes it with a separator, which
    # is the whole of what `Q` claims: the rung normalises the comma away.
    ("Q", "<think>work</think>\nSo the answer is \\boxed{1,000}.", "1000", False),
    ("R", "<think>work</think>\nSo the answer is \\boxed{42}.", "\\boxed{42}", False),
    ("B", "<think>work, and I get \\boxed{42} here", GOLD, False),
    ("C", "<think>work</think>\nSo the answer is \\boxed{42", GOLD, True),
    ("D", "<think>work, and I get 42, so \\boxed{42", GOLD, True),
    # The same shape as `C` on a completion that ended by itself: unbalanced
    # braces rather than a cap, which is the only thing separating the two.
    ("E", "<think>work</think>\nSo the answer is \\boxed{\\frac{42}{1}.", GOLD, False),
    ("F", "<think>work</think>\nI computed 42 but wrote \\boxed{}.", GOLD, False),
    ("G", "<think>work</think>\nSo the answer is \\fbox{42}.", GOLD, False),
    ("H", "<think>work</think>\nThe answer is 42", GOLD, False),
    ("I", "<think>work</think>\nSome prose.\nTherefore $42$", GOLD, False),
    ("J", "<think>work</think>\nSome prose.\n42", GOLD, False),
    ("L", "<think>work: carrying the 42 across and then", GOLD, False),
    ("K", "<think>it comes to 42</think>\nI cannot finish this one.", GOLD, False),
    ("N", "<think>work</think>\nMaybe \\boxed{42}, no: \\boxed{7}.", GOLD, False),
    ("O", "<think>work</think>\nThe answer is 42.\nHope that helps!", GOLD, False),
    ("P", "<think>work</think>\nWe had 42 apples at that point, but I am stuck.\nUnresolved.", GOLD, False),
    ("T", "<think>so \\boxed{42}</think>\nI give up on this one.", GOLD, False),
    ("S", "<think>work</think>\nI have no idea how to approach this.", GOLD, False),
]


def test_every_case_has_a_fixture() -> None:
    """The suite covers the table, not a subset of it that happens to pass."""
    assert {case for case, *_ in FIXTURES} == set(taxonomy.CASES)


@pytest.mark.parametrize(("case", "completion", "gold", "truncated"), FIXTURES, ids=lambda v: v)
def test_fixture_classifies_to_its_case(
    case: taxonomy.Case, completion: str, gold: str, truncated: bool
) -> None:
    assert taxonomy.classify(completion, gold, truncated=truncated) == case


@pytest.mark.parametrize(("case", "completion", "gold", "truncated"), FIXTURES, ids=lambda v: v)
def test_fixture_scores_what_the_table_predicts(
    case: taxonomy.Case, completion: str, gold: str, truncated: bool
) -> None:
    """The real rungs, on the real fixture, against the frozen table's row.

    `ladder.score` is executed rather than described. A change to the extraction
    that moved one of these rows would be a change to what the taxonomy claims,
    and it fails here rather than surfacing as a reconciliation gap on a run.
    """
    result = score(completion, gold, answer_format="boxed")
    assert result.strict is taxonomy.expects(case, "strict")
    assert result.permissive is taxonomy.expects(case, "permissive")


def test_classification_is_total_and_deterministic() -> None:
    """Every completion reaches a case, and reaches the same one twice.

    Run over the fixtures crossed with both truncation flags, which is the one
    input outside the text that the order reads.
    """
    for _, completion, gold, _ in FIXTURES:
        for truncated in (True, False):
            first = taxonomy.classify(completion, gold, truncated=truncated)
            assert first in taxonomy.CASES
            assert taxonomy.classify(completion, gold, truncated=truncated) == first


def test_the_strict_column_is_exactly_the_four_cases() -> None:
    """The table's two derived sets, read back off the table."""
    assert taxonomy.STRICT_YES == {"A", "M", "Q", "R"}
    assert set(taxonomy.CASES) - taxonomy.PERMISSIVE_YES == {"S"}


def test_expects_refuses_a_rung_the_table_does_not_state() -> None:
    """`lenient` is a real rung and the table says nothing about it, so asking is
    an error rather than a default."""
    with pytest.raises(ValueError, match="lenient"):
        taxonomy.expects("A", "lenient")


def _sample(case: taxonomy.Case, completion: str, gold: str, truncated: bool, index: int) -> SampleResult:
    """One fixture as a scored row, exactly as the runner would have written it."""
    result = score(completion, gold, answer_format="boxed")
    return SampleResult(
        task="math500",
        problem_index=index,
        rollout_index=0,
        gold=gold,
        completion=completion,
        finish_reason="length" if truncated else "stop",
        truncated=truncated,
        # A protocol run's rows carry these; `failure_case` is only derived for a
        # row a grader scored, so a bare row would be skipped by aggregation.
        protocols={"reference": ProtocolSample(protocol="reference", correct=result.strict)},
        metrics={"reference": float(result.strict)},
        strict=result.strict,
        lenient=result.lenient,
        permissive=result.permissive,
        format_ok=result.format_ok,
        failure_case=case,
    )


FIXTURE_SAMPLES = [
    _sample(case, completion, gold, truncated, index)
    for index, (case, completion, gold, truncated) in enumerate(FIXTURES)
]


def _boxed_task() -> object:
    return summarize_task(
        "math500",
        FIXTURE_SAMPLES,
        metric_set=PROTOCOL_METRICS,
        answer_format="boxed",
        resamples=8,
    )


def test_case_counts_partition_the_rollouts() -> None:
    """One rollout per case, in the table's order, and nothing lost or doubled."""
    counts = case_counts(FIXTURE_SAMPLES, answer_format="boxed")
    assert [count.case for count in counts] == list(taxonomy.CASES)
    assert sum(count.samples for count in counts) == len(FIXTURE_SAMPLES)


def test_case_counts_reconcile_with_both_lenses() -> None:
    """The claim, on the fixtures: neither lens scored outside the cases allowed it."""
    task = _boxed_task()
    for rung in ("strict", "permissive"):
        total, predicted, disagreeing = case_reconciliation(task, rung)
        assert disagreeing == 0
        assert total == predicted == len(taxonomy.STRICT_YES if rung == "strict" else taxonomy.PERMISSIVE_YES)


def test_a_row_without_a_stored_case_is_classified_from_its_text() -> None:
    """A sample file written before the tracker existed still partitions.

    This is what lets `limite-rescore` apply the taxonomy to a finished run at no
    generation cost, so it is asserted rather than assumed.
    """
    stored = case_counts(FIXTURE_SAMPLES, answer_format="boxed")
    derived = case_counts(
        [s.model_copy(update={"failure_case": None}) for s in FIXTURE_SAMPLES],
        answer_format="boxed",
    )
    assert [(c.case, c.samples) for c in derived] == [(c.case, c.samples) for c in stored]


def test_a_declared_metric_wins_over_the_field_of_the_same_name() -> None:
    """On a protocol run, `permissive` here is the protocol, not the ladder rung.

    `scored` reads a taskset's own declared metric before any attribute sharing
    its name, and routing the case counts through it is what makes this table
    reconcile against the columns the report prints rather than against a second
    reading of the same rollouts. The two artefacts do differ -- pinned here so
    the choice stays deliberate.
    """
    row = FIXTURE_SAMPLES[0].model_copy(
        update={"permissive": False, "metrics": {"reference": 1.0, "permissive": 1.0}}
    )
    assert case_counts([row])[0].permissive_scored == 1


def test_a_constraint_checked_row_gets_no_case() -> None:
    """No answer to extract is not the same fact as no answer found.

    `S` on a row about comma usage would read as a measured shape, and the
    taskset's empty `protocols` is how every other surface here already tells
    the two apart.
    """
    row = SampleResult(
        task="native-task",
        problem_index=0,
        rollout_index=0,
        gold="",
        completion="Here is a paragraph with no commas in it whatsoever.",
        metrics={"prompt_accuracy": 1.0},
    )
    assert case_counts([row]) == []


def test_the_tracker_changes_no_score() -> None:
    """The observe-only property, asserted on the fixtures rather than argued.

    Summarising the same rollouts with the taxonomy's field cleared has to leave
    every reported number identical: the partition is beside the metrics and
    never inside them.
    """
    with_cases = _boxed_task()
    without = summarize_task(
        "math500",
        [s.model_copy(update={"failure_case": None}) for s in FIXTURE_SAMPLES],
        metric_set=PROTOCOL_METRICS,
        answer_format="boxed",
        resamples=8,
    )
    assert with_cases.metrics == without.metrics


# --- The strict column's domain -------------------------------------------
#
# Historical GSM8K rows asked for `#### 42` and fell back to the whole
# completion, so their strict score can come from any shape at all. The fixtures
# below protect taxonomy support for those stored rows without making the
# retired format selectable by a current taskset.

HASHED_ROLLOUTS = [
    # `O`: the phrase is not on the last line, so nothing boxed or last-line
    # reaches the answer -- but the `####` marker does.
    ("O", "The answer is 42.\n#### 42\nHope that helps!"),
    # `P`: a number in the working and no marker at all, so the reference
    # compares the whole completion and math-verify finds the answer in it.
    ("P", "We had 42 apples at that point, but I am stuck.\nUnresolved."),
]


def _hashed_sample(index: int, case: str, completion: str) -> SampleResult:
    result = score(completion, GOLD, answer_format="hashed")
    return SampleResult(
        task="gsm8k",
        problem_index=index,
        rollout_index=0,
        gold=GOLD,
        completion=completion,
        protocols={"reference": ProtocolSample(protocol="reference", correct=result.strict)},
        metrics={"reference": float(result.strict)},
        strict=result.strict,
        permissive=result.permissive,
        format_ok=result.format_ok,
        failure_case=case,
    )


HASHED_SAMPLES = [
    _hashed_sample(index, case, completion)
    for index, (case, completion) in enumerate(HASHED_ROLLOUTS)
]


def test_the_hashed_fixtures_do_score_strict_from_a_forbidden_case() -> None:
    """The premise. Without it the scoping test below would pass vacuously."""
    assert [s.failure_case for s in HASHED_SAMPLES] == ["O", "P"]
    assert all(s.strict for s in HASHED_SAMPLES)
    assert all(s.failure_case not in taxonomy.STRICT_YES for s in HASHED_SAMPLES)


def test_a_hashed_taskset_is_counted_but_carries_no_strict_prediction() -> None:
    """Counts and permissive expectations survive; the strict column states nothing."""
    counts = case_counts(HASHED_SAMPLES, answer_format="hashed")
    assert [(c.case, c.samples) for c in counts] == [("O", 1), ("P", 1)]
    assert all(count.strict_expected is None for count in counts)
    assert all(count.permissive_expected is True for count in counts)
    # The observation is still recorded -- only the prediction is withheld.
    assert sum(count.strict_scored for count in counts) == 2


def test_a_hashed_taskset_reports_no_strict_violation() -> None:
    """The resolution, end to end: 132 findings on EV0 become none.

    `None` rather than `0` for both the prediction and the disagreement, because
    a zero would read as "predicted, and nothing disagreed". The count of what
    the lens scored stays a number, since that is an observation either way.
    """
    task = summarize_task(
        "gsm8k",
        HASHED_SAMPLES,
        metric_set=PROTOCOL_METRICS,
        answer_format="hashed",
        resamples=8,
    )
    assert task.answer_format == "hashed"
    assert case_reconciliation(task, "strict") == (2, None, None)
    # Permissive is not scoped and still reconciles.
    _, predicted, disagreeing = case_reconciliation(task, "permissive")
    assert disagreeing == 0
    assert predicted is not None


def test_an_undeclared_answer_format_states_no_strict_prediction() -> None:
    """A caller that did not say which reference this taskset answers to gets no
    claim, rather than the boxed one asserted over a taskset nobody declared."""
    assert all(count.strict_expected is None for count in case_counts(FIXTURE_SAMPLES))


def test_a_withheld_column_is_not_rendered_as_a_refusal() -> None:
    """`S` on a hashed taskset expects no `permissive` and *says nothing* about
    `strict`. A bare "neither" would refuse on the taxonomy's behalf, which is
    the `None`-into-`False` collapse the scoping exists to prevent.
    """
    hashed_s = case_counts(
        [_hashed_sample(0, "S", "I have no idea how to approach this.")],
        answer_format="hashed",
    )[0]
    boxed_s = next(c for c in case_counts(FIXTURE_SAMPLES, answer_format="boxed") if c.case == "S")
    assert _expected(hashed_s) == "not `permissive`"
    assert _expected(boxed_s) == "neither"
    # And a case the table does expect something of still names only that.
    hashed_a = case_counts(HASHED_SAMPLES, answer_format="hashed")[0]
    assert _expected(hashed_a) == "`permissive`"


def test_the_scoping_reaches_only_strict() -> None:
    """Read off the two functions rather than off a report, so it cannot drift."""
    assert taxonomy.expectation("A", "strict", "boxed") is True
    assert taxonomy.expectation("A", "strict", "hashed") is None
    assert taxonomy.expectation("C", "strict", "boxed") is False
    assert taxonomy.expectation("C", "strict", "hashed") is None
    for answer_format in ("boxed", "hashed", None):
        assert taxonomy.expectation("A", "permissive", answer_format) is True
        assert taxonomy.expectation("S", "permissive", answer_format) is False


# --- the stored case letter is a cache, and caches carry versions ------------


def test_a_case_letter_from_another_scorer_is_recomputed_rather_than_believed() -> None:
    """The footgun this closes, stated as the wrong answer it used to give.

    A stored letter is written once by the scoring path and read back by
    aggregation, which makes it a cache -- and an unversioned cache keeps
    answering after the function behind it has changed. That is not
    hypothetical here: when the reasoning delimiters began surviving the decode,
    five cases that could never fire started firing, and every letter already on
    disk went on describing text nobody would classify that way now.

    The row below carries a deliberate lie, `S`, beside a completion that is
    plainly `A`. Stamped with this scorer the lie is trusted, because within one
    scorer the stored classification and the reported one must not be able to
    disagree. Stamped with any other, or unstamped, the text wins.
    """
    honest = _sample("A", "<think>work</think>\nSo the answer is \\boxed{42}.", GOLD, False, 0)
    lying = honest.model_copy(update={"failure_case": "S"})

    trusted = lying.model_copy(update={"scorer_version": SCORER_VERSION})
    stale = lying.model_copy(update={"scorer_version": "2026.01-some-older-scorer"})

    assert failure_case(trusted) == "S"
    assert failure_case(stale) == "A"
    # Unstamped is the same state as stale: nothing said which scorer wrote it.
    assert lying.scorer_version is None
    assert failure_case(lying) == "A"


def test_a_rescore_repartitions_a_run_stamped_by_an_older_scorer() -> None:
    """The consequence at the level a reader sees: the table, not the row.

    A summary rebuilt from rows an older scorer classified must report this
    scorer's partition of the same completions. Reading the stored letters would
    silently republish the previous instrument's table under this one's name.
    """
    stale = [
        sample.model_copy(
            update={"failure_case": "S", "scorer_version": "2026.01-some-older-scorer"}
        )
        for sample in FIXTURE_SAMPLES
    ]
    repartitioned = case_counts(stale, answer_format="boxed")

    assert [(c.case, c.samples) for c in repartitioned] == [
        (c.case, c.samples) for c in case_counts(FIXTURE_SAMPLES, answer_format="boxed")
    ]
    assert [c.case for c in repartitioned] != ["S"]
