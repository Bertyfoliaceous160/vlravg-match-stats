"""Aggregation: the hierarchy the interval depends on, and the truncation rule."""

from __future__ import annotations

import pytest

from limite_evals.aggregate import (
    DEFAULT_TRUNCATION_POLICY,
    RUNGS,
    by_taskset,
    group_by_problem,
    k_absence_reason,
    relaxation_gap,
    scored,
    stop_class,
    stop_reasons,
    summarize,
    summarize_task,
)
from limite_evals_core.schema import (
    PROTOCOL_METRICS,
    Fingerprint,
    MetricSet,
    NativeMetric,
    Pins,
    ProtocolSample,
    RunManifest,
    SampleResult,
    Sampling,
)

PHYSICS_GROUPED = MetricSet(
    name="physics-grouped",
    metrics=("static_accuracy", "dynamic_accuracy"),
    correctness=("static_accuracy", "dynamic_accuracy"),
    native_metrics=(
        NativeMetric(
            name="static_accuracy",
            minimum=0.0,
            maximum=1.0,
            presentation="percent",
            provenance="official ABench static accuracy",
            aggregation="source_group_mean",
            source_variants=("0",),
        ),
        NativeMetric(
            name="dynamic_accuracy",
            minimum=0.0,
            maximum=1.0,
            presentation="percent",
            provenance="official ABench dynamic accuracy",
            aggregation="source_group_all",
            source_variants=("0", "1", "2", "3"),
        ),
    ),
)


def _physics_group(group: int, values: tuple[float, float, float, float]) -> list[SampleResult]:
    return [
        SampleResult(
            task="abench_phy_b",
            problem_index=group * 4 + variant,
            rollout_index=0,
            gold="42",
            completion="42",
            source_group_id=str(group),
            source_variant_id=str(variant),
            metrics={"static_accuracy": value, "dynamic_accuracy": value},
        )
        for variant, value in enumerate(values)
    ]


def test_source_group_metrics_resample_outer_groups_without_redrawing_variants() -> None:
    samples = _physics_group(0, (1.0, 1.0, 1.0, 1.0)) + _physics_group(
        1, (1.0, 1.0, 0.0, 1.0)
    )
    summary = summarize_task(
        "abench_phy_b", samples, metric_set=PHYSICS_GROUPED, resamples=200
    )

    # Four fixed variants form one source problem; they are not four stochastic
    # rollouts of one prompt and must not be reported as avg@4.
    assert (summary.problems, summary.group_size) == (2, 1)
    assert summary.static_accuracy.value == 1.0
    assert summary.dynamic_accuracy.value == 0.5


@pytest.mark.parametrize(
    "samples, message",
    [
        (
            _physics_group(0, (1.0, 1.0, 1.0, 1.0))[:-1],
            "incomplete source variants",
        ),
        (
            _physics_group(0, (1.0, 1.0, 1.0, 1.0))
            + [_physics_group(0, (1.0, 1.0, 1.0, 1.0))[0]],
            "duplicate source variant",
        ),
        (
            [
                SampleResult(
                    task="abench_phy_b",
                    problem_index=0,
                    rollout_index=0,
                    gold="42",
                    completion="42",
                    source_group_id="0",
                    metrics={"dynamic_accuracy": 1.0},
                )
            ],
            "source_variant_id",
        ),
    ],
)
def test_invalid_source_group_shapes_fail_loudly(samples, message) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_task(
            "abench_phy_b", samples, metric_set=PHYSICS_GROUPED, resamples=20
        )


def test_static_accuracy_ignores_nonzero_variants() -> None:
    samples = _physics_group(0, (1.0, 0.0, 0.0, 0.0)) + _physics_group(
        1, (0.0, 1.0, 1.0, 1.0)
    )
    summary = summarize_task(
        "abench_phy_b", samples, metric_set=PHYSICS_GROUPED, resamples=100
    )
    assert summary.static_accuracy.value == 0.5
    assert summary.dynamic_accuracy.value == 0.0


def test_source_variants_are_not_redrawn_inside_the_bootstrap() -> None:
    summary = summarize_task(
        "abench_phy_b",
        _physics_group(0, (1.0, 1.0, 1.0, 0.0)),
        metric_set=PHYSICS_GROUPED,
        resamples=500,
    )
    assert (
        summary.dynamic_accuracy.value,
        summary.dynamic_accuracy.low,
        summary.dynamic_accuracy.high,
    ) == (0.0, 0.0, 0.0)

#: The minimum a manifest needs to exist, for the one test that asserts the
#: policy is read off it rather than passed beside it.
MANIFEST = RunManifest(
    run_id="r001",
    created_at="2026-08-10T00:00:00Z",
    checkpoint="/work/ckpt",
    stage="sft",
    profile="chat",
    suite="math-standard",
    sampling=Sampling(
        temperature=0.0, top_p=1.0, max_tokens=4096, stop=[], skip_special_tokens=True
    ),
    pins=Pins(
        verifiers="d30a3f48",
        research_environments="f9c43a74",
        math_verify="0.9.0",
        dataset_revision="6e4ed1a2",
    ),
    fingerprint=Fingerprint(
        architecture="LimiteForCausalLM",
        artifact_sha256="a" * 64,
        template_sha256="b" * 64,
        max_model_len=8192,
        dtype="bfloat16",
    ),
)

#: Two metrics and no rungs: the shape a constraint-checked taskset reports, and
#: the case none of the five fixed fields could hold.
CONSTRAINTS = MetricSet(
    name="native-constraints",
    metrics=("prompt_level", "instruction_level"),
    correctness=("prompt_level", "instruction_level"),
)


def _sample(task: str, problem: int, rollout: int, correct: bool = True, **overrides) -> SampleResult:
    fields = {
        "task": task,
        "problem_index": problem,
        "rollout_index": rollout,
        "gold": "42",
        "completion": "\\boxed{42}",
        "strict": correct,
        "lenient": correct,
        "permissive": correct,
        "format_ok": True,
    }
    return SampleResult(**{**fields, **overrides})


def _group(task: str, problem: int, size: int, correct: bool, **overrides) -> list[SampleResult]:
    return [_sample(task, problem, r, correct, **overrides) for r in range(size)]


def test_rollouts_are_grouped_under_their_problem() -> None:
    """avg@k averages problems, so one problem cannot outweigh another by rollout count."""
    samples = _group("aime24", 0, 32, True) + _group("aime24", 1, 4, False)
    assert summarize_task("aime24", samples, resamples=200).strict.value == 0.5


def test_the_interval_reflects_between_problem_variance() -> None:
    """The failure mode decision 12 names: flattening the rollouts.

    Every group here is unanimous, so a flat list of 960 rollouts would be a
    single group and `bootstrap_interval` would return a zero-width interval.
    Grouping by problem is what keeps the AIME interval honest.
    """
    samples = [s for i in range(30) for s in _group("aime24", i, 32, i % 3 == 0)]
    summary = summarize_task("aime24", samples, resamples=2000)
    assert summary.strict.high - summary.strict.low > 0.2


def test_problems_and_group_size_are_counted_from_the_grouping() -> None:
    samples = [s for i in range(30) for s in _group("aime24", i, 32, True)]
    summary = summarize_task("aime24", samples, resamples=200)
    assert (summary.problems, summary.group_size) == (30, 32)


def test_a_lost_rollout_does_not_rewrite_the_group_size() -> None:
    """A ragged group reports the size the run intended, not the one it achieved."""
    samples = _group("math500", 0, 4, True) + _group("math500", 1, 2, True)
    assert summarize_task("math500", samples, resamples=200).group_size == 4


def _half_truncated() -> list[SampleResult]:
    """Four problems, two of them truncated, every rollout carrying an answer."""
    truncated = [
        s
        for i in (0, 1)
        for s in _group("aime24", i, 4, True, truncated=True, finish_reason="length")
    ]
    return truncated + [s for i in (2, 3) for s in _group("aime24", i, 4, True)]


def test_a_truncated_sample_is_never_dropped_under_either_policy() -> None:
    """The regression this test exists for: filtering truncated samples out.

    Dropping the two truncated problems would report strict 100% over 2 problems
    at a 0% truncation rate -- a number inflated by removing the completions that
    ran out of context. Neither policy may do that, so both keep four problems
    and both report the rate; what they disagree about is only what those
    completions scored.
    """
    for policy in ("score", "fail"):
        summary = summarize_task(
            "aime24", _half_truncated(), resamples=200, truncation_policy=policy
        )
        assert summary.problems == 4, policy
        assert summary.truncated.value == 0.5, policy


def test_the_policy_decides_what_a_truncated_sample_scores() -> None:
    """The two answers, on the same four problems, side by side.

    Under `fail` the truncated half is wrong whatever it wrote: strict 50%.
    Under `score` its boxed answer stands: strict 100% at a 50% truncation rate,
    which is why the rate is reported beside every column rather than under a
    diagnostic heading.
    """
    gated = summarize_task("aime24", _half_truncated(), resamples=200, truncation_policy="fail")
    graded = summarize_task("aime24", _half_truncated(), resamples=200, truncation_policy="score")

    assert (gated.strict.value, gated.lenient.value, gated.permissive.value) == (0.5, 0.5, 0.5)
    assert (graded.strict.value, graded.lenient.value, graded.permissive.value) == (1.0, 1.0, 1.0)
    assert gated.truncated.value == graded.truncated.value == 0.5


def test_the_policy_is_read_off_the_manifest_not_passed_beside_it() -> None:
    """`summarize` may not compute one quantity and manifest another.

    The policy is the field that says what the numbers mean, so a summary
    computed under one and manifested under the other is the single
    inconsistency this function must be unable to produce.
    """
    samples = _half_truncated()
    graded = MANIFEST.model_copy(update={"truncation_policy": "score"})
    gated = MANIFEST.model_copy(update={"truncation_policy": "fail"})

    assert summarize(graded, samples, resamples=200).tasks[0].strict.value == 1.0
    assert summarize(gated, samples, resamples=200).tasks[0].strict.value == 0.5


def test_an_unstated_policy_is_the_one_the_older_summary_was_computed_under() -> None:
    """The manifest defaults to `fail`; the instrument defaults to `score`.

    The two defaults deliberately disagree, and both are right where they sit. A
    result file written before the field existed was produced under the gate, so
    loading it as anything else would restate its numbers as something they are
    not -- while a new run has no history to preserve and gets today's rule.
    """
    assert MANIFEST.truncation_policy == "fail"
    assert summarize(MANIFEST, _half_truncated(), resamples=200).tasks[0].strict.value == 0.5
    assert DEFAULT_TRUNCATION_POLICY == "score"


def test_a_truncated_rollout_still_counts_inside_its_problem() -> None:
    """Dropping within a group is the subtler version of the same inflation.

    The group keeps both rollouts under either policy. Only the second value
    moves, which is exactly the difference the policy names.
    """
    samples = [
        _sample("math500", 0, 0, True),
        _sample("math500", 0, 1, True, truncated=True, finish_reason="length"),
    ]
    groups = group_by_problem(samples)
    assert [len(group) for group in groups] == [2]
    assert [scored(s, "strict", truncation_policy="fail") for s in groups[0]] == [True, False]
    assert [scored(s, "strict", truncation_policy="score") for s in groups[0]] == [True, True]


def test_truncation_does_not_erase_the_format_diagnostic() -> None:
    """`format_ok` describes the completion, not whether it was right.

    Forcing it to False on truncation would hide that the model did emit a
    well-formed answer before the context ran out, which is the distinction the
    diagnostic exists to make.
    """
    cut = _sample("math500", 0, 0, True, truncated=True)
    assert scored(cut, "format_ok") is True
    assert scored(cut, "truncated") is True


def test_grouping_does_not_depend_on_arrival_order() -> None:
    """A concurrent runner returns samples in finish order; the interval must not move."""
    ordered = [s for i in range(10) for s in _group("aime24", i, 4, i % 2 == 0)]
    shuffled = list(reversed(ordered))
    assert summarize_task("aime24", shuffled, resamples=500) == summarize_task(
        "aime24", ordered, resamples=500
    )


def test_tasksets_are_kept_apart_and_in_suite_order() -> None:
    """Pooling tasksets would average a 30-problem set into a 500-problem one."""
    samples = _group("aime24", 0, 2, True) + _group("math500", 0, 2, False)
    split = by_taskset(samples)
    assert list(split) == ["aime24", "math500"]
    assert [len(group) for group in split.values()] == [2, 2]


def test_the_gap_is_lenient_minus_its_base() -> None:
    """A model that solves the problem but does not box it: 0 strict, 1 lenient.

    On a ladder summary, which records no protocols, the base is `strict` --
    which is what every summary written before protocols existed carries.
    """
    samples = [
        _sample("math500", 0, 0, False, lenient=True, permissive=True, format_ok=False),
        _sample("math500", 1, 0, True),
    ]
    summary = summarize_task("math500", samples, resamples=200)
    assert summary.strict.value == 0.5
    assert summary.lenient.value == 1.0
    assert relaxation_gap(summary) == ("strict", 0.5)


# --- protocols: what a column names, and what absence looks like -------------


def _protocol_sample(
    task: str,
    problem: int,
    rollout: int,
    *,
    exact: bool | None,
    reference: bool,
    lenient: bool,
    relaxes: str,
    reason: str = "",
    **overrides,
) -> SampleResult:
    """A rollout carrying protocol verdicts, as the runner writes them.

    `exact=None` is the D-107 shape: no artefact was bound, so there is no
    verdict and the metric is absent from `metrics` rather than present as zero.
    """
    protocols = {
        "exact": ProtocolSample(
            protocol="exact",
            anchor=None if exact is None else "hendrycks/math is_equiv",
            correct=exact,
            unavailable_reason=reason or None,
        ),
        "reference": ProtocolSample(
            protocol="reference", anchor="verify_boxed_math_answer", correct=reference
        ),
        "lenient": ProtocolSample(
            protocol="lenient", anchor=f"{relaxes} + waivers", correct=lenient, relaxes=relaxes
        ),
    }
    metrics = {
        name: float(sample.correct)
        for name, sample in protocols.items()
        if sample.correct is not None
    }
    fields = {
        "task": task,
        "problem_index": problem,
        "rollout_index": rollout,
        "gold": "42",
        "completion": "\\boxed{42}",
        "protocols": protocols,
        "metrics": metrics,
        "format_ok": True,
    }
    return SampleResult(**{**fields, **overrides})


def test_an_unavailable_protocol_is_absent_rather_than_zero() -> None:
    """The reading D-107 exists to prevent: an unobtainable published grader
    reported as a checkpoint that answered every problem wrongly."""
    reason = "the AIME publishes an answer key, not a grader"
    samples = [
        _protocol_sample(
            "aime24", i, 0, exact=None, reference=True, lenient=True, relaxes="reference",
            reason=reason,
        )
        for i in range(4)
    ]
    summary = summarize_task(
        "aime24", samples, metric_set=PROTOCOL_METRICS, resamples=200
    )

    assert "exact" not in summary.metrics
    assert summary.metrics["reference"].value == 1.0
    record = summary.protocol("exact")
    assert record is not None and not record.available
    assert record.unavailable_reason == reason


def test_the_relaxation_base_is_recorded_per_taskset() -> None:
    """D-108's fallback: the same column relaxes two different anchors, and each
    taskset's summary has to say which one it relaxed."""
    with_grader = [
        _protocol_sample("math500", i, 0, exact=True, reference=True, lenient=True, relaxes="exact")
        for i in range(2)
    ]
    without = [
        _protocol_sample(
            "aime24", i, 0, exact=None, reference=True, lenient=True, relaxes="reference",
            reason="no published grader",
        )
        for i in range(2)
    ]
    math500 = summarize_task("math500", with_grader, metric_set=PROTOCOL_METRICS, resamples=200)
    aime = summarize_task("aime24", without, metric_set=PROTOCOL_METRICS, resamples=200)

    assert math500.protocol("lenient").relaxes == "exact"
    assert aime.protocol("lenient").relaxes == "reference"
    # And the gap names the base it was measured against, on each of them.
    assert relaxation_gap(math500)[0] == "exact"
    assert relaxation_gap(aime)[0] == "reference"


def test_the_waiver_list_travels_with_the_relaxation() -> None:
    """"A bit more permissive" is a list in the record, not an adjective."""
    samples = [
        _protocol_sample("math500", 0, 0, exact=True, reference=True, lenient=True, relaxes="exact")
    ]
    summary = summarize_task("math500", samples, metric_set=PROTOCOL_METRICS, resamples=200)
    assert "unclosed_think" in summary.protocol("lenient").waivers
    assert summary.protocol("reference").waivers == ()


def test_the_truncation_policy_reaches_every_protocol() -> None:
    """It reaches the protocols exactly as it reached the rungs, both ways."""
    cut = _protocol_sample(
        "math500", 0, 0, exact=True, reference=True, lenient=True, relaxes="exact",
        truncated=True, finish_reason="length",
    )
    for protocol in ("exact", "reference", "lenient"):
        assert (
            scored(
                cut,
                protocol,
                correctness=PROTOCOL_METRICS.correctness,
                truncation_policy="fail",
            )
            is False
        )
        assert scored(
            cut,
            protocol,
            correctness=PROTOCOL_METRICS.correctness,
            truncation_policy="score",
        ) == 1.0


# --- the stop-reason split ---------------------------------------------------


def test_a_stop_string_and_a_learned_terminator_are_told_apart() -> None:
    """Both report `finish_reason: stop`, and the difference between them is the
    difference between a checkpoint halted from outside and one that ends
    itself -- which is what separates a pretraining checkpoint from an SFT one."""
    halted = _sample("math500", 0, 0, finish_reason="stop", stop_reason="\nProblem:")
    ended = _sample("math500", 0, 1, finish_reason="stop", stop_reason=None)
    by_token = _sample("math500", 0, 2, finish_reason="stop", stop_reason=151645)
    cut = _sample("math500", 0, 3, finish_reason="length", truncated=True)

    assert stop_class(halted) == "stop_string"
    assert stop_class(ended) == "eos"
    assert stop_class(by_token) == "stop_token"
    assert stop_class(cut) == "length"


def test_the_stop_reason_breakdown_reaches_the_summary() -> None:
    samples = [
        _sample("math500", 0, 0, finish_reason="stop", stop_reason="\nProblem:"),
        _sample("math500", 0, 1, finish_reason="stop", stop_reason="\nProblem:"),
        _sample("math500", 1, 0, finish_reason="stop", stop_reason=None),
        _sample("math500", 1, 1, finish_reason="length", truncated=True),
    ]
    assert stop_reasons(samples) == {"stop_string": 2, "eos": 1, "length": 1}
    summary = summarize_task("math500", samples, resamples=200)
    assert summary.stop_reasons["eos"] == 1


def _constraint_sample(problem: int, prompt: float, instruction: float, **overrides) -> SampleResult:
    """A rollout with no boxed answer to extract, scored by constraint checks alone."""
    fields = {
        "task": "native-task",
        "problem_index": problem,
        "rollout_index": 0,
        "gold": "",
        "completion": "a response",
        "metrics": {"prompt_level": prompt, "instruction_level": instruction},
    }
    return SampleResult(**{**fields, **overrides})


def test_a_taskset_can_declare_metrics_that_are_not_rungs() -> None:
    """The change itself: a taskset reports what it measured rather than five rungs.

    Nothing on the ladder path runs here -- these rollouts carry no boxed answer
    and score at no rung -- and the numbers still come back grouped by problem
    with a hierarchical interval each.
    """
    samples = [_constraint_sample(0, 1.0, 1.0), _constraint_sample(1, 0.0, 0.5)]
    summary = summarize_task("native-task", samples, metric_set=CONSTRAINTS, resamples=200)

    assert list(summary.metrics) == ["prompt_level", "instruction_level"]
    assert summary.metrics["prompt_level"].value == 0.5
    assert summary.metrics["instruction_level"].value == 0.75
    assert summary.metric_set.headline == "prompt_level"


def test_the_policy_reaches_a_metric_set_that_is_not_the_ladder() -> None:
    """Whichever set names the correctness metrics, the policy governs them.

    Reaching only the three rungs would let the next taskset resolve truncation
    on its own terms and do it silently -- a constraint checker would score a
    cut-off response on whatever constraints it satisfied before the context ran
    out, whatever the run declared.
    """
    cut = _constraint_sample(0, 1.0, 1.0, truncated=True, finish_reason="length")
    for metric in ("prompt_level", "instruction_level"):
        assert (
            scored(cut, metric, correctness=CONSTRAINTS.correctness, truncation_policy="fail")
            is False
        )
        assert (
            scored(cut, metric, correctness=CONSTRAINTS.correctness, truncation_policy="score")
            == 1.0
        )

    both = [cut, _constraint_sample(1, 1.0, 1.0)]
    gated = summarize_task(
        "native-task", both, metric_set=CONSTRAINTS, resamples=200, truncation_policy="fail"
    )
    graded = summarize_task(
        "native-task", both, metric_set=CONSTRAINTS, resamples=200, truncation_policy="score"
    )
    assert gated.problems == graded.problems == 2
    assert gated.metrics["prompt_level"].value == 0.5
    assert graded.metrics["prompt_level"].value == 1.0


# --- the answer region: a metric that cannot be evaluated on a rollout -------


def _mid_reasoning(task: str, problem: int, rollout: int = 0, **overrides) -> SampleResult:
    """A rollout that hit the token cap with its think block still open.

    The shape decision PD3-2 is about: there is no answer region, so `format_ok`
    has no question to answer, and the row records that as its own fact rather
    than as a formatting failure.
    """
    fields = {
        "task": task,
        "problem_index": problem,
        "rollout_index": rollout,
        "gold": "42",
        "completion": "<think>I am still working on it",
        "finish_reason": "length",
        "truncated": True,
        "format_ok": False,
        "format_not_evaluable": True,
    }
    return SampleResult(**{**fields, **overrides})


def test_a_rollout_with_no_answer_region_scores_nothing_at_format_ok() -> None:
    """`None`, not zero, and only for the metrics the taskset declared.

    A `False` here would be a formatting failure the model never got the chance
    to commit; the correctness rungs are untouched, because a completion that
    never presented an answer did not get one right.
    """
    sample = _mid_reasoning("math500", 0)
    assert scored(sample, "format_ok", requires_answer_region=("format_ok",)) is None
    assert scored(sample, "strict", requires_answer_region=("format_ok",)) is False
    assert scored(sample, "truncated", requires_answer_region=("format_ok",)) is True


def test_the_not_evaluable_rollouts_leave_only_the_format_ok_denominator() -> None:
    """Four rollouts, one of them mid-reasoning. `format_ok` is 2 of 3, not 2 of 4.

    Everything else still counts all four -- the truncation rate, the strict
    column and the problem count -- which is what separates this from dropping a
    rollout.
    """
    samples = [
        _sample("math500", 0, 0, True, format_ok=True),
        _sample("math500", 1, 0, True, format_ok=True),
        _sample("math500", 2, 0, False, format_ok=False),
        _mid_reasoning("math500", 3),
    ]
    summary = summarize_task("math500", samples, resamples=200)

    assert summary.problems == 4
    assert summary.metrics["format_ok"].value == 2 / 3
    assert summary.metrics["truncated"].value == 0.25
    assert summary.format_not_evaluable_rate == 0.25


def test_a_taskset_that_never_left_its_reasoning_reports_no_format_ok_at_all() -> None:
    """Absent with a reason rather than zero, which is the D-107 distinction.

    Zero would say the checkpoint formatted nothing. What happened is that the
    question was never reachable on any rollout.
    """
    samples = [_mid_reasoning("math500", i) for i in range(4)]
    summary = summarize_task("math500", samples, resamples=200)

    assert "format_ok" not in summary.metrics
    assert summary.metrics["truncated"].value == 1.0
    assert summary.format_not_evaluable_rate == 1.0


def test_the_exclusion_is_not_the_truncation_policy_wearing_another_name() -> None:
    """A truncated rollout that closed its think block is scored exactly as before.

    The policy decides what an unfinished completion scores; this decides
    whether there was anything to score. Only the second shape leaves a
    denominator, under either policy.
    """
    closed = _sample(
        "math500", 0, 0, True, truncated=True, finish_reason="length", format_ok=True
    )
    for policy in ("score", "fail"):
        assert scored(closed, "format_ok", truncation_policy=policy) is True
        assert scored(
            _mid_reasoning("math500", 1), "format_ok", truncation_policy=policy
        ) is None


def test_a_constraint_checker_excludes_both_accuracies_and_not_format_ok() -> None:
    """Which metrics the exclusion reaches is the taskset's declaration.

    A constraint checker has no column that survives the answer region being
    absent: scoring "use no commas" against a pure reasoning fragment is not a
    strict reading, it is an invented one. So its metric set names both
    accuracies, and the row carries no value for either.
    """
    metric_set = CONSTRAINTS.model_copy(
        update={"requires_answer_region": ("prompt_level", "instruction_level")}
    )
    samples = [
        _constraint_sample(0, 1.0, 1.0),
        _constraint_sample(1, 0.0, 0.5),
        # The runner writes no metrics at all on such a row, because there was
        # no response for the checkers to read.
        _mid_reasoning("native-task", 2, metrics={}, completion="<think>still thinking"),
    ]
    summary = summarize_task("native-task", samples, metric_set=metric_set, resamples=200)

    assert summary.problems == 3
    assert summary.metrics["prompt_level"].value == 0.5
    assert summary.metrics["instruction_level"].value == 0.75
    assert summary.format_not_evaluable_rate == 1 / 3


# --- the k columns -----------------------------------------------------------


def _protocol_group(task: str, problem: int, size: int, correct: int, **kwargs):
    """One problem's rollouts under the protocols, `correct` of them right."""
    return [
        _protocol_sample(
            task,
            problem,
            rollout,
            exact=rollout < correct,
            reference=rollout < correct,
            lenient=rollout < correct,
            relaxes="exact",
            **kwargs,
        )
        for rollout in range(size)
    ]


def test_pass_at_one_is_the_headline_column_itself() -> None:
    """The property node RES-396 verifies, asserted here at the boundary that
    produces both numbers: same estimator, same seed, same resample sequence, so
    the whole interval matches and not only the point estimate."""
    samples = [s for i in range(4) for s in _protocol_group("math500", i, 4, i)]
    summary = summarize_task("math500", samples, metric_set=PROTOCOL_METRICS, resamples=200)

    assert summary.metrics["pass@1"] == summary.metrics["exact"]


def test_the_k_columns_read_the_declaration_and_the_estimators() -> None:
    """Four problems, one of them answered by every rollout and the rest by none.

    `pass@4` sees the whole group of four, so it is 1 on that problem and 0 on
    the others; `pass^4` agrees here because the one solved problem is solved
    throughout. The k set is the metric set's, so both columns exist and neither
    was selected by a flag.
    """
    samples = [s for i in range(4) for s in _protocol_group("math500", i, 4, 4 if i == 0 else 0)]
    summary = summarize_task("math500", samples, metric_set=PROTOCOL_METRICS, resamples=200)

    assert summary.metrics["pass@4"].value == 0.25
    assert summary.metrics["pass^4"].value == 0.25
    # One right rollout in four: a draw of four always holds it, a draw that is
    # right throughout never does. The two columns are what separates them.
    mixed = [s for i in range(4) for s in _protocol_group("aime24", i, 4, 1)]
    summary = summarize_task("aime24", mixed, metric_set=PROTOCOL_METRICS, resamples=200)
    assert (summary.metrics["pass@4"].value, summary.metrics["pass^4"].value) == (1.0, 0.0)


def test_a_k_above_the_group_size_is_absent_with_its_reason_not_zero() -> None:
    """D-107 at the k columns: `pass@32` over a group of four is unasked.

    Absent from the metrics, and the reason is recoverable from the summary
    alone -- `group_size` beside the k set the metric set declares.
    """
    samples = [s for i in range(4) for s in _protocol_group("math500", i, 4, 2)]
    summary = summarize_task("math500", samples, metric_set=PROTOCOL_METRICS, resamples=200)

    assert "pass@32" not in summary.metrics and "pass^32" not in summary.metrics
    assert "pass@4" in summary.metrics
    assert summary.group_size == 4 and 32 in summary.metric_set.pass_at_k


def test_the_absent_k_reason_separates_an_unasked_question_from_an_unexplained_one() -> None:
    """One sentence, two surfaces: the run's report and the cross-run index both read it here.

    A group size that cannot answer the k is a reason; a k the group size could
    have answered and nobody measured is not, and must read as the unexplained
    absence it is rather than borrow a cause that does not describe it.
    """
    samples = [s for i in range(4) for s in _protocol_group("math500", i, 4, 2)]
    summary = summarize_task("math500", samples, metric_set=PROTOCOL_METRICS, resamples=200)

    assert k_absence_reason(summary, "pass@32", 32) == (
        "this taskset drew 4 rollouts per problem, and `pass@32` needs 32"
    )
    assert k_absence_reason(summary, "pass@4", 4) == "not measured"


def test_a_single_draw_taskset_reports_pass_at_one_and_nothing_else() -> None:
    """The other end of the same rule: k=1 is answerable and 4 and 32 are not."""
    samples = [s for i in range(4) for s in _protocol_group("gsm8k", i, 1, i % 2)]
    summary = summarize_task("gsm8k", samples, metric_set=PROTOCOL_METRICS, resamples=200)

    assert [name for name in summary.metrics if name.startswith("pass")] == ["pass@1"]
    assert summary.metrics["pass@1"] == summary.metrics["exact"]


def test_the_k_columns_follow_the_headline_to_reference_where_exact_is_unbound() -> None:
    """D-06 with D-107's fallback: the headline protocol, whichever one that is.

    `exact` is unavailable here, so the k columns describe `reference` -- the
    one anchor that cannot be absent -- rather than disappearing with it or
    being promoted onto a relaxation.
    """
    samples = [
        _protocol_sample(
            "aime24", problem, rollout,
            exact=None, reference=rollout == 0, lenient=True, relaxes="reference",
            reason="no published grader",
        )
        for problem in range(4)
        for rollout in range(4)
    ]
    summary = summarize_task("aime24", samples, metric_set=PROTOCOL_METRICS, resamples=200)

    assert "exact" not in summary.metrics
    assert summary.metrics["pass@1"] == summary.metrics["reference"]
    assert summary.metrics["pass@4"].value == 1.0


def test_the_truncation_policy_reaches_the_k_columns_at_the_same_boundary() -> None:
    """Not a second rule inside an estimator: the k columns are fed exactly what
    `scored` produced for the headline, so `fail` empties them with it."""
    samples = [
        s
        for i in range(4)
        for s in _protocol_group("math500", i, 4, 4, truncated=True, finish_reason="length")
    ]
    scoring = summarize_task(
        "math500", samples, metric_set=PROTOCOL_METRICS, resamples=200, truncation_policy="score"
    )
    failing = summarize_task(
        "math500", samples, metric_set=PROTOCOL_METRICS, resamples=200, truncation_policy="fail"
    )

    assert (scoring.metrics["pass@4"].value, scoring.metrics["pass^4"].value) == (1.0, 1.0)
    assert (failing.metrics["pass@4"].value, failing.metrics["pass^4"].value) == (0.0, 0.0)


def test_a_metric_set_that_declares_no_k_set_reports_no_k_column() -> None:
    """The ladder and every constraint checker: unchanged, down to the columns."""
    samples = [s for i in range(4) for s in _group("aime24", i, 4, True)]
    summary = summarize_task("aime24", samples, resamples=200)

    assert not [name for name in summary.metrics if name.startswith("pass")]
    assert list(summary.metrics) == list(RUNGS)


def test_the_run_records_which_k_each_taskset_was_asked() -> None:
    """The comparability key: stamped from what was computed, not from a caller.

    A taskset declaring no k set contributes nothing, so a run that reports no k
    column records the empty string a pre-D-06 summary carries.
    """
    samples = [s for i in range(2) for s in _protocol_group("math500", i, 4, 2)]
    summary = summarize(MANIFEST, samples, metric_sets={"math500": PROTOCOL_METRICS}, resamples=200)

    assert summary.manifest.pass_at_k == "math500=1+4+32"
    assert summarize(MANIFEST, _half_truncated(), resamples=200).manifest.pass_at_k == ""
