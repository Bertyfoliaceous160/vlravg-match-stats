"""From a flat list of scored samples to the numbers a run reports.

Two things here are load-bearing rather than incidental.

The grouping is hierarchical -- taskset, then problem, then that problem's
rollouts -- because that is the shape `bootstrap_interval` resamples. Handing it
one flat group per taskset would leave a single group to draw from, collapsing
the interval to zero width on exactly the 30-problem AIME sets where
between-problem variance dominates and decision 12 says the interval matters
most.

And what a truncated completion scores is decided here, once, for every sample
source -- `TruncationPolicy` chooses between the grader's verdict on the text
that was produced (`score`, the default) and wrong at every correctness metric
(`fail`). The choice is the run's and is recorded in its manifest; the boundary
is not, which is the part that was always load-bearing. A rule each scoring path
is trusted to have applied is a rule some future path will not apply.

What neither policy does is **drop** the rollout. Dropping reads as tidying and
silently inflates every column: the completions that run out of context are the
ones that would have failed, so removing them raises the score by removing the
failures. That number is not obtainable from this module under any policy, and
the truncation rate travels beside every column so a reader can see how much of
it is at stake.

Which numbers get computed is the taskset's declaration rather than this
module's. `MetricSet` names them; a taskset scored by a constraint checker
declares its own and reaches the same grouping and the same truncation policy.

A third thing is load-bearing now that a column names an artefact rather than a
rung: **a metric nothing measured is left out rather than averaged as zero.** A
taskset whose published grader could not be bound has no `exact` verdict on any
rollout, and a bootstrap over that absence would report the missing artefact as
a checkpoint that got every problem wrong. The reason travels beside the numbers
in `TaskSummary.protocols`, so the column is absent *and* explained.

A fourth is the boundary between "this rollout scored nothing" and "this metric
had nothing to read on this rollout". A completion whose think block ran into the
token cap never reached an answer region, so the metrics a taskset declares in
`MetricSet.requires_answer_region` -- `format_ok` for the ladder, both accuracies
for a constraint checker -- leave that rollout out of their denominator instead
of recording a failure the model never got to commit. The rollout is not dropped:
every other column still counts it, and the excluded share is reported as
`TaskSummary.format_not_evaluable_rate` beside the numbers it qualifies.

And a fifth, which is about the instrument rather than about any rollout: this
module refuses to summarise an E42-era run at all. Those completions reached the
grader with every reasoning delimiter already deleted, so numbers computed from
them under think-aware scoring describe the tokenizer rather than the
checkpoint. `assert_scoreable` is where that refusal lives, and it refuses --
loudly, with the boundary named -- rather than warning.

A sixth is what the same rollouts say about a smaller budget. A taskset that drew
32 rollouts per problem carries the answer to "how often is one of four right"
and "how often are all thirty-two right" as well as to the mean it reports, and
`pass@k` and `pass^k` are those readings. Three rules keep them from becoming a
second scoring path. They are computed on the **headline protocol only**, which
is D-06: `exact` where a published grader was bound and `reference` where none
was, exactly as `relaxation_base` already falls back. They are fed the values
`scored` produced for that column, so the truncation policy reaches them at the
boundary it reaches everything else at rather than a second time inside an
estimator. And a k above the rollouts a problem got is **absent with its reason**
rather than zero, for the same reason an unbindable grader is: `pass@32` over a
group of four is a question this run did not ask.

The comparison-timeout rate is computed here too and is none of those things: not
a declared metric, not resampled, not something a truncated completion can fail.
It is the fraction of rollouts whose answer comparison ran out of time, and since
a timeout scores wrong, it is how far the metrics beside it may be understated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from limite_evals import taxonomy
from limite_evals_core.answer_scoring import validate_answer_scorer_records
from limite_evals_core.protocols import PROTOCOLS, RELAXATION_FALLBACK_BASE, TasksetProtocols
from limite_evals_core.protocols import Unavailable as UnavailableGrader
from limite_evals_core.schema import (
    LADDER,
    SCORER_VERSION,
    CaseCount,
    InstrumentEraMismatch,
    Interval,
    MetricSet,
    NativeMetric,
    ProtocolRecord,
    RunManifest,
    RunSummary,
    SampleResult,
    StopClass,
    TaskSummary,
    TruncationPolicy,
)
from limite_evals_core.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    Statistic,
    bootstrap_interval,
    mean_of_group,
    pass_all_k,
    pass_any_k,
)

#: The ladder's own metrics and its correctness subset, kept under the names the
#: report and the tests have always read them by. They are now one metric set's
#: contents rather than the schema itself, which is the whole of the change:
#: `LADDER` is what a taskset gets when it declares nothing.
RUNGS = LADDER.metrics
CORRECTNESS_RUNGS = LADDER.correctness

#: What a truncated completion scores where no policy was stated. It is the CLI's
#: default too; `RunManifest` deliberately defaults the other way, because there
#: the field's job is to say what an older summary was produced under.
DEFAULT_TRUNCATION_POLICY: TruncationPolicy = "score"


#: What a caller is told when it asks this instrument to score, or score again,
#: rollouts produced before the E42 correction. Written once so the refusal reads
#: identically wherever it is raised.
E42_QUARANTINE = (
    "{run_id} was produced by the E42-era instrument and its stored completions cannot be "
    "scored by this one. Under that instrument the legacy artifact tokenizer could not name "
    "ids 151665 and 151666, so `<think>` and `</think>` decoded to the empty string and every "
    "reasoning delimiter was deleted between the engine and the grader. What `samples.jsonl` "
    "holds is that damaged text: the delimiters were destroyed at decode time and no token ids "
    "were persisted beside the completions, so there is nothing here to recover them from. "
    "Scoring it under think-aware rules would report a model that never opened a think block, "
    "which is a claim about the tokenizer wearing a checkpoint's name. Regenerating the "
    "rollouts through the pinned vocabulary is the only honest path; the run directory is left "
    "exactly as it is, because it remains the record of what was reported at the time."
)


def assert_scoreable(manifest: RunManifest) -> None:
    """Refuse to produce numbers from an E42-era run's stored rollouts.

    Enforcement lives in the tooling and nowhere else: no run directory is
    moved, renamed or deleted by this quarantine. Those directories are the
    record of what the instrument reported when it reported it, and destroying
    them to prevent a misreading would destroy the evidence of the misreading.

    What is refused is the *production of new numbers* from that text -- which
    is what a rescore and a re-summarisation both are. Reading such a summary,
    rendering its report and comparing its manifest all still work, because none
    of them re-derives a verdict from a completion.
    """
    if manifest.instrument_era == "e42":
        raise InstrumentEraMismatch(E42_QUARANTINE.format(run_id=manifest.run_id))


def relaxation_base(bindings: TasksetProtocols) -> str:
    """Which anchor this taskset's relaxations actually relax.

    D-108's fallback, read off the registry rather than restated: `lenient`
    declares `exact` as its base, and where no published grader could be bound
    the relaxations fall back to `reference`, the one anchor that cannot be
    absent. The answer is a per-taskset fact and it belongs in the manifest,
    because `lenient(exact)` and `lenient(reference)` are two quantities under
    one column heading.
    """
    declared = PROTOCOLS["lenient"].anchor.base
    if declared == "exact" and isinstance(bindings.exact, UnavailableGrader):
        return RELAXATION_FALLBACK_BASE
    return declared or RELAXATION_FALLBACK_BASE


def headline_metric(measured: Iterable[str], metric_set: MetricSet) -> str | None:
    """Which column the k statistics are computed on, from what the taskset produced.

    D-06 says one protocol carries them, and D-107 says which: `exact` where a
    published grader was bound, and `reference` -- the one anchor that cannot be
    absent -- where none was. That is the same fallback `relaxation_base`
    applies, read here off the metrics that exist rather than off the bindings,
    so a summary answers the question from its own numbers.

    **Nothing else is ever promoted.** A relaxation is not a headline, so a
    taskset that produced neither anchor gets no k columns at all rather than
    `lenient` wearing the leading column's name -- which is `report._headline`'s
    rule, applied at the point the columns are decided instead of at the point
    they are printed.

    `measured` is the metric names the taskset actually has a number for, which
    is why this takes them rather than the samples: aggregation passes the
    intervals it just computed and the report passes `TaskSummary.metrics`, and
    the rule cannot drift between the two.
    """
    names = set(measured)
    for name in (metric_set.headline, RELAXATION_FALLBACK_BASE):
        if name in metric_set.metrics and name in names:
            return name
    return None


def k_columns(metric_set: MetricSet) -> list[tuple[str, int, Statistic]]:
    """Every `pass@k` and `pass^k` column this metric set declares, in table order.

    The name and the estimator are returned together so a column cannot be
    labelled by one function and computed by another. All the `pass@k` first and
    the `pass^k` after, because the two families answer different questions and
    a table that interleaved them would read as one series.

    **`pass^1` is not among them.** "Every one of one rollout was right" is
    `pass@1`, to the last bit, so a column for it would be the headline printed
    a third time under a name suggesting it was something else.
    """
    return [
        *((f"pass@{k}", k, pass_any_k(k)) for k in metric_set.pass_at_k),
        *((f"pass^{k}", k, pass_all_k(k)) for k in metric_set.pass_at_k if k > 1),
    ]


def k_absence_reason(task: TaskSummary, name: str, k: int) -> str:
    """Why a declared k column has no number here.

    The group size is the reason in every case this instrument produces, since a
    correctness anchor never drops a rollout from its own denominator. It is
    still written as a branch rather than as the only sentence: "not measured"
    is what an absence nobody has explained must read as, and a reason asserted
    over a cause it does not describe is worse than none.

    Beside `k_columns` and `_k_intervals` rather than in a rendering surface,
    because it is a fact about what a group of that size can answer and not
    about how a report prints it. Both surfaces that state an absent k -- the
    per-run markdown and the cross-run index -- read the sentence from here, so
    the two cannot drift into two wordings of one fact, and neither has to
    import the other to avoid it.
    """
    if k > task.group_size:
        return f"this taskset drew {task.group_size} rollouts per problem, and `{name}` needs {k}"
    return "not measured"


def stop_class(sample: SampleResult) -> StopClass:
    """Why this completion ended, as `finish_reason` alone cannot say.

    `finish_reason` reads `"stop"` for a matched stop string and for a learned
    end-of-sequence token alike, and that distinction is the whole point of the
    breakdown: a pretraining checkpoint has no reliable terminator and is halted
    from outside, while a post-trained one ends by itself. vLLM's `stop_reason`
    carries it -- the matched string, a token id, or nothing at all when the
    model simply finished -- so the split is read from the engine rather than
    guessed from the text.
    """
    if sample.finish_reason == "length":
        return "length"
    if sample.finish_reason != "stop":
        return "unknown"
    if isinstance(sample.stop_reason, str):
        return "stop_string"
    if isinstance(sample.stop_reason, int):
        return "stop_token"
    return "eos"


def stop_reasons(samples: Sequence[SampleResult]) -> dict[str, int]:
    """How this taskset's rollouts ended, counted. Empty classes are omitted."""
    counts: dict[str, int] = {}
    for sample in samples:
        counts[stop_class(sample)] = counts.get(stop_class(sample), 0) + 1
    return counts


def protocol_records(samples: Sequence[SampleResult]) -> list[ProtocolRecord]:
    """What each protocol meant for this taskset, recovered from its own rows.

    Read from the samples rather than from the bindings so that a summary can be
    rebuilt from `samples.jsonl` alone -- which is the property that makes an
    interval evidence rather than an assertion. The waiver list comes from the
    registry, since it is a property of the protocol and not of any rollout.
    """
    first = next((sample for sample in samples if sample.protocols), None)
    if first is None:
        return []
    return [
        ProtocolRecord(
            protocol=name,
            anchor=recorded.anchor,
            waivers=tuple(sorted(PROTOCOLS[name].waivers)) if name in PROTOCOLS else (),
            relaxes=recorded.relaxes,
            unavailable_reason=recorded.unavailable_reason,
        )
        for name, recorded in first.protocols.items()
    ]


#: The two lenses the A--T table states expectations for. Named here rather than
#: read off `LADDER.metrics`, because the taxonomy predicts these two and says
#: nothing about `lenient`, `format_ok` or `truncated`. Whether it predicts
#: anything at *this* taskset's `strict` is a second question, and
#: `taxonomy.expectation` is where it is answered.
TRACKED_RUNGS = ("strict", "permissive")


def failure_case(sample: SampleResult) -> str | None:
    """This rollout's A--T case: the one **this scorer** recorded, or a fresh one.

    A stored case letter is a cache, not a measurement. The row's completion is
    the evidence and the classifier is this checkout's, so a letter written by
    some other version of that classifier is an answer to a question that has
    since changed -- which is exactly what happened when the reasoning
    delimiters started surviving the decode: five cases that could never fire
    began firing, and the letters already on disk went on describing text nobody
    would classify that way now.

    So the stored letter is trusted only where `SampleResult.scorer_version`
    matches this scorer, and recomputed everywhere else -- an unstamped row, a
    row from an older scorer, a row a rescore is about to restate. The saving
    the cache buys is one classification per row; the cost of an unversioned one
    is a summary that quietly reports the previous scorer's partition.

    A constraint-checked rollout has no answer to extract, so it has no shape to
    describe and gets `None` rather than `S`. `protocols` being empty is how
    that taskset is already recognised everywhere else in this module.
    """
    if sample.failure_case is not None and sample.scorer_version == SCORER_VERSION:
        return sample.failure_case
    if not sample.protocols:
        return None
    return taxonomy.classify(sample.completion, sample.gold, truncated=sample.truncated)


def case_counts(
    samples: Sequence[SampleResult],
    *,
    answer_format: str | None = None,
    truncation_policy: TruncationPolicy = DEFAULT_TRUNCATION_POLICY,
) -> list[CaseCount]:
    """This taskset's rollouts partitioned by A--T case, with what each lens scored.

    In the frozen table's order, and cases nothing fell in are omitted for the
    same reason `stop_reasons` omits empty classes: a table of twenty rows,
    seventeen of them zero, buries the three that say something.

    `answer_format` is the taskset's own, and it decides whether the strict
    column predicts anything here: the table describes where a `\\boxed{}`
    answer sits, so it says nothing about a strict lens reading `#### 42`. It
    **defaults to stating no strict prediction** rather than to `boxed`. A
    caller that has not said which reference this taskset answers to is not
    entitled to a claim about it, and the failure mode of the other default is
    silent: it would assert the boxed column over a hashed taskset and report
    the resulting disagreements as findings.

    The two names are read through `scored`, which is what makes this table
    reconcile against the numbers beside it rather than against a second reading
    of the same rollouts. Two consequences follow and both are deliberate. The
    counts answer to the run's truncation policy: under `fail` a truncated
    rollout scores nothing under either name whatever its shape, which is the
    same statement the metrics make, and the case it fell in is unchanged
    because the policy decides what a completion scores and not what it looked
    like. And a taskset's own declared metric of that name wins over the field
    of the same name, so on a protocol run `permissive` here is the `permissive`
    protocol -- the column the report prints -- and not the ladder rung the
    taxonomy was written against. Those two are different artefacts; they
    disagreed on two of `res339-ev0-baseline-002`'s 6,015 rollouts, and the
    reconciliation is stated against the one a reader would quote.
    """
    grouped: dict[str, list[SampleResult]] = {}
    for sample in samples:
        case = failure_case(sample)
        if case is not None:
            grouped.setdefault(case, []).append(sample)
    return [
        CaseCount(
            case=spec.case,
            description=spec.description,
            samples=len(grouped[spec.case]),
            **{
                f"{rung}_expected": taxonomy.expectation(spec.case, rung, answer_format)
                for rung in TRACKED_RUNGS
            },
            **{
                f"{rung}_scored": sum(
                    bool(
                        scored(
                            sample,
                            rung,
                            correctness=TRACKED_RUNGS,
                            # Neither tracked lens needs an answer region to
                            # exist: `strict` and `permissive` are correctness,
                            # and a completion that never reached an answer did
                            # not get one right. The exclusion is `format_ok`'s,
                            # so nothing here leaves this table's denominator.
                            requires_answer_region=(),
                            truncation_policy=truncation_policy,
                        )
                    )
                    for sample in grouped[spec.case]
                )
                for rung in TRACKED_RUNGS
            },
        )
        for spec in taxonomy.TAXONOMY
        if spec.case in grouped
    ]


def case_reconciliation(
    task: TaskSummary, rung: str
) -> tuple[int, int | None, int | None]:
    """`(what the lens scored, what the taxonomy predicts, rollouts in disagreement)`.

    The taxonomy's prediction for a rollout is that it scores at `rung` only if
    its case is one the table marks "yes" there. So the predicted total is the
    lens's own count restricted to those cases, and the disagreement is the rest
    of that count -- rollouts a case said the lens could not accept and the lens
    accepted anyway. The two totals are equal exactly when the disagreement is
    empty, which is the criterion; the itemised rollouts behind a non-empty one
    are in `samples.jsonl`, found by their `failure_case`.

    **The last two are `None` where the taxonomy predicted nothing**, which for
    `strict` is every taskset whose reference does not read a box. A zero there
    would read as "predicted, and nothing disagreed", and the whole point of the
    scoping is that no prediction was made; the count of what the lens scored is
    still returned, because that is an observation and stays true either way.
    """
    scored_total = sum(getattr(count, f"{rung}_scored") for count in task.failure_cases)
    if not any(getattr(count, f"{rung}_expected") is not None for count in task.failure_cases):
        return scored_total, None, None
    predicted = sum(
        getattr(count, f"{rung}_scored")
        for count in task.failure_cases
        if getattr(count, f"{rung}_expected")
    )
    return scored_total, predicted, scored_total - predicted


def scored(
    sample: SampleResult,
    metric: str,
    *,
    correctness: Sequence[str] = CORRECTNESS_RUNGS,
    requires_answer_region: Sequence[str] = LADDER.requires_answer_region,
    truncation_policy: TruncationPolicy = DEFAULT_TRUNCATION_POLICY,
) -> float | None:
    """What `sample` contributes at `metric`, or `None` where it contributes nothing.

    **`None` is not zero and is the whole of decision PD3-2.** A rollout whose
    think block ran into the token cap never reached an answer region, so the
    metrics in `requires_answer_region` have nothing to read: `format_ok` would
    report a formatting failure the model never got the chance to commit, and a
    constraint checker would score "use no commas" against abandoned reasoning
    that happens to contain none. Those rollouts leave **that metric's**
    denominator and are counted in `TaskSummary.format_not_evaluable_rate`
    instead. Every other column still counts them, and nothing drops the
    rollout: this is a metric that cannot be evaluated on a row, not a row that
    is thrown away.

    It is deliberately not the truncation policy wearing another name. The
    policy answers "what does an unfinished completion score"; this answers "was
    there anything here to score at all", and the second question is only
    reachable now that the delimiters survive the decode. Under either policy a
    truncated completion that *closed* its think block is scored exactly as
    before.

    Under `fail` a truncated completion is wrong at every correctness metric
    whatever the scorer recovered from its text, so no column can be raised by an
    answer that survives only because the completion was cut off mid-reasoning
    and a relaxation guessed at the tail. Under `score` -- the default -- the
    grader's verdict on the text that was produced stands, and the truncation
    rate beside the column is what says how much of it came from completions that
    never reached an end.

    The decision stays here, at the aggregation boundary, under either policy.
    That is what it was for: one place that every sample source passes through,
    rather than a rule each scoring path is trusted to have applied. What changed
    is which of two answers this place gives, not who gives it.

    Neither policy drops the rollout. A truncated completion is in the
    denominator either way, because a model runs out of context on the problems
    it could not finish and removing them removes the failures with them.

    `correctness` comes from the taskset's own metric set rather than from a
    module constant, so the rule reaches a constraint-checked taskset on the
    same terms it reaches the ladder. The default is the ladder's, which is what
    every existing suite scores.

    A metric the taskset declared is read from `sample.metrics` first, before any
    field of the same name: what a taskset says it measured wins over an
    attribute that happens to share its name.
    """
    if sample.format_not_evaluable and metric in requires_answer_region:
        return None
    if truncation_policy == "fail" and metric in correctness and sample.truncated:
        return False
    if metric in sample.metrics:
        return sample.metrics[metric]
    return bool(getattr(sample, metric))


def group_by_problem(samples: Sequence[SampleResult]) -> list[list[SampleResult]]:
    """One group per problem, that problem's rollouts ordered inside it.

    Sorted rather than left in arrival order: a concurrent runner returns samples
    in whatever order the engine finished them, and the bootstrap indexes into
    this list, so an unsorted input would make the interval depend on scheduling.
    """
    groups: dict[int, list[SampleResult]] = {}
    for sample in samples:
        groups.setdefault(sample.problem_index, []).append(sample)
    return [sorted(group, key=lambda s: s.rollout_index) for _, group in sorted(groups.items())]


def by_taskset(samples: Sequence[SampleResult]) -> dict[str, list[SampleResult]]:
    """Samples split per taskset, in the order the tasksets first appear.

    First-appearance order is the suite order the runner walked, which keeps the
    report's rows in the order the operator asked for them.
    """
    split: dict[str, list[SampleResult]] = {}
    for sample in samples:
        split.setdefault(sample.task, []).append(sample)
    return split


def summarize_task(
    task: str,
    samples: Sequence[SampleResult],
    *,
    metric_set: MetricSet = LADDER,
    answer_format: str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    truncation_policy: TruncationPolicy = DEFAULT_TRUNCATION_POLICY,
) -> TaskSummary:
    """One taskset's declared numbers, each with its hierarchical bootstrap interval.

    `metric_set` is the taskset's declaration and defaults to the ladder. It
    decides both what is computed and, through its `correctness` list, which of
    those numbers a truncated completion is forced to fail where the policy is
    `fail`.

    `answer_format` is the taskset's other declaration and reaches only the A--T
    partition, where it decides whether the strict column predicts anything at
    all. It is recorded on the summary as well as used, so a reader can see the
    scope of the claim without opening the suite that produced it.

    **A metric no sample could produce is omitted, not zeroed.** A protocol whose
    artefact could not be bound has no verdict on any rollout, and averaging that
    absence would report an unobtainable published grader as a checkpoint that
    answered every problem wrongly (D-107). The reason it is missing travels in
    `protocols`, so the column reads as absent *and explained* rather than merely
    missing.

    The k columns the metric set declares are computed after the rest and from
    the same values, on whichever anchor `headline_metric` names. They are
    absent on the same terms: a k above the rollouts this taskset drew is a
    question it did not answer, and `group_size` beside them is what says so.
    """
    grouped_definitions = [
        definition
        for definition in metric_set.native_metrics
        if definition.aggregation != "sample_mean"
    ]
    groups = (
        _group_by_source(samples) if grouped_definitions else group_by_problem(samples)
    )
    declared = {
        metric: _interval(
            groups,
            metric,
            correctness=metric_set.correctness,
            requires_answer_region=metric_set.requires_answer_region,
            resamples=resamples,
            seed=seed,
            truncation_policy=truncation_policy,
            native=metric_set.native(metric),
        )
        for metric in metric_set.metrics
        if _measured(samples, metric, requires_answer_region=metric_set.requires_answer_region)
    }
    return TaskSummary(
        task=task,
        problems=len(groups),
        # Source variants are fixed benchmark inputs, not repeated samples of
        # one prompt. Grouped native metrics currently require each declared
        # variant exactly once, so their stochastic rollout count is one even
        # though an outer source group contains several rows.
        group_size=(
            1
            if grouped_definitions and groups
            # The largest ordinary group observed, not the modal one: a rollout
            # lost to an engine error must not silently rewrite the declared
            # group size used to interpret avg@k.
            else max((len(group) for group in groups), default=0)
        ),
        metric_set=metric_set,
        answer_format=answer_format,
        protocols=protocol_records(samples),
        stop_reasons=stop_reasons(samples),
        failure_cases=case_counts(
            samples, answer_format=answer_format, truncation_policy=truncation_policy
        ),
        comparison_timeout_rate=comparison_timeout_rate(samples),
        format_not_evaluable_rate=format_not_evaluable_rate(samples),
        **declared,
        **_k_intervals(
            groups,
            metric_set=metric_set,
            measured=declared,
            resamples=resamples,
            seed=seed,
            truncation_policy=truncation_policy,
        ),
    )


def _measured(
    samples: Sequence[SampleResult],
    metric: str,
    *,
    requires_answer_region: Sequence[str] = (),
) -> bool:
    """Whether any rollout carries a value for this metric.

    A taskset's own declared metrics arrive in `sample.metrics`; the ladder's
    observations are fields on the row. A metric found in neither was not
    measured, and the difference between "measured nought" and "not measured" is
    the whole of D-107.

    A rollout excluded from this metric's denominator does not count as carrying
    it, and that matters at exactly one boundary: a taskset where *every*
    rollout ran into the cap mid-reasoning has no `format_ok` denominator at
    all, so the column is **absent** rather than reported as zero. Zero there
    would say the checkpoint formatted nothing, when what happened is that the
    question was never reachable -- the same distinction an unbindable published
    grader already draws.
    """
    if metric in requires_answer_region:
        samples = [sample for sample in samples if not sample.format_not_evaluable]
    return any(metric in sample.metrics or hasattr(sample, metric) for sample in samples)


def format_not_evaluable_rate(samples: Sequence[SampleResult]) -> float:
    """Fraction of `samples` that reached no answer region for a format lens.

    A flat rate over rollouts and no interval, for the reason
    `comparison_timeout_rate` carries neither: it counts what happened in this
    run rather than estimating something a resample could move. It is reported
    beside the columns it qualifies so a reader can see how much of a
    `format_ok` was computed over -- a 40% rate beside a 100% `format_ok` says
    three fifths of the rollouts formatted an answer and the rest never got to
    try, which is a different sentence from either number alone.
    """
    if not samples:
        return 0.0
    return sum(sample.format_not_evaluable for sample in samples) / len(samples)


def summarize(
    manifest: RunManifest,
    samples: Sequence[SampleResult],
    *,
    metric_sets: Mapping[str, MetricSet] | None = None,
    answer_formats: Mapping[str, str | None] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> RunSummary:
    """The whole run: one `TaskSummary` per taskset, bound to the manifest.

    The manifest travels with the numbers rather than beside them because a
    summary read months later has to answer "may this be compared with that?"
    without a second file.

    `metric_sets` is how a taskset's declaration reaches its numbers: taskset
    name to the metric set it reports. Anything absent gets the ladder, so a
    caller with no opinion -- and every suite that has none to express -- keeps
    the five rungs it always had.

    `answer_formats` is the same shape and reaches only the A--T partition:
    taskset name to the reference notation its strict lens reads. Anything
    absent states no strict prediction for that taskset, which is the direction
    a missing declaration has to fail in -- the boxed column asserted over a
    taskset nobody said was boxed would manufacture findings.

    **The truncation policy is read off the manifest rather than passed in**, and
    that is deliberate: it is the field that says what these numbers mean, so a
    summary computed under one policy and manifested under another is the single
    inconsistency this function must be unable to produce. Re-summarising an
    older run's samples with its own manifest therefore reproduces that run.

    **An E42-era manifest is refused**, and this is the boundary the refusal
    belongs at: it is the one function every summary in this repository is
    written by, so a re-summarisation cannot reach around it. See
    `assert_scoreable`.

    The scorer version is stamped here rather than accepted from the caller, for
    the same reason the truncation policy is read rather than passed: it names
    the code that produced these numbers, and code cannot be told what it is.
    **`pass_at_k` is stamped on the same terms and for the same reason**: it
    names which k columns exist, which is a fact about what this function just
    computed rather than a declaration a caller may make about it.
    """
    assert_scoreable(manifest)
    for sample in samples:
        validate_answer_scorer_records(manifest.pins, sample.task, sample.protocols)
    declared = metric_sets or {}
    formats = answer_formats or {}
    tasks = [
        summarize_task(
            task,
            task_samples,
            metric_set=declared.get(task, LADDER),
            answer_format=formats.get(task),
            resamples=resamples,
            seed=seed,
            truncation_policy=manifest.truncation_policy,
        )
        for task, task_samples in by_taskset(samples).items()
    ]
    return RunSummary(
        manifest=manifest.model_copy(
            update={"scorer_version": SCORER_VERSION, "pass_at_k": _declared_k(tasks)}
        ),
        tasks=tasks,
    )


def _declared_k(tasks: Sequence[TaskSummary]) -> str:
    """Which k each taskset reported its headline at, in one comparable string.

    Sorted by taskset and joined deterministically for the reason
    `cli._relaxation_bases` is: string equality is what `comparable_with` needs,
    and a field whose value depended on the order the runner walked the suite
    would refuse two identical runs.

    A taskset declaring no k set contributes nothing rather than `=none`, so a
    run over constraint-checked tasksets alone records the empty string -- the
    same value a summary written before D-06 carries. Those two states are
    deliberately one: neither reports a k column, so there is nothing for a
    reader to be told apart.
    """
    return ",".join(
        f"{task.task}={'+'.join(str(k) for k in task.metric_set.pass_at_k)}"
        for task in sorted(tasks, key=lambda t: t.task)
        if task.metric_set.pass_at_k
    )


def comparison_timeout_rate(samples: Sequence[SampleResult]) -> float:
    """Fraction of `samples` whose answer comparison hit its time bound.

    A flat rate over rollouts, not the ladder's mean over problems of means over
    rollouts. The rungs are averaged that way because they estimate how a
    checkpoint would do on a fresh problem, and the problem is the unit being
    generalised over. This estimates nothing: it counts how often the verifier
    gave up in this run, and the rollout is the thing that either saw a timeout
    or did not. For the balanced groups every taskset here produces the two
    agree numerically anyway; they differ in what they claim, and only one of
    those claims is true of a harness fault.

    It carries no interval for the same reason. A confidence interval says the
    number would move if the experiment were repeated, and would invite reading
    a wide one as "this might really be fine". The timeouts happened; a rerun on
    a busier node would produce more, not fewer.

    **The flag is read, never recomputed, and that is not an oversight.** Every
    other number in a summary can be re-derived by rescoring the rollouts, and
    this one cannot: whether a comparison hits its bound depends on what else the
    machine was doing. Rescoring `res297-preflight-math-extended-003` twice with
    identical code flagged 12 rollouts and then 13. So the flag recorded while
    the run was scored is the evidence, and a rescore that disagrees with a
    stored rate has not found a bug in either.
    """
    if not samples:
        return 0.0
    return sum(sample.comparison_timeout for sample in samples) / len(samples)


def relaxation_gap(task: TaskSummary) -> tuple[str, float] | None:
    """Lenient minus the anchor it relaxed, with that anchor named.

    This is the quantity that moves between a pretrained checkpoint and a
    post-trained one -- completions that reached an answer the grader accepts but
    not in the form it requires -- so it is computed once here rather than left
    for a reader to subtract.

    **It returns the base as well as the number, and callers must render both.**
    Under D-108 the base varies by taskset, so "the gap" on MATH-500 is measured
    against a published grader and on AIME against prime-rl's verifier. Handing
    back a bare float would let a report put two different measurements in one
    column, which is precisely the objection the user was shown and the reason
    the base is mandatory on every surface.

    `None` where the taskset reports no such pair: a native-metric taskset has no
    protocol gap to subtract, and a
    taskset whose base was never bound has nothing to subtract from.
    """
    record = task.protocol("lenient")
    base = record.relaxes if record is not None and record.relaxes else "strict"
    if "lenient" not in task.metrics or base not in task.metrics:
        return None
    return base, task.metrics["lenient"].value - task.metrics[base].value


def _interval(
    groups: Sequence[Sequence[SampleResult]],
    metric: str,
    *,
    correctness: Sequence[str],
    requires_answer_region: Sequence[str],
    resamples: int,
    seed: int,
    truncation_policy: TruncationPolicy = DEFAULT_TRUNCATION_POLICY,
    native: NativeMetric | None = None,
) -> Interval:
    """One metric's hierarchical bootstrap, over the rollouts that could produce it.

    A rollout `scored` returns `None` for is left out of its problem's list
    rather than entered as a zero, and a problem every one of whose rollouts was
    left out drops from the resample -- `bootstrap_interval` already skips an
    empty group, which is the same rule applied one level up. That keeps the
    exclusion inside the metric it belongs to: the same problem is still
    resampled at every other column.
    """
    metric_groups = (
        _source_metric_groups(
            groups,
            metric,
            native=native,
            correctness=correctness,
            requires_answer_region=requires_answer_region,
            truncation_policy=truncation_policy,
        )
        if native is not None and native.aggregation != "sample_mean"
        else _metric_groups(
            groups,
            metric,
            correctness=correctness,
            requires_answer_region=requires_answer_region,
            truncation_policy=truncation_policy,
        )
    )
    return _bootstrap(
        metric_groups,
        resamples=resamples,
        seed=seed,
    )


def _group_by_source(samples: Sequence[SampleResult]) -> list[list[SampleResult]]:
    """Rows grouped by their benchmark source identity, variants fixed inside.

    Every row must name both identities once a taskset declares source-group
    aggregation. Ordering is deterministic, while the metric-specific complete
    variant check remains in `_source_metric_groups` because static and dynamic
    metrics intentionally select different declared sets.
    """
    groups: dict[str, list[SampleResult]] = {}
    for sample in samples:
        if sample.source_group_id is None:
            raise ValueError(
                f"{sample.task}: source_group_id is required for source-group metrics"
            )
        if sample.source_variant_id is None:
            raise ValueError(
                f"{sample.task}: source_variant_id is required for source-group metrics"
            )
        groups.setdefault(sample.source_group_id, []).append(sample)
    return [
        sorted(group, key=lambda sample: sample.source_variant_id or "")
        for _, group in sorted(groups.items())
    ]


def _source_metric_groups(
    groups: Sequence[Sequence[SampleResult]],
    metric: str,
    *,
    native: NativeMetric,
    correctness: Sequence[str],
    requires_answer_region: Sequence[str],
    truncation_policy: TruncationPolicy,
) -> list[list[float]]:
    """Collapse each complete fixed variant set to one outer-group scalar."""
    expected = set(native.source_variants)
    collapsed: list[list[float]] = []
    for group in groups:
        selected: dict[str, SampleResult] = {}
        for sample in group:
            variant = sample.source_variant_id
            if variant not in expected:
                continue
            if variant in selected:
                raise ValueError(
                    f"{sample.task}: duplicate source variant {variant!r} in group "
                    f"{sample.source_group_id!r} for metric {metric!r}"
                )
            selected[variant] = sample
        missing = [variant for variant in native.source_variants if variant not in selected]
        if missing:
            task = group[0].task if group else "taskset"
            source = group[0].source_group_id if group else "unknown"
            raise ValueError(
                f"{task}: incomplete source variants in group {source!r} for metric "
                f"{metric!r}; missing {', '.join(missing)}"
            )
        values: list[float] = []
        for variant in native.source_variants:
            value = scored(
                selected[variant],
                metric,
                correctness=correctness,
                requires_answer_region=requires_answer_region,
                truncation_policy=truncation_policy,
            )
            if value is None:
                raise ValueError(
                    f"{selected[variant].task}: native metric {metric!r} was omitted for "
                    f"source group {selected[variant].source_group_id!r}, variant {variant!r}"
                )
            values.append(float(value))
        if native.aggregation == "source_group_all":
            endpoints = {native.minimum, native.maximum}
            if any(value not in endpoints for value in values):
                raise ValueError(
                    f"{group[0].task}: source_group_all metric {metric!r} requires endpoint "
                    f"verdicts {sorted(endpoints)}, got {values}"
                )
            collapsed.append(
                [native.maximum if all(value == native.maximum for value in values) else native.minimum]
            )
        else:
            collapsed.append([sum(values) / len(values)])
    return collapsed


def _metric_groups(
    groups: Sequence[Sequence[SampleResult]],
    metric: str,
    *,
    correctness: Sequence[str],
    requires_answer_region: Sequence[str],
    truncation_policy: TruncationPolicy,
) -> list[list[float]]:
    """What each problem's rollouts scored at one metric, ready to resample.

    Separated from `_interval` so that a k column reads the *same* per-sample
    values its avg@k does, rather than a second traversal that could apply the
    truncation policy differently. The policy is decided in `scored` and this is
    the only door to it.
    """
    return [
        [
            value
            for sample in group
            if (
                value := scored(
                    sample,
                    metric,
                    correctness=correctness,
                    requires_answer_region=requires_answer_region,
                    truncation_policy=truncation_policy,
                )
            )
            is not None
        ]
        for group in groups
    ]


def _bootstrap(
    groups: Sequence[Sequence[float]],
    *,
    statistic: Statistic = mean_of_group,
    resamples: int,
    seed: int,
) -> Interval:
    """One statistic's hierarchical bootstrap over already-scored groups."""
    value, low, high = bootstrap_interval(
        groups, statistic=statistic, resamples=resamples, seed=seed
    )
    return Interval(value=value, low=low, high=high, confidence=DEFAULT_CONFIDENCE)


def _k_intervals(
    groups: Sequence[Sequence[SampleResult]],
    *,
    metric_set: MetricSet,
    measured: Mapping[str, Interval],
    resamples: int,
    seed: int,
    truncation_policy: TruncationPolicy,
) -> dict[str, Interval]:
    """The declared `pass@k` and `pass^k` columns, over the headline's own values.

    Nothing at all where the metric set declares no k set, which keeps every
    report written before D-06 exactly as it was, and nothing where neither
    anchor produced a number, since there is no headline for them to describe.

    **A k above the rollouts this taskset drew is left out**, and that is D-107
    rather than an optimisation: `pass@32` on a group of four is not zero and is
    not low, it is unasked, and `group_size` on the summary beside these columns
    is what a reader recovers the reason from. The threshold is read off the
    groups this metric actually scored rather than off the declared group size;
    for a correctness anchor the two are the same number, because no correctness
    metric drops a rollout from its denominator.

    Every column is drawn at the same seed as the headline it is computed from,
    so `pass@1` is that column's interval and not a second estimate of it.
    """
    metric = headline_metric(measured, metric_set)
    if metric is None:
        return {}
    scored_groups = _metric_groups(
        groups,
        metric,
        correctness=metric_set.correctness,
        requires_answer_region=metric_set.requires_answer_region,
        truncation_policy=truncation_policy,
    )
    drawn = max((len(group) for group in scored_groups), default=0)
    return {
        name: _bootstrap(scored_groups, statistic=statistic, resamples=resamples, seed=seed)
        for name, k, statistic in k_columns(metric_set)
        if k <= drawn
    }
