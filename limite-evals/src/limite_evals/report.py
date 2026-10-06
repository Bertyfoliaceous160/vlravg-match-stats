"""Where a run's artifacts land, and what the markdown has to say to stand alone.

Decision 14 fixes the layout and forbids a run from overwriting an older one, so
`write_run` refuses an existing directory outright instead of merging into it. A
rerun that lands on top of its predecessor destroys the only copy of numbers
someone may already have quoted, and it does so silently.

The markdown is written for a reader with no local files open. Every table of
numbers is therefore followed by the pins, the fingerprint, the rendering
profile, the template hash and the format policy: those are what decide whether
these numbers may be compared with another run's at all, and a number quoted
without them is diagnostic only rather than merely imprecise.

A run also writes as it goes, because a suite is now long enough that losing its
first eight tasksets to a failure in the ninth costs an hour of GPU time. That
turns one guarantee into two: the rollouts of a finished taskset are on disk
before the next one starts, and the directory says out loud that it is not a
finished run. `status.json` is the thing that says so, and `metrics/summary.json`
stays what it has always been -- the file a complete run has and an incomplete
one does not.

**A taskset's licence and caveat are part of its number, so they are rendered
next to it.** A dataset admitted on the condition that its licence stays visible,
or with a contamination risk accepted rather than resolved, has that condition
discharged only if the reader holding the score also holds the condition. Both
therefore land inside the taskset's own section, above its table -- not in a
run-level footer, which a reader quoting one taskset out of nine would leave
behind, and not in a document they would have to go and open.

They are read from `suites` rather than from `TaskSummary` because they belong to
the taskset specification and the result schema has nowhere to put them: the
extras on `TaskSummary` are typed `Interval`, and a licence is not a measurement.
The lookup is by `(suite, taskset)` from the manifest and yields nothing when the
pair is unknown, so a summary naming a taskset this checkout does not define
still renders instead of failing at the last step of the run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

# Selected before `pyplot` is imported, because the backend is fixed at import
# time. Reports are written from batch jobs and from continuous integration,
# neither of which has a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from limite_evals import suites
from limite_evals.aggregate import (
    TRACKED_RUNGS,
    case_reconciliation,
    headline_metric,
    k_absence_reason,
    k_columns,
    relaxation_gap,
)
from limite_evals.checkpoint import atomic_json
from limite_evals_core.schema import (
    CaseCount,
    Fingerprint,
    ProtocolRecord,
    RunManifest,
    RunSummary,
    SampleResult,
    Sampling,
    TaskSummary,
)

#: Local root from decision 14. On the cluster the caller passes
#: `/work/$USER/evals` instead; the per-run layout below is the same either way.
DEFAULT_ROOT = Path("outputs/eval")

SUBDIRECTORIES = ("metrics", "plots", "reports", "artifacts")

#: What the run directory holds and what it still owes, at the run root where a
#: reader lands first. It is a positive statement rather than an inference from
#: which files are present, because "three of nine tasksets" and "a complete run
#: over three tasksets" are indistinguishable in the numbers themselves.
STATUS_FILE = "status.json"

#: The first line of a stopped run's report, so the document says what it is
#: wherever it is read rather than only where it was written.
INCOMPLETE_NOTE = (
    "> **This run did not finish.** The tasksets below are the ones that completed; "
    "`status.json` in the run directory lists the ones it still owed. Every number "
    "here is that taskset's own, and the manifest describes only these tasksets.\n\n"
)

#: Ember Light, this project's chart palette. It replaced Palette B when the
#: protocol metric set arrived: Palette B holds five colours and that set draws
#: six columns, so `format_ok` and `truncated` were assigned the same one and
#: stood side by side as a single indistinguishable block. Eight colours is not
#: decoration, it is the headroom that made the collision impossible to
#: reintroduce silently -- `test_no_declared_metric_set_draws_two_bars_alike`
#: holds it.
#:
#: The muted shade is spent on `permissive`, which is a diagnostic upper bound
#: rather than a reported score, so the rungs a reader actually compares stay the
#: most legible. Adjacency is deliberate too: the draw order interleaves warm and
#: cool, so no two neighbouring bars are near-hues of each other.
RUNG_COLOURS = {
    "strict": "#3a6080",
    "exact": "#3a6080",
    "reference": "#b84c30",
    "lenient": "#7a6820",
    "permissive": "#706070",
    "format_ok": "#386858",
    "truncated": "#905050",
}

#: How the instrument era is worded for a reader who was not there for E42. The
#: label carries the consequence rather than the code name: a reader comparing
#: two runs has to be able to see, from the table alone, that one of them was
#: scored over text with its reasoning delimiters removed.
INSTRUMENT_ERA_LABELS = {
    "corrected": "corrected (completions carry their reasoning delimiters)",
    "e42": "E42 (every `<think>` and `</think>` was deleted before the grader read it)",
}

#: How the stop-reason breakdown is worded for a reader who has not read
#: vLLM's API. The order is the order the rows are printed in.
STOP_CLASS_LABELS = {
    "eos": "ended itself (end-of-sequence token)",
    "stop_string": "halted by a stop string",
    "stop_token": "halted by a stop token id",
    "length": "ran out of generation budget",
    "unknown": "engine reported no finish reason",
}

#: Ember Light in the order a metric with no entry above draws in. Keying colours
#: by name is right for the ladder, where the assignment is deliberate, and
#: impossible for a metric set this module has never heard of -- which would
#: otherwise be a `KeyError` at plot time, after the run. Drawing the leftovers
#: from here keeps every chart inside the one approved palette; `_colours` is
#: what skips the shades a rung on the same chart already claimed. The two no
#: rung claims at all lead, so an unnamed set drawn on its own opens on colours
#: the ladder never uses.
EMBER_LIGHT = (
    "#946030",
    "#4a6830",
    "#3a6080",
    "#b84c30",
    "#7a6820",
    "#386858",
    "#905050",
    "#706070",
)
BAR_EDGE = "#3a6080"
ERROR_BAR = "#b84c30"
WHITE = "#FFFFFF"

#: The divider between two tasksets, and how far up the axes it runs.
#:
#: A group of six bars is wider than the gap to the next taskset, so the eye
#: reads across a boundary before it reads within a group -- which is the wrong
#: grouping, because a metric is only comparable with the same metric on the
#: same taskset. The height stops short of the legend at the top; a line through
#: it reads as part of it.
SEPARATOR = "#706070"
SEPARATOR_HEIGHT = 0.88

#: The share of a taskset's slot left empty at its edges. Bar width is derived
#: from it rather than fixed, because a fixed width is a gutter that shrinks as
#: metrics are added: at the protocol set's six columns the old 0.16 left the
#: groups 0.04 apart, which is not a gap a divider can sit in.
GROUP_GUTTER = 0.18


def reserve(run_id: str, *, root: Path = DEFAULT_ROOT) -> Path:
    """Claim `root/<run_id>/` before anything is spent, and return it.

    The claim is the `mkdir` itself rather than a check followed by one, so two
    runs racing on the same identifier cannot both believe they own it. It is
    made first because the serve step writes the chat template and the engine log
    into `artifacts/` long before there is a summary to write beside them --
    discovering the collision afterwards would mean discovering it after the GPU
    time.
    """
    run_dir = Path(root) / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"run directory already exists and a run never overwrites an older one: {run_dir}"
        ) from None
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir()
    return run_dir


def write_status(
    run_dir: Path, *, completed: Sequence[str], pending: Sequence[str], complete: bool = False
) -> None:
    """Which tasksets this directory holds, which it still owes, and whether it finished.

    `complete` is a recorded fact rather than `pending == []`: the last taskset
    finishes before the summary is written, so an emptied queue is not yet a
    finished run and a rule that read it as one would mislabel a directory that
    died between the two.
    """
    atomic_json(
        run_dir / STATUS_FILE,
        {
            "run_id": run_dir.name,
            "complete": complete,
            "completed": list(completed),
            "pending": list(pending),
        },
    )


def write_transport_retries(run_dir: Path, retries: Mapping[str, object]) -> None:
    """How many rollout requests had to be resent, and after which transport fault.

    In `artifacts/` rather than `metrics/` because it is a property of the wire
    and not a number anyone compares: no reported figure moves when it is
    non-zero. It is written on every run, zeros included, so that a reader can
    tell a run whose transport was clean from one written by an instrument old
    enough not to record this at all -- and it is rewritten after each taskset
    so a run that dies mid-suite still says what the transport was doing on the
    way down.
    """
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "transport-retries.json").write_text(json.dumps(dict(retries), indent=2) + "\n")


def write_partial(run_dir: Path, summary: RunSummary) -> None:
    """The numbers from the tasksets that did finish, for a run that did not.

    Written under its own name because `metrics/summary.json` is what marks a run
    finished, and a partial run that could pass for a finished one would trade a
    loud failure for a quiet wrong number.

    The manifest inside it describes only the tasksets present -- their dataset
    revisions, their largest effective generation budget, their rendering -- so
    it is true of the data beside it, and `comparable_with` refuses it against a
    full run of the same suite mechanically, exactly as `--limit` already is.

    The markdown says so in its first line as well as in its filename, because a
    report is the artifact that gets pasted somewhere else, and there it would
    otherwise read as a complete run over however many tasksets it happens to
    hold.
    """
    (run_dir / "metrics" / "partial-summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    (run_dir / "reports" / "partial-summary.md").write_text(INCOMPLETE_NOTE + render_markdown(summary))


def write_run(
    summary: RunSummary,
    samples: Sequence[SampleResult],
    *,
    root: Path = DEFAULT_ROOT,
    persist_samples: bool = True,
) -> Path:
    """Write the whole run under `root/<run_id>/` and return that directory.

    Raises `FileExistsError` if that run already has a summary. Decision 14 makes
    that a hard rule rather than a flag: there is no argument for overwriting a
    run, and offering one would mean the guard is off in the situation where it
    matters, which is a hurried rerun under the same identifier.

    A directory `reserve` already claimed is written into, since that is the same
    run rather than an older one; the summary file is what marks a run finished.

    Live evaluations pass `persist_samples=False` after validating their durable
    journal's exact coverage. Reporting then leaves those bytes intact, even if
    interrupted. Offline reporting/rescoring retains the default sample write.
    """
    run_dir = Path(root) / summary.manifest.run_id
    if (run_dir / "metrics" / "summary.json").exists():
        raise FileExistsError(
            f"run directory already exists and a run never overwrites an older one: {run_dir}"
        )
    for name in SUBDIRECTORIES:
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    if persist_samples:
        write_samples(run_dir / "metrics" / "samples.jsonl", samples)
    (run_dir / "reports" / "summary.md").write_text(render_markdown(summary))
    write_ladder_plot(run_dir / "plots" / "ladder.png", summary)
    validated = RunSummary.model_validate_json(summary.model_dump_json())
    atomic_json(run_dir / "metrics" / "summary.json", validated.model_dump(mode="json"), sort_keys=False)
    write_status(run_dir, completed=[task.task for task in summary.tasks], pending=[], complete=True)
    return run_dir


def write_samples(path: Path, samples: Sequence[SampleResult], *, mode: str = "w") -> None:
    """Every rollout, one JSON object per line.

    Per-sample rather than aggregated only, because the summary is reproducible
    from these rows and an interval nobody can recompute is not evidence.

    Offline report and rescore callers use this writer. Live evaluations use
    the durable checkpoint journal and final reporting preserves its bytes.
    """
    with path.open(mode) as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")


def write_ladder_plot(path: Path, summary: RunSummary) -> None:
    fig = ladder_figure(summary)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def ladder_figure(summary: RunSummary) -> Figure:
    """Each taskset's declared metrics, returned rather than saved so it can be inspected.

    Whiskers are drawn on the headline alone -- `strict` for the ladder. It is
    the number the report leads with, and a set of whiskers on every bar would
    bury the metric-to-metric steps the chart exists to show.
    """
    tasks = summary.tasks
    metrics = _plotted_metrics(summary)
    colours = _colours(metrics)
    fig, ax = plt.subplots(figsize=(max(7.0, 2.5 + 1.7 * len(tasks)), 4.6), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    width = (1.0 - GROUP_GUTTER) / len(metrics)
    centre = (len(metrics) - 1) / 2
    for offset, metric in enumerate(metrics):
        drawn = [(index, task) for index, task in enumerate(tasks) if metric in task.metrics]
        positions = [index + (offset - centre) * width for index, _ in drawn]
        intervals = [task.metrics[metric] for _, task in drawn]
        headline = [metric == task.metric_set.headline for _, task in drawn]
        yerr = None
        if any(headline):
            yerr = [
                [
                    _plot_value(task, metric, i.value) - _plot_value(task, metric, i.low) if h else 0.0
                    for (_, task), i, h in zip(drawn, intervals, headline)
                ],
                [
                    _plot_value(task, metric, i.high) - _plot_value(task, metric, i.value) if h else 0.0
                    for (_, task), i, h in zip(drawn, intervals, headline)
                ],
            ]
        ax.bar(
            positions,
            [_plot_value(task, metric, interval.value) for (_, task), interval in zip(drawn, intervals)],
            width=width,
            label=metric,
            color=colours[metric],
            edgecolor=BAR_EDGE,
            linewidth=0.5,
            yerr=yerr,
            ecolor=ERROR_BAR,
            capsize=3,
        )
        # Printed on every bar, because a reader quoting a number off a chart
        # should not have to estimate it against a gridline. The headline clears
        # its own whisker rather than the bar, or the label lands inside the cap.
        for position, (_, task), interval, whiskered in zip(positions, drawn, intervals, headline):
            ax.annotate(
                _metric_value(task, metric, interval.value),
                (
                    position,
                    _plot_value(task, metric, interval.high if whiskered else interval.value),
                ),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=6.5,
                rotation=90,
            )

    # Behind the bars, so a divider never cuts one, and inside the axes rather
    # than between the tick labels, which a reader does not scan for.
    for boundary in range(1, len(tasks)):
        ax.axvline(
            boundary - 0.5,
            ymax=SEPARATOR_HEIGHT,
            color=SEPARATOR,
            linewidth=0.8,
            alpha=0.35,
            zorder=0,
        )

    ax.set_xticks(list(range(len(tasks))))
    ax.set_xticklabels([_task_label(task) for task in tasks])
    ax.set_ylim(0.0, 1.30)
    ax.set_ylabel(
        "declared metric range (normalized)"
        if any(task.metric_set.native_metrics for task in tasks)
        else "fraction of problems (avg@k)"
    )
    # Three short lines rather than two long ones: a single-taskset chart is
    # narrow, and a subtitle wider than the axes is silently clipped at the edge.
    ax.set_title(
        f"{summary.manifest.run_id} — {summary.manifest.stage} / {summary.manifest.profile}\n"
        f"{_regime(summary)}\n"
        f"{_headline_note(summary)}; whiskers show its 95% interval",
        fontsize="medium",
    )
    ax.legend(frameon=False, ncols=len(metrics), loc="upper center", fontsize="small")
    fig.tight_layout()
    return fig


def _plotted_metrics(summary: RunSummary) -> list[str]:
    """Every metric the run's metric sets declare, in the order they declare it.

    Read off `metric_set.metrics` rather than off the numbers each taskset
    actually produced, and that distinction is the whole point: a protocol that
    could not be bound is *absent* from `task.metrics`, so ordering by what was
    measured put `exact` wherever the first taskset that happened to have one
    fell. On a math suite led by AIME 2024 -- which has no published grader --
    the headline was drawn last, to the right of `truncated`. Declaring the
    order fixes each column's slot for the whole chart no matter which tasksets
    filled it.

    A taskset that does not report a declared metric still has no bar in that
    slot, rather than a zero, which would read as a measured nought.
    """
    order: dict[str, None] = {}
    for task in summary.tasks:
        order.update(dict.fromkeys(task.metric_set.metrics))
    return list(order)


def _colours(metrics: Sequence[str]) -> dict[str, str]:
    """One colour per metric on this chart, no two of them the same.

    The named assignment comes first and is taken as given -- `strict` is that
    blue in every chart this repository draws, and a run that mixed metric sets
    must not shift it. What is left over is handed to the metrics no rung names,
    skipping every shade already spoken for: cycling by position instead was the
    defect, because `EMBER_LIGHT[5]` is exactly `format_ok`'s colour, so a
    constraint checker drawn beside the ladder repeated it.

    A set longer than the palette would still wrap and repeat. That is left to
    fail loudly in `test_no_declared_metric_set_draws_two_bars_alike` rather than
    papered over with a generated colour, since eight is already more columns
    than a reader can tell apart and the answer at that point is a second chart.
    """
    assigned = {metric: RUNG_COLOURS[metric] for metric in metrics if metric in RUNG_COLOURS}
    spare = [colour for colour in EMBER_LIGHT if colour not in set(assigned.values())]
    unnamed = [metric for metric in metrics if metric not in assigned]
    for position, metric in enumerate(unnamed):
        assigned[metric] = spare[position % len(spare)]
    return assigned


def _headline_note(summary: RunSummary) -> str:
    """What the chart leads with, said in terms true of every taskset drawn.

    A run can mix tasksets whose headline was measured with tasksets whose was
    not, and a caption asserting the headline for all of them would be false of
    some. Naming the shortfall is the honest caption.
    """
    if not summary.tasks:
        return "the headline"
    headlines = {task.metric_set.headline for task in summary.tasks}
    if len(headlines) == 1:
        headline = summary.tasks[0].metric_set.headline
        missing = [task.task for task in summary.tasks if headline not in task.metrics]
        if not missing:
            return f"{headline} is the headline"
        return f"{headline} is the headline, and is unavailable for {', '.join(missing)}"

    missing = [
        f"{task.task} ({task.metric_set.headline})"
        for task in summary.tasks
        if task.metric_set.headline not in task.metrics
    ]
    note = "each taskset's first declared metric is its headline"
    if missing:
        note += f", and is unavailable for {', '.join(missing)}"
    return note


def _task_label(task: TaskSummary) -> str:
    """The taskset, the k its numbers are averaged over, and its relaxation base.

    The group size belongs on the axis rather than in a caption: `avg@32` on
    thirty AIME problems and `avg@1` on 1319 GSM8K ones sit side by side in the
    same chart, and nothing else on it says they are different statistics.

    The relaxation base belongs there for a stronger reason. D-108 permits the
    base to differ per taskset, so one chart can carry a `lenient` bar measured
    against a published grader beside a `lenient` bar measured against prime-rl's
    verifier -- and the legend, which is per chart rather than per taskset,
    cannot say so. No surface may show a relaxation without its base, and on this
    surface the axis label is the only place the base is per taskset.
    """
    label = f"{task.task}\n{task.problems} problems, avg@{task.group_size}"
    record = task.protocol("lenient")
    if record is not None and record.relaxes:
        label += f"\nrelaxes {record.relaxes}"
    return label


def _regime(summary: RunSummary) -> str:
    """How the rollouts were drawn, spelled out on the chart itself.

    Read without this, a ladder chart is ambiguous in exactly the way that
    matters: sampling at temperature 0.6 and decoding greedily produce the same
    picture and are not the same measurement.
    """
    sampling = summary.manifest.sampling
    if not sampling.temperature:
        return "greedy decoding"
    return f"sampled at temperature {sampling.temperature:g}, top-p {sampling.top_p:g}"


def render_markdown(summary: RunSummary) -> str:
    """The run as a document that answers "may I quote this?" without another file."""
    manifest = summary.manifest
    lines = [
        f"# Evaluation run `{manifest.run_id}`",
        "",
        f"Checkpoint `{manifest.checkpoint}` at the **{manifest.stage}** stage, suite "
        f"`{manifest.suite}`, rendered with the **{manifest.profile}** profile and scored at "
        f"the **{manifest.format_policy}** format policy. Produced {manifest.created_at}.",
        "",
    ]
    for note in _metric_notes(summary):
        lines += [note, ""]
    if truncation := _truncation_note(summary):
        lines += [truncation, ""]
    for task in summary.tasks:
        lines += _task_section(task, manifest)
    lines += _comparability_section(manifest)
    return "\n".join(lines) + "\n"


def _metric_notes(summary: RunSummary) -> list[str]:
    """One paragraph per metric set the run reports, in first-appearance order.

    A run whose tasksets all report the ladder gets exactly the ladder's own
    paragraph, which is what every `math-standard` report has always said. A
    suite that mixes metric sets gets one paragraph each, rather than a single
    sentence that would be true of neither.
    """
    notes: dict[str, str] = {}
    for task in summary.tasks:
        notes.setdefault(task.metric_set.name, task.metric_set.note)
    return [note for note in notes.values() if note]


def _truncation_note(summary: RunSummary) -> str:
    """What a completion that ran out of budget scored, in the run's own terms.

    Rendered from the manifest rather than carried in `MetricSet.note`, because
    the note is a property of what a taskset reports and this is a property of
    how the run was aggregated. The same metric set under the two policies
    describes two different quantities, and a reader who quotes a column has to
    be told which without opening the comparability table.

    **A run that truncated nothing says nothing about it**, under either policy.
    The policy decides what a truncated completion scores, so where there was no
    truncated completion it decided nothing, and a paragraph about it would
    describe the run in terms of a distinction it was never subject to. The
    policy is still stated among the comparability facts, because two runs that
    resolved it differently must not compare equal even where one of them never
    exercised it.
    """
    if all(task.metrics["truncated"].value == 0.0 for task in summary.tasks if "truncated" in task.metrics):
        return ""
    if summary.manifest.truncation_policy == "fail":
        return (
            "**Truncated completions score wrong here.** A completion that ran out of "
            "generation budget fails every correctness column whatever its text says, so no "
            "number below is raised by an answer that survives only because the completion "
            "was cut off mid-reasoning. It is counted in the truncation rate and is never "
            "dropped, and the rate is therefore how far these numbers are a lower bound."
        )
    return (
        "**Truncated completions are scored on the text they produced.** A completion that "
        "ran out of generation budget is graded on what it wrote before the budget ended, "
        "which means a column can be raised by reasoning that never reached a conclusion -- "
        "most of all `permissive`, whose waivers read the answer from anywhere in the text. "
        "It is counted in the truncation rate and is never dropped, and the rate is "
        "therefore how much of each number rests on unfinished completions."
    )


def _task_section(task: TaskSummary, manifest: RunManifest) -> list[str]:
    """The taskset's own page, widened by exactly what it has to say.

    The anchor column appears only where protocols were bound. A ladder run has
    no anchors to name, and neither does a constraint-checked taskset, which has
    no answer for an anchored grader to check; adding an all-`—` column to either
    would move every report this repository has already quoted without telling a
    reader anything, and decision 16's stop rule exists to prevent exactly that
    kind of silent re-presentation.

    The licence and the caveat sit above the table, inside the section holding
    the number, so a reader who quotes one taskset out of nine carries the
    condition the dataset was admitted on along with the score.
    """
    metric_set = task.metric_set
    anchored = bool(task.protocols)
    identified = anchored or bool(metric_set.native_metrics)
    lines = [f"## {task.task}", "", _headline(task), ""]
    for note in _dataset_notes(manifest.suite, task.task):
        lines += [note, ""]
    if identified:
        lines += [f"| {metric_set.row_label} | Reproduces | Value | Interval |", "|---|---|---|---|"]
    else:
        lines += [f"| {metric_set.row_label} | Value | Interval |", "|---|---|---|"]
    for name in metric_set.metrics:
        lines.append(_metric_row(task, name, identified=identified))
    lines.append("")
    lines += _comparison_timeout_note(task)
    lines += _format_not_evaluable_note(task)
    lines += _format_ok_note(task, manifest)
    lines += _gap_paragraph(task)
    lines += _pass_at_k_section(task)
    lines += _stop_reason_section(task)
    lines += _failure_case_section(task)
    return lines


def _failure_case_section(task: TaskSummary) -> list[str]:
    """How the rollouts were shaped, partitioned by the A--T table, and whether
    the two rungs agreed with what the table expected of each case.

    Beneath the metric table rather than inside it. Every row here is a count of
    rollouts, and the columns above are rates with intervals; more to the point,
    a taxonomy that observes the graders must not be able to arrive in the same
    table as a number a grader produced, or a later reader will quote one as the
    other.

    A taskset with no cases gets nothing -- a constraint-checked taskset has no
    answer to extract, and a summary written before the tracker existed carries
    no partition to render, and in both the absence is the honest output.
    """
    if not task.failure_cases:
        return []
    rows = [
        f"| `{count.case}` | {count.description} | {count.samples} | "
        f"{count.strict_scored} | {count.permissive_scored} | {_expected(count)} |"
        for count in task.failure_cases
    ]
    return [
        "### Failure modes",
        "",
        "Every rollout falls in exactly one case. `strict` and `permissive` are what this "
        "taskset scored under those two names -- read exactly as the columns above are, so "
        "each is the protocol where the taskset declares one and the extraction rung "
        "otherwise. They are observations; the taxonomy's prediction is the last column. A "
        "case counts the shape a completion took, so a well-formed box holding the wrong "
        "number is still `A`; correctness separates two cases only where their names say it "
        "does, `M` from `N`.",
        "",
        "| Case | Shape | Rollouts | `strict` | `permissive` | Table expects |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        *_reconciliation_paragraph(task),
    ]


def _expected(count: CaseCount) -> str:
    """What the table predicts for this case, over the lenses it predicts at all.

    A lens the taxonomy says nothing about here is left out of the cell rather
    than printed as a "no": the scoping paragraph beneath the table says which
    lens that is and why, and repeating "not `strict`" on twenty rows would read
    as twenty refusals instead of one absent claim.
    """
    yes = [name for name in TRACKED_RUNGS if getattr(count, f"{name}_expected") is True]
    if yes:
        return ", ".join(f"`{name}`" for name in yes)
    # Nothing is expected to score. Say so only about the lenses the taxonomy
    # actually spoke about: a bare "neither" on a taskset whose strict column
    # was withheld would refuse on its behalf, which is the collapse of `None`
    # into `False` this whole scoping exists to prevent.
    # Never empty: `permissive_expected` is not optional, because that column is
    # not scoped to an answer format.
    stated = [name for name in TRACKED_RUNGS if getattr(count, f"{name}_expected") is not None]
    return "neither" if len(stated) == len(TRACKED_RUNGS) else f"not `{stated[0]}`"


def _reconciliation_paragraph(task: TaskSummary) -> list[str]:
    """Whether each rung stayed inside the cases the table allows it.

    This is the whole claim the tracker makes, so it is stated in the report
    rather than left for a reader to add the column up. A rung that scored a
    rollout from a case marked "no" is the falsification, and it is named with
    its cases so the rows can be found in `samples.jsonl` by `failure_case`.

    **Nothing here changes a number above.** A disagreement says the taxonomy
    does not describe what that rung recovered; the rung's own verdict stands
    exactly as it was scored.
    """
    lines = []
    for rung in TRACKED_RUNGS:
        total, predicted, disagreeing = case_reconciliation(task, rung)
        if disagreeing is None:
            lines += [
                f"**The table states no `{rung}` prediction for this taskset.** Every case in "
                "it describes where a `\\boxed{}` answer sits, and this taskset's reference -- "
                "named in the table above -- reads something else, so the shape a completion "
                f"took and what `{rung}` recovered are unrelated here. The {total} rollouts it "
                "accepted are counted against each case as an observation, and nothing above "
                "is a claim they could have falsified.",
                "",
            ]
            continue
        if not disagreeing:
            lines += [
                f"Every one of the {total} rollouts that scored at `{rung}` fell in a case the "
                "table expects it to: the taxonomy and the column agree exactly.",
                "",
            ]
            continue
        cases = ", ".join(
            f"`{count.case}`"
            for count in task.failure_cases
            if getattr(count, f"{rung}_scored") and not getattr(count, f"{rung}_expected")
        )
        lines += [
            f"**{disagreeing} of the {total} rollouts that scored at `{rung}` came from "
            f"cases the table says it cannot accept** ({cases}), leaving {predicted} it "
            "predicted. The table describes where a `\\boxed{}` answer sits, so a rung "
            "reproducing a reference that recovers an answer some other way -- its anchor is "
            "named in the table above -- reaches answers this partition does not model. The "
            "rollouts are in `samples.jsonl`, found by their `failure_case`.",
            "",
        ]
    return lines


def _headline(task: TaskSummary) -> str:
    """The number the report leads with, or a plain statement that there is none.

    **Nothing is promoted into an absent headline.** Where the benchmark
    published no grader this repository could bind, `exact` has no number, and
    printing `reference` in its place under the same heading would hand a reader
    a leaderboard-comparable figure that is nothing of the kind. The column reads
    as absent, with the reason, and the other protocols are read from the table
    on their own terms.
    """
    name = task.metric_set.headline
    counted = f"{task.problems} problems, {task.group_size} rollouts each."
    if name in task.metrics:
        headline = task.metrics[name]
        return (
            f"{counted} Headline: **{name} {_metric_value(task, name, headline.value)}** "
            f"({_confidence(headline.confidence)} interval "
            f"{_metric_value(task, name, headline.low)}–"
            f"{_metric_value(task, name, headline.high)})."
        )
    reason = _absence_reason(task, name, task.protocol(name))
    return (
        f"{counted} **There is no {name} number for this taskset**, and no other column "
        f"stands in for it: {reason}."
    )


def _metric_row(task: TaskSummary, name: str, *, identified: bool) -> str:
    """One row, naming the artefact that produced it wherever one is bound.

    A protocol absent from `metrics` still gets a row, carrying its reason.
    Omitting it would leave a reader comparing two tasksets' tables without being
    told that one of them was never measured, which is indistinguishable from its
    having been left out by accident.
    """
    record = task.protocol(name)
    if name not in task.metrics:
        cells = ["*unavailable*", _absence_reason(task, name, record)]
    else:
        interval = task.metrics[name]
        cells = [
            _metric_value(task, name, interval.value),
            f"{_metric_value(task, name, interval.low)}–{_metric_value(task, name, interval.high)}",
        ]
    if not identified:
        return f"| `{name}` | " + " | ".join(cells) + " |"
    native = task.metric_set.native(name)
    identity = (
        record.anchor
        if record is not None and record.anchor
        else native.provenance
        if native is not None
        else "—"
    )
    if record is not None and record.relaxes and task.protocol_source(name):
        identity += f"; base: {task.protocol_source(name)}"
    return f"| `{name}` | {identity} | " + " | ".join(cells) + " |"


def _absence_reason(task: TaskSummary, name: str, record: ProtocolRecord | None) -> str:
    """Why this column has no number, in the terms of whichever absence it is.

    Three of them exist and they are not one fact in three wordings. A protocol
    that bound no artefact carries its own reason and states it. A metric that
    needed an answer region and never got one on any rollout is a *fourth*
    state, arrived at only since the reasoning delimiters survived the decode,
    and "not measured" would describe it as an oversight. Anything else was
    genuinely not measured, which is what that phrase is for.
    """
    if record is not None and record.unavailable_reason:
        return record.unavailable_reason
    native = task.metric_set.native(name)
    if native is not None and native.unavailable_reason:
        return native.unavailable_reason
    if name in task.metric_set.requires_answer_region and task.format_not_evaluable_rate == 1.0:
        return "every rollout stopped mid-reasoning, so no answer region existed for it to read"
    return "not measured"


def _gap_paragraph(task: TaskSummary) -> list[str]:
    """The relaxation gap, always naming the anchor it was measured against.

    D-108 makes the base a per-taskset fact, so a sentence saying "the gap" would
    mean one thing on MATH-500 and another on AIME. Naming it is mandatory on
    every surface, and this is one of them.

    The ladder's own sentence already named its base -- there was only ever one --
    so a run that bound no protocols keeps it verbatim. What is added for a
    protocol run is the warning that the base is not the same everywhere, which
    is a true statement only once the base can vary.
    """
    gap = relaxation_gap(task)
    if gap is None:
        return []
    base, value = gap
    if not task.protocols:
        return [
            f"The {base}-to-lenient gap is **{_points(value)}**: completions that reached "
            f"an answer the verifier accepts, but not in the form {base} requires. That gap "
            "is the format-compliance measurement, and it is the quantity that moves "
            "between a pretrained checkpoint and a post-trained one.",
            "",
        ]
    return [
        f"The {base}-to-lenient gap is **{_points(value)}**: completions that reached an "
        f"answer `{base}` would accept, but not in the form it requires. That gap is the "
        "format-compliance measurement, and it is the quantity that moves between a "
        f"pretrained checkpoint and a post-trained one. It is measured against `{base}` "
        "here, which is not the same base every taskset uses.",
        "",
    ]


def _pass_at_k_section(task: TaskSummary) -> list[str]:
    """What the same rollouts say about a smaller budget, and what they cannot say.

    Its own table beneath the metrics rather than more columns beside them. The
    columns above are one estimator at several protocols; these are several
    estimators at one protocol, and a reader who mistook the axis would compare
    a `pass^32` against a `permissive` as though they were two rungs.

    Nothing at all where the metric set declares no k set. That is every
    constraint-checked taskset -- a fraction of instructions followed is not a
    verdict a draw of `k` can be right at -- and every report written before
    D-06, which must render exactly the document it rendered then.

    **A k above the rollouts this taskset drew is printed as absent with its
    reason, not omitted from the table.** A missing row would leave a reader
    unable to tell an unasked question from a forgotten one, which is the
    distinction the whole absence rule exists to keep.
    """
    columns = k_columns(task.metric_set)
    if not columns:
        return []
    anchor = headline_metric(task.metrics, task.metric_set)
    if anchor is None:
        return []
    rows = [f"| `{name}` | " + " | ".join(_k_cells(task, name, k)) + " |" for name, k, _ in columns]
    return [
        "### pass@k",
        "",
        f"Computed on `{anchor}` alone, from the same rollouts and under the same truncation "
        f"policy as the table above. `pass@k` is the unbiased estimate that a draw of k of a "
        f"problem's {task.group_size} rollouts holds a correct one; `pass^k` is the share of "
        "draws in which every one of them is correct, which is what a single sample of this "
        "checkpoint would reproduce. `pass@1` is the avg@k figure above, by construction and "
        "not by coincidence: it is the same estimator at k=1 and the same interval. A k above "
        "the rollouts this taskset drew is absent with its reason rather than reported as a "
        "low number.",
        "",
        "| Estimator | Value | Interval |",
        "|---|---|---|",
        *rows,
        "",
    ]


def _k_cells(task: TaskSummary, name: str, k: int) -> list[str]:
    """One k row's two cells: the number and its interval, or the absence and its reason.

    The reason is `aggregate.k_absence_reason`, which is where the fact lives:
    what a group of that size can answer is a property of the aggregation, and
    the cross-run index states the same absence in the same words by reading the
    same function.
    """
    if name not in task.metrics:
        return ["*unavailable*", k_absence_reason(task, name, k)]
    interval = task.metrics[name]
    return [_percent(interval.value), f"{_percent(interval.low)}–{_percent(interval.high)}"]


def _stop_reason_section(task: TaskSummary) -> list[str]:
    """How the rollouts ended, which the truncation rate alone does not say.

    A completion that stopped on one of the rendering's stop strings and one that
    emitted a learned end-of-sequence token both report `finish_reason: stop`,
    and the difference between them is the difference between a checkpoint halted
    from outside and one that knows how to finish. That is the evidence a
    pretraining checkpoint and a post-trained one are being scored on the same
    instrument without a per-stage switch, so it is reported rather than left in
    the rollout rows.

    A taskset whose every rollout is unclassifiable gets no section. That is the
    pre-F-11 state -- a result file written before the stop reason was recorded,
    or an engine that never sent one -- and a table reading "unknown, 100%" would
    dress the absence of the evidence up as a finding.
    """
    if not task.stop_reasons or set(task.stop_reasons) <= {"unknown"}:
        return []
    total = sum(task.stop_reasons.values())
    rows = [
        f"| {STOP_CLASS_LABELS.get(name, name)} | {count} | {_percent(count / total)} |"
        for name, count in sorted(task.stop_reasons.items(), key=lambda item: -item[1])
    ]
    return [
        "| How the rollout ended | Rollouts | Share |",
        "|---|---|---|",
        *rows,
        "",
    ]


def _comparison_timeout_note(task: TaskSummary) -> list[str]:
    """How often the answer comparator gave up, or nothing when it never did.

    Directly under the table, so it sits against the `truncated` row it is the
    sibling of: both name a way a number above can be too low without the model
    having been wrong.

    Nothing rather than a `0.0%` line, for the reason `_dataset_notes` renders
    nothing: a run with no timeouts has to render exactly the document it
    rendered before this paragraph could exist, or every number already quoted
    from a report like it is being presented differently than when it was
    quoted. It is also the honest default -- silence here has always meant "no
    timeouts", and now it means it because they were counted.
    """
    if not task.comparison_timeout_rate:
        return []
    return [
        f"The answer comparator timed out on **{_rate(task.comparison_timeout_rate)}** of "
        "rollouts. A comparison that times out scores wrong, so the correctness numbers "
        "above are a lower bound, understated by at most that much.",
        "",
    ]


def _format_ok_note(task: TaskSummary, manifest: RunManifest) -> list[str]:
    """What `format_ok` does and does not say once rollouts are being truncated.

    A preflight reported `format_ok` at 100% beside `truncated` at 100%: every
    rollout ran to the generation cap without finishing, and an answer was still
    extractable from all of them, because the checkpoint writes a `\\boxed{}` and
    keeps going. Both numbers are correct -- `format_ok` is `strict_answer is not
    None`, which is a statement about extraction and never about termination --
    and read side by side they invite a conclusion neither supports.

    **The paragraph this replaces went on to say the correctness numbers were
    safe from it "since a truncated rollout scores wrong at every one of them".
    That has not been true since the truncation policy became a choice**, and the
    default went the other way: under `score` a truncated rollout is graded on
    the text it produced, so a column *can* be raised by an answer written into a
    completion that never concluded. The sentence was written when `fail` was the
    only rule, and it survived the change as a reassurance about a gate that is
    no longer on by default. It is stated from the manifest now, so it can only
    say what this run actually did.

    Nothing is printed when the truncation rate is zero, which is the same rule
    `_comparison_timeout_note` follows and for the same reason: a report that
    never truncated has to render exactly the document it rendered before this
    paragraph existed. No threshold beyond zero is applied, because "high" is not
    a number this repository has anywhere else to take from.
    """
    if "format_ok" not in task.metrics or "truncated" not in task.metrics:
        return []
    truncated = task.metrics["truncated"].value
    if not truncated:
        return []
    consequence = (
        "No correctness number above is raised by that: this run scored a truncated rollout "
        "wrong at every one of them."
        if manifest.truncation_policy == "fail"
        else "This run graded a truncated rollout on the text it produced, so the correctness "
        "numbers above do rest in part on completions that never reached an end -- most of "
        "all `permissive`, whose waivers read the answer from anywhere in the text."
    )
    return [
        f"**{_rate(truncated)}** of rollouts ran out of generation budget, and `format_ok` "
        "says only that an answer sat where the reference wanted it, never that the completion "
        "finished: a rollout cut off after writing its answer still counts as formatted. So a "
        f"high `format_ok` beside a truncation rate like this one says the model reached an "
        f"answer and kept going, not that it terminated. {consequence}",
        "",
    ]


def _format_not_evaluable_note(task: TaskSummary) -> list[str]:
    """How many rollouts had no answer region, and what that did to the column.

    Directly beneath the metric table, beside the truncation-rate reading it
    qualifies. This is the share that left the denominator of the metrics the
    taskset declares as needing an answer region -- `format_ok` for the ladder,
    both accuracies for a constraint checker -- because the think block was
    opened, never closed, and ended by the token cap.

    Printed only when non-zero, for the reason `_comparison_timeout_note` is: a
    run in which every completion closed its reasoning renders exactly the
    document it rendered before this paragraph could exist. That is also every
    report written under the E42-era decode, where no completion carried a
    delimiter at all.
    """
    if not task.format_not_evaluable_rate:
        return []
    excluded = ", ".join(f"`{name}`" for name in task.metric_set.requires_answer_region)
    return [
        f"**{_rate(task.format_not_evaluable_rate)}** of rollouts stopped mid-reasoning: the "
        "think block was opened, never closed, and ended by the token cap, so there is no "
        f"answer region for {excluded} to read. Those rollouts are **not evaluable** at "
        f"{excluded} and are left out of that denominator rather than counted as a formatting "
        "failure the model never got the chance to commit. They are counted everywhere else -- "
        "in the truncation rate, in the correctness columns and in the case partition -- "
        "because nothing here drops a rollout.",
        "",
    ]


def _dataset_notes(suite: str, task: str) -> list[str]:
    """This taskset's licence and caveat, as blockquotes, or nothing if it has neither.

    Nothing is deliberately nothing rather than an empty line or an "n/a" row: a
    taskset carrying neither field has to render exactly the document it rendered
    before either field existed, or every `math-standard` number already quoted is
    being presented differently than when it was quoted.

    Blockquotes rather than a table row or a footnote, because the failure mode
    being designed against is a reader who copies the section holding the number
    and leaves the rest. A blockquote is inside that section, survives a paste
    into anything that renders markdown, and still reads as an aside in anything
    that does not.

    The licence gloss is the contract's own rule for a restricted dataset rather
    than an interpretation of the identifier: the field is set only where the
    terms constrain what may be done with the data, and what they constrain is
    always redistribution and publication of items, never the aggregate score.
    Printing the bare identifier would put a string beside the number that a
    reader would still have to go and look up, which is the thing this is for.
    """
    spec = _spec(suite, task)
    if spec is None:
        return []
    notes = []
    if spec.licence:
        notes.append(
            f"> **Licence `{spec.licence}`.** The aggregate scores in this section may be "
            "reported normally. The dataset itself is not redistributed, and its individual "
            "problems, answers and rollouts are not published."
        )
    if spec.caveat:
        notes.append(f"> **Caveat.** {spec.caveat}")
    return notes


def _spec(suite: str, task: str) -> suites.TasksetSpec | None:
    """The specification behind one of a run's tasksets, or None if it is unknown.

    Unknown rather than fatal, because this runs after the rollouts. A summary may
    name a suite or a taskset this checkout no longer defines -- an older result
    file re-rendered, or a taskset moved between suites -- and a report that
    refused to render those numbers would be destroying the record to protect a
    footnote.
    """
    try:
        return suites.taskset(suite, task)
    except ValueError:
        return None


def _comparability_section(manifest: RunManifest) -> list[str]:
    """What must match for these numbers to be quoted beside another run's.

    The relaxation base appears -- as a row and as a sentence -- only for a run
    that recorded one. It is a fact about protocol-scored runs, and a result file
    written before protocols existed has no base to state: printing an empty row
    would describe the run in terms of a distinction it was never subject to,
    while `comparable_with` still refuses to pair it with a run that has one.
    """
    sampling, pins, fingerprint = manifest.sampling, manifest.pins, manifest.fingerprint
    bases = manifest.relaxation_bases
    # Named in the preamble only where the run declared one, for the reason the
    # row below is conditional: a summary that reports no k column is not
    # subject to the distinction, and a sentence gating on it would describe
    # such a run in terms it was never scored under.
    # The parenthetical is the gate's actual shape rather than a hedge: an
    # unrecorded k set is no declaration rather than a different one, because
    # the k columns are additional readings and every shared column means what
    # it always meant. `RunManifest.comparable_with` argues it.
    ks = ", the k set (where both runs declared one)" if manifest.pass_at_k else ""
    rows = [
        # First, because it is the only row here that is a refusal rather than a
        # fact to weigh: two runs on opposite sides of it were scored over
        # different *text*, since the E42-era decode deleted every reasoning
        # delimiter before the grader saw it. Derived from the record rather
        # than declared, so a summary cannot misreport which instrument made it.
        ("instrument era", INSTRUMENT_ERA_LABELS[manifest.instrument_era]),
        # Beside it because the pair is the scoring provenance: which text the
        # graders read, and which graders read it. "unstamped" is a real state
        # and not a gap -- it is every summary written before the stamp existed,
        # and it is why a case letter stored in one is reclassified rather than
        # trusted.
        ("scorer version", manifest.scorer_version or "unstamped (predates the version)"),
        ("format policy", manifest.format_policy),
        ("truncation policy", manifest.truncation_policy),
        ("suite", manifest.suite),
        *([("relaxation base per taskset", bases)] if bases else []),
        # Beside the relaxation base because it is the same kind of fact: which
        # question each taskset's headline was asked. Absent from a run that
        # reported no k column, so a report written before D-06 is unchanged.
        *([("pass@k per taskset", manifest.pass_at_k)] if manifest.pass_at_k else []),
        ("rendering profile", manifest.profile),
        ("template SHA-256", fingerprint.template_sha256),
        # Beside the template digest rather than down among the engine facts,
        # because they are the other half of the same statement: the digest says
        # which bytes were served and these say which alphabet they were served
        # in and which id ended the answer. A reader comparing two numbers has to
        # be able to see, without recomputing a hash, whether they were produced
        # through the same vocabulary -- an E42-era run and a corrected one differ
        # here and nowhere else a reader would look.
        #
        # Conditional for the reason the relaxation base above is: a summary
        # written before these were recorded has nothing to say here, and a row
        # reading "not recorded" would describe such a run in terms of a
        # distinction it was never subject to. The pair moves together, so a
        # reader never sees a terminator without knowing whose vocabulary named
        # it -- and `tokenizer` alone stays absent where the artifact's own was
        # served, which is what its absence has always meant.
        *_vocabulary_rows(fingerprint),
        # The third member of that statement: which alphabet, which terminator,
        # and which decode. A vocabulary that can name an id says nothing about
        # whether the id reached the grader, since the completion path drops
        # whatever the vocabulary marks special -- so a reader establishing that
        # a number was produced with its reasoning delimiters intact needs this
        # row and the tokenizer row together.
        *_detokenisation_rows(sampling),
        ("exemplars SHA-256", manifest.exemplars_sha256 or "n/a (profile carries no exemplars)"),
        ("stop strings", ", ".join(f"`{s!r}`" for s in sampling.stop) or "none"),
        ("temperature / top_p", f"{sampling.temperature} / {sampling.top_p}"),
        *(
            [
                (
                    "top_k / min_p",
                    f"{sampling.top_k if sampling.top_k is not None else 'unset'} / "
                    f"{sampling.min_p if sampling.min_p is not None else 'unset'}",
                ),
                (
                    "presence / repetition penalty",
                    f"{sampling.presence_penalty if sampling.presence_penalty is not None else 'unset'} / "
                    f"{sampling.repetition_penalty if sampling.repetition_penalty is not None else 'unset'}",
                ),
            ]
            if any(
                value is not None
                for value in (
                    sampling.top_k,
                    sampling.min_p,
                    sampling.presence_penalty,
                    sampling.repetition_penalty,
                )
            )
            else []
        ),
        ("max tokens", str(sampling.max_tokens)),
        ("seed", str(sampling.seed) if sampling.seed is not None else "unset"),
        ("`verifiers` pin", pins.verifiers),
        ("`research-environments` pin", pins.research_environments),
        ("`math-verify` pin", pins.math_verify),
        ("dataset revision pin", pins.dataset_revision or "unpinned by the reference taskset"),
        *([("task prompt revision pin", pins.task_prompt_revision)] if pins.task_prompt_revision else []),
        *(
            [("native scorer revision pin", pins.native_scorer_revision)]
            if pins.native_scorer_revision
            else []
        ),
        ("served architecture", fingerprint.architecture),
        ("artifact SHA-256", fingerprint.artifact_sha256),
        ("max_model_len", str(fingerprint.max_model_len)),
        # Conditional because a run served at the artifact's own trained context
        # has no scaling to name, and that is what the row's absence means. The
        # length above already differs between a scaled run and an unscaled one,
        # but it does not say which of the two produced it.
        *([("RoPE scaling", manifest.rope_scaling)] if manifest.rope_scaling else []),
        ("dtype", fingerprint.dtype),
        *(
            [("expected vLLM version", manifest.expected_vllm_version)]
            if "expected_vllm_version" in manifest.model_fields_set
            else []
        ),
        ("vLLM version", fingerprint.vllm_version or "not reported"),
        ("engine environment", fingerprint.engine_venv or "not reported"),
    ]
    preamble = (
        (
            "These numbers may be compared with another run's only when the format policy, the "
            "truncation policy, the "
            f"suite, the relaxation base{ks} and every pin below match. The rendering profile "
            "deliberately may differ: comparing a base rendering against a chat one is what this "
            "repository exists to do. The relaxation base may not, because `lenient` measured "
            "against a published grader and `lenient` measured against prime-rl's verifier are "
            "two quantities under one name. A number quoted without the facts in this table is "
            "diagnostic only."
        )
        if bases
        else (
            "These numbers may be compared with another run's only when the format policy, the "
            "truncation policy, the "
            f"suite{ks} and every pin below match. The rendering profile deliberately may differ: "
            "comparing a base rendering against a chat one is what this repository exists to do. "
            "A number quoted without the facts in this table is diagnostic only."
        )
    )
    return [
        "## Comparability",
        "",
        preamble,
        "",
        "| Fact | Value |",
        "|---|---|",
        *[f"| {name} | {value} |" for name, value in rows],
        "",
        "A change to `verifiers`, `math-verify`, a dataset revision or a rendering template "
        "requires a new run. Earlier results retain their recorded contract. Different "
        "template hashes identify distinct rendering conditions; label that difference when "
        "comparing results and do not pool them as repetitions of one condition.",
    ]


def _vocabulary_rows(fingerprint: Fingerprint) -> list[tuple[str, str]]:
    """Which tokenizer named the ids, and which id ended a completion.

    Nothing at all where the run recorded neither, which is every summary written
    before instrument bug E42 was found. Those runs were served the artifact's own
    tokenizer -- there was no other option -- and printing that back as a row
    would read as a choice somebody made rather than as the default nobody knew
    they were taking.
    """
    if fingerprint.tokenizer is None and not fingerprint.eos_token_ids:
        return []
    return [
        ("tokenizer", fingerprint.tokenizer or "the artifact's own"),
        (
            "end-of-completion ids",
            ", ".join(str(one) for one in fingerprint.eos_token_ids or ()) or "the tokenizer's own",
        ),
    ]


def _detokenisation_rows(sampling: Sampling) -> list[tuple[str, str]]:
    """How the completions this run scored were turned back into text.

    Nothing at all where the run did not record it, for the reason
    `_vocabulary_rows` prints nothing there: such a run took whatever the bound
    engine defaulted to, and a row stating that default would read as a decision
    somebody made rather than one nobody knew they were taking.

    The parenthetical is not decoration. `skip_special_tokens` is the reason a
    completion can arrive at the grader with its `<think>` blocks missing while
    every other field of the record is correct, and a reader who does not already
    know that cannot tell what the flag's value bought.
    """
    if sampling.skip_special_tokens is None:
        return []
    kept = "the reasoning delimiters survive it; the ChatML markers do not"
    dropped = "every special token reaches the grader, ChatML markers included"
    return [
        (
            "completion detokenisation",
            f"`skip_special_tokens={str(sampling.skip_special_tokens).lower()}` "
            f"({kept if sampling.skip_special_tokens else dropped})",
        )
    ]


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _metric_value(task: TaskSummary, name: str, value: float) -> str:
    """Render a scalar in the unit its taskset declared."""
    native = task.metric_set.native(name)
    if native is not None and native.presentation == "points":
        return f"{value:.1f} points"
    return _percent(value)


def _plot_value(task: TaskSummary, name: str, value: float) -> float:
    """Place mixed native units on one chart without changing stored values."""
    native = task.metric_set.native(name)
    if native is None:
        return value
    return (value - native.minimum) / (native.maximum - native.minimum)


def _rate(value: float) -> str:
    """A percentage at the document's usual precision, unless that reads as zero.

    One timeout in MMLU-Pro's 12,032 rollouts is 0.008%, and the one decimal
    every other percentage here uses would print it as `0.0%` inside a sentence
    saying it happened. Widened only in that case, so nothing else in the
    document changes shape.
    """
    percent = _percent(value)
    return percent if percent != _percent(0.0) else f"{100.0 * value:.2g}%"


def _points(value: float) -> str:
    return f"{100.0 * value:.1f} points"


def _confidence(value: float) -> str:
    return f"{100.0 * value:.0f}%"
