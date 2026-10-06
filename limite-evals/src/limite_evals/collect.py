"""An out-root of finished runs, read into one cross-run index.

Aggregating across runs has so far happened beside the runs, with ad-hoc
scripts and hand-made ``-merged`` directories. This module is that operation
brought inside the instrument, under the two rules the rest of the repository
already enforces per run.

**Two numbers share a comparison group only when `comparable_with` says so.**
The index does not decide what is comparable; the schema does, and the index
records the verdict as a group id on every row. An aggregation built off the
index can then group by that id and be mechanically unable to put a `greedy`
column beside an `avg-k` one, or a `score` run beside a `fail` one.
The group permits comparison, not pooling: different rendering conditions
retain their run ids and template hashes on every row.

**A row never carries a bare number.** Every metric travels with the profile
that rendered it and the scoring protocol that produced it -- the protocol's
anchor where one was bound, and the base a relaxation relaxed, in the same
`describe()` string every rendering surface uses. A protocol nothing measured
keeps its row with the reason it is absent, never a zero (D-107).

That rule is why the `pass@k` and `pass^k` columns are rows of the same table
rather than columns beside it. They are one estimator family at one protocol,
and which protocol is not constant: D-06 computes them on the headline alone,
which is `exact` on a taskset with a published grader and `reference` on one
without, so two tasksets in the same run report their k at different anchors.
A k row therefore carries the headline's own `describe()` string, its anchor and
its relaxation base in the columns every other row uses them for -- a reader
filters `metric` and gets the number, the protocol that produced it and the
interval around it from one file. A k the taskset's group size cannot produce is
absent with the reason `aggregate` states for it -- the same sentence the run's
own report prints, read from the module that owns the fact -- never a zero.

The merge exists to retire the hand-made directories. A suite run one taskset
at a time is still one measurement if and only if everything but the taskset
list matches, so `merge_summaries` verifies exactly that and fails loudly on
anything else. The fields a manifest derives *from* its tasksets -- the dataset
pin, the relaxation bases, the k set each taskset reported its headline at, the
largest effective budget, the stop-string union, the rendering digest -- are
recombined the way they were built; every other field must be identical. The merged summary is an index artifact: nothing
is ever written into a run directory, which decision 14 already forbids.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from limite_evals.aggregate import headline_metric, k_absence_reason, k_columns
from limite_evals_core.answer_scoring import validate_answer_scorer_records
from limite_evals_core.schema import Interval, ProtocolRecord, RunManifest, RunSummary, TaskSummary

#: A complete run's summary, then an interrupted run's. `metrics/summary.json`
#: is what marks a run finished (see `report`), so which of the two a directory
#: carries *is* the completeness fact -- no second file is consulted.
SUMMARY_NAMES = ("summary.json", "partial-summary.json")

#: The long-format header: one row per (run, taskset, metric), every row
#: carrying the manifest facts an aggregation must group by. `protocol` is the
#: `describe()` string -- a relaxation is never rendered without its base.
CSV_COLUMNS = (
    "run_id",
    "complete",
    "checkpoint",
    "stage",
    "profile",
    "template_sha256",
    "expected_vllm_version",
    "suite",
    "sampling_policy",
    "truncation_policy",
    "format_policy",
    "pass_at_k",
    "comparison_group",
    "taskset",
    "problems",
    "group_size",
    "metric",
    "protocol",
    "provenance",
    "anchor",
    "relaxes",
    "waivers",
    "value",
    "low",
    "high",
    "unavailable_reason",
)

#: Manifest fields `merge_summaries` requires to be identical, by name. What is
#: absent from this tuple is exactly what a taskset split legitimately varies:
#: `run_id`, `created_at`, and the fields derived from the taskset list
#: (`relaxation_bases`, `pass_at_k`, the dataset and task-prompt pins, the
#: budget, the stop strings, the rendering digest and the exemplar hash), which
#: are recombined instead.
MERGE_IDENTICAL_FIELDS = (
    "checkpoint",
    "stage",
    "profile",
    "suite",
    "format_policy",
    "sampling_policy",
    "truncation_policy",
)


def find_runs(out_root: Path) -> list[Path]:
    """Every directory under `out_root` carrying a summary, however deep.

    Found by the summary file rather than by naming convention, because the
    out-root this exists for accumulated runs under several conventions and a
    scan that trusted names would silently drop the ones it did not expect.
    """
    return sorted(
        {
            path.parent.parent
            for name in SUMMARY_NAMES
            for path in Path(out_root).rglob(f"metrics/{name}")
        }
    )


def load_summary(run_dir: Path) -> tuple[RunSummary, bool]:
    """The run's summary and whether it finished, from the file that says so."""
    for name, complete in zip(SUMMARY_NAMES, (True, False)):
        candidate = run_dir / "metrics" / name
        if candidate.exists():
            return RunSummary.model_validate_json(candidate.read_text()), complete
    raise FileNotFoundError(f"{run_dir} carries no summary to index")


def comparison_groups(manifests: Sequence[RunManifest]) -> list[int]:
    """One group id per manifest; two share one only when `comparable_with` is empty.

    First-fit against each group's first member is enough: every check in
    `comparable_with` is a field equality, so agreeing with the representative
    is agreeing with every member.
    """
    representatives: list[RunManifest] = []
    assigned: list[int] = []
    for manifest in manifests:
        for group, representative in enumerate(representatives):
            if not manifest.comparable_with(representative):
                assigned.append(group)
                break
        else:
            assigned.append(len(representatives))
            representatives.append(manifest)
    return assigned


def metric_entries(task: TaskSummary) -> dict[str, dict]:
    """Every metric the taskset declares, measured or not, with its scoring identity.

    The scoring protocol is never a bare name. An anchored protocol names its
    anchor; a relaxation names the base it relaxed and what it waived, through
    the same `describe()` string every other surface renders; a metric with no
    protocol record -- a ladder rung, or a constraint checker's own metric --
    is its own scoring type. A metric nothing measured keeps its entry with the
    reason and a null value, because absence and zero are different facts.
    """
    entries: dict[str, dict] = {}
    for name in task.metric_set.metrics:
        record = task.protocol(name)
        native = task.metric_set.native(name)
        entries[name] = _entry(
            record,
            unnamed=name,
            provenance=native.provenance if native is not None else task.protocol_source(name),
            interval=task.metrics.get(name),
            absence=(
                record.unavailable_reason
                if record is not None and record.unavailable_reason
                else native.unavailable_reason
                if native is not None and native.unavailable_reason
                else "not measured"
            ),
        )
    return entries


def k_entries(task: TaskSummary) -> dict[str, dict]:
    """Every `pass@k` and `pass^k` column this taskset declares, at the protocol D-06 computed it on.

    Spelled exactly as the summary stores them and the report labels them --
    `pass@32`, `pass^32` -- so the three surfaces name one column one way. The
    `@` and the `^` sort the two families apart and beneath every protocol name,
    and `metric.startswith("pass@")` is the filter, so nothing is gained by a
    tidier spelling that would make a reader translate between the table and the
    run's own report.

    **The row states which protocol produced it**, in the columns every other row
    states it in: the headline's `describe()` string, its anchor, and the base it
    relaxed where it is a relaxation. Prose in one report cannot carry that
    across a matrix, because the headline is per taskset -- `exact` where a
    published grader is bound and `reference` where none is -- so a `pass@32`
    whose protocol went unstated is a number a reader may not put beside the next
    taskset's.

    Nothing at all where the metric set declares no k set, which is every
    constraint-checked taskset and every run written before D-06, and nothing
    where neither anchor produced a number: there is no headline for a k column
    to be computed on, and `metric_entries` has already given that absence its
    reason on the protocol's own row. Both are the conditions `report` renders
    the section under, so the table and the report agree on which rows exist.

    **A k above the rollouts the taskset drew is absent with its reason**
    (D-107), and the reason is `aggregate.k_absence_reason` -- the one the run's
    own report prints -- rather than a second wording of it. Read from the
    aggregation rather than from the report, so indexing an out-root pulls in no
    plotting library and no cache directory to point somewhere writable.
    """
    columns = k_columns(task.metric_set)
    anchor = headline_metric(task.metrics, task.metric_set)
    if not columns or anchor is None:
        return {}
    record = task.protocol(anchor)
    return {
        name: _entry(
            record,
            unnamed=anchor,
            provenance=task.protocol_source(anchor),
            interval=task.metrics.get(name),
            absence=k_absence_reason(task, name, k),
        )
        for name, k, _ in columns
    }


def _entry(
    record: ProtocolRecord | None,
    *,
    unnamed: str,
    provenance: str | None,
    interval: Interval | None,
    absence: str,
) -> dict:
    """One row: the scoring identity, then the number or the reason there is none.

    Shared by both row families so that a k row carries the same scoring
    columns as the protocol rows around it structurally rather than by
    agreement. `unnamed` is what the protocol column reads when the taskset
    recorded no record to describe -- the metric's own name for a rung or a
    constraint checker's metric, and the headline's name for a k column.
    """
    entry = {
        "protocol": record.describe() if record is not None else unnamed,
        "provenance": provenance,
        "anchor": record.anchor if record is not None else None,
        "relaxes": record.relaxes if record is not None else None,
        "waivers": list(record.waivers) if record is not None else [],
        "value": None,
        "low": None,
        "high": None,
        "unavailable_reason": None,
    }
    if interval is None:
        entry["unavailable_reason"] = absence
    else:
        entry.update(value=interval.value, low=interval.low, high=interval.high)
    return entry


def run_entry(run_dir: Path, summary: RunSummary, complete: bool) -> dict:
    """One run as the index carries it: the comparability facts, then the numbers."""
    manifest = summary.manifest
    return {
        "run_id": manifest.run_id,
        "path": str(run_dir),
        "complete": complete,
        "checkpoint": manifest.checkpoint,
        "stage": manifest.stage,
        "profile": manifest.profile,
        "expected_vllm_version": manifest.expected_vllm_version,
        "suite": manifest.suite,
        "format_policy": manifest.format_policy,
        "truncation_policy": manifest.truncation_policy,
        "sampling_policy": manifest.sampling_policy,
        # Which k each taskset reported its headline at. It is a comparability
        # key in its own right (`comparable_with` refuses two runs that declared
        # different sets), so a row that carried the numbers without it would let
        # an aggregation group a `pass@4` column beside a `pass@32` one.
        "pass_at_k": manifest.pass_at_k,
        "relaxation_bases": manifest.relaxation_bases,
        "sampling": {
            "temperature": manifest.sampling.temperature,
            "top_p": manifest.sampling.top_p,
            "top_k": manifest.sampling.top_k,
            "min_p": manifest.sampling.min_p,
            "presence_penalty": manifest.sampling.presence_penalty,
            "repetition_penalty": manifest.sampling.repetition_penalty,
            "max_tokens": manifest.sampling.max_tokens,
            "seed": manifest.sampling.seed,
        },
        "pins": manifest.pins.model_dump(),
        "template_sha256": manifest.fingerprint.template_sha256,
        "artifact_sha256": manifest.fingerprint.artifact_sha256,
        "tasksets": [
            {
                "taskset": task.task,
                "problems": task.problems,
                "group_size": task.group_size,
                "metric_set": task.metric_set.name,
                # The declared metrics first, then the k columns computed on
                # whichever of them was the headline: one map, so every metric
                # this taskset reported reaches the long table through one loop
                # and is filtered out of it the same way.
                "metrics": metric_entries(task) | k_entries(task),
            }
            for task in summary.tasks
        ],
    }


def merge_summaries(summaries: Sequence[RunSummary]) -> RunSummary:
    """Runs split by taskset, recombined into the one summary they measure.

    Refuses -- with every reason, not the first -- inputs whose manifests
    differ on anything except the taskset list. The per-taskset fields are
    recombined the way the CLI builds them: the dataset, task-prompt, and
    native-scorer pins and the relaxation bases as sorted unions that must not conflict per taskset,
    the budget as the largest effective one, the stop strings as a
    first-appearance union.
    The rendering digest cannot be rebuilt from summaries -- the per-taskset
    rendering hashes are not in them -- so it is the digest of each taskset's
    source digest, which still moves whenever any source's rendering does.
    """
    if len(summaries) < 2:
        raise ValueError("a merge needs at least two runs")
    for summary in summaries:
        for task in summary.tasks:
            validate_answer_scorer_records(
                summary.manifest.pins, task.task, {record.protocol: record for record in task.protocols}
            )
    first = summaries[0].manifest
    reasons: list[str] = []
    for other in (summary.manifest for summary in summaries[1:]):
        for field in MERGE_IDENTICAL_FIELDS:
            mine, theirs = getattr(first, field), getattr(other, field)
            if mine != theirs:
                reasons.append(f"{other.run_id}: different {field}: {mine!r} vs {theirs!r}")
        for field in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "seed",
        ):
            mine, theirs = getattr(first.sampling, field), getattr(other.sampling, field)
            if mine != theirs:
                reasons.append(f"{other.run_id}: different sampling {field}: {mine} vs {theirs}")
        for field in ("verifiers", "research_environments", "math_verify"):
            mine, theirs = getattr(first.pins, field), getattr(other.pins, field)
            if mine != theirs:
                reasons.append(f"{other.run_id}: different {field} pin: {mine} vs {theirs}")
        for field, (mine, theirs) in other.fingerprint.mismatches(first.fingerprint).items():
            if field != "template_sha256":
                reasons.append(f"{other.run_id}: different served {field}: {mine} vs {theirs}")
        if first.exemplars_sha256 != other.exemplars_sha256:
            # Per-taskset-derived, but not recombinable: the hash covers file
            # bytes the summaries do not carry. No run merged so far uses
            # exemplars, so this refuses rather than guesses.
            reasons.append(f"{other.run_id}: different exemplars hash; merging those is not supported")
    seen: dict[str, str] = {}
    for summary in summaries:
        for task in summary.tasks:
            if task.task in seen:
                reasons.append(
                    f"taskset {task.task} appears in both {seen[task.task]} and "
                    f"{summary.manifest.run_id}"
                )
            seen[task.task] = summary.manifest.run_id
    manifests = [summary.manifest for summary in summaries]
    _merge_per_taskset([m.pins.dataset_revision for m in manifests], "dataset_revision", reasons)
    _merge_per_taskset(
        [m.pins.task_prompt_revision for m in manifests], "task_prompt_revision", reasons
    )
    _merge_per_taskset(
        [m.pins.native_scorer_revision for m in manifests],
        "native_scorer_revision",
        reasons,
    )
    _merge_per_taskset(
        [m.pins.answer_scorer_revision for m in manifests], "answer_scorer_revision", reasons
    )
    _merge_per_taskset([m.relaxation_bases for m in manifests], "relaxation_bases", reasons)
    _merge_per_taskset([m.pass_at_k for m in manifests], "pass_at_k", reasons)
    if reasons:
        raise ValueError("these runs are not one measurement:\n  " + "\n  ".join(reasons))

    stop: dict[str, None] = {}
    for manifest in manifests:
        stop.update(dict.fromkeys(manifest.sampling.stop))
    template_payload = {
        task.task: summary.manifest.fingerprint.template_sha256
        for summary in summaries
        for task in summary.tasks
    }
    merged = first.model_copy(
        update={
            "run_id": "+".join(manifest.run_id for manifest in manifests),
            "created_at": max(manifest.created_at for manifest in manifests),
            "relaxation_bases": _merge_per_taskset(
                [m.relaxation_bases for m in manifests], "relaxation_bases", []
            ),
            # Recombined for the reason `relaxation_bases` is, and it is the same
            # shape: `taskset=fact` pairs joined by commas, the k of one taskset
            # joined with `+` inside its own value. Carrying the first fragment's
            # string instead would state one taskset's k set over a merge that
            # measured several, and `comparable_with` gates on the field.
            "pass_at_k": _merge_per_taskset([m.pass_at_k for m in manifests], "pass_at_k", []),
            "sampling": first.sampling.model_copy(
                update={
                    "max_tokens": max(m.sampling.max_tokens for m in manifests),
                    "stop": list(stop),
                }
            ),
            "pins": first.pins.model_copy(
                update={
                    "dataset_revision": _merge_per_taskset(
                        [m.pins.dataset_revision for m in manifests], "dataset_revision", []
                    ),
                    "task_prompt_revision": _merge_per_taskset(
                        [m.pins.task_prompt_revision for m in manifests],
                        "task_prompt_revision",
                        [],
                    ),
                    "answer_scorer_revision": _merge_per_taskset(
                        [m.pins.answer_scorer_revision for m in manifests], "answer_scorer_revision", []
                    ),
                    "native_scorer_revision": _merge_per_taskset(
                        [m.pins.native_scorer_revision for m in manifests],
                        "native_scorer_revision",
                        [],
                    ),
                }
            ),
            "fingerprint": first.fingerprint.model_copy(
                update={
                    "template_sha256": hashlib.sha256(
                        json.dumps(template_payload, sort_keys=True).encode()
                    ).hexdigest()
                }
            ),
        }
    )
    return RunSummary(
        manifest=merged, tasks=[task for summary in summaries for task in summary.tasks]
    )


def _merge_per_taskset(values: Sequence[str], field: str, reasons: list[str]) -> str:
    """`taskset=fact` strings unioned, refusing a taskset two runs disagree on.

    The CLI writes these fields sorted by taskset name and joined with commas
    (`_dataset_revisions`, `_relaxation_bases`), so the union is written back
    the same way and string equality keeps meaning what `comparable_with`
    needs it to mean.
    """
    merged: dict[str, str] = {}
    for value in values:
        for pair in filter(None, value.split(",")):
            name, _, fact = pair.partition("=")
            if name in merged and merged[name] != fact:
                reasons.append(f"{field} disagrees at {name}: {merged[name]!r} vs {fact!r}")
            merged[name] = fact
    return ",".join(f"{name}={fact}" for name, fact in sorted(merged.items()))


def collect(out_root: Path, *, merges: Sequence[Sequence[str]] = ()) -> dict:
    """The whole index: every run under `out_root`, plus any requested merges."""
    out_root = Path(out_root)
    loaded = [(run_dir, *load_summary(run_dir)) for run_dir in find_runs(out_root)]
    entries = [run_entry(run_dir, summary, complete) for run_dir, summary, complete in loaded]
    summaries = {entry["run_id"]: summary for entry, (_, summary, _) in zip(entries, loaded)}
    manifests = [summary.manifest for _, summary, _ in loaded]

    for group in merges:
        missing = [run_id for run_id in group if run_id not in summaries]
        if missing:
            raise ValueError(f"nothing under {out_root} carries run id {', '.join(missing)}")
        merged = merge_summaries([summaries[run_id] for run_id in group])
        entry = run_entry(out_root, merged, True)
        entry["path"] = None
        entry["merged_from"] = list(group)
        entries.append(entry)
        manifests.append(merged.manifest)

    for entry, group in zip(entries, comparison_groups(manifests)):
        entry["comparison_group"] = group
    return {"out_root": str(out_root), "runs": entries}


def csv_rows(index: dict) -> list[dict]:
    """The index in long format: one row per (run, taskset, metric)."""
    rows = []
    for run in index["runs"]:
        for taskset in run["tasksets"]:
            for metric, entry in taskset["metrics"].items():
                rows.append(
                    {
                        **{key: run[key] for key in CSV_COLUMNS if key in run},
                        "taskset": taskset["taskset"],
                        "problems": taskset["problems"],
                        "group_size": taskset["group_size"],
                        "metric": metric,
                        **{
                            key: entry[key]
                            for key in (
                                "protocol",
                                "provenance",
                                "anchor",
                                "relaxes",
                                "value",
                                "low",
                                "high",
                            )
                        },
                        "waivers": ";".join(entry["waivers"]),
                        "unavailable_reason": entry["unavailable_reason"],
                    }
                )
    return rows


def write_index(index: dict, out_dir: Path) -> tuple[Path, Path]:
    """`index.json` and `index.csv` under `out_dir`, which is never a run directory."""
    out_dir = Path(out_dir)
    json_path, csv_path = out_dir / "index.json", out_dir / "index.csv"
    json_path.write_text(json.dumps(index, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows(index))
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        index = collect(args.out_root, merges=[m.split(",") for m in args.merge])
    except (ValueError, FileNotFoundError) as failure:
        print(f"error: {failure}")
        return 1
    json_path, csv_path = write_index(index, args.out or args.out_root)
    groups = len({run["comparison_group"] for run in index["runs"]})
    print(f"{len(index['runs'])} runs in {groups} comparison groups -> {json_path}, {csv_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limite-collect",
        description="index every run under an out-root into one cross-run table",
    )
    parser.add_argument("--out-root", required=True, type=Path, help="the directory runs were written under")
    parser.add_argument(
        "--merge",
        action="append",
        default=[],
        metavar="RUN_ID,RUN_ID[,...]",
        help="comma-separated run ids split by taskset, indexed as one merged measurement; "
        "repeatable. Refused unless their manifests match on everything but the taskset list",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="where index.json and index.csv land; default is the out-root itself",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
