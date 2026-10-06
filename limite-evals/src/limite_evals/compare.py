"""Paired comparison between two runs, from the rollouts they stored.

Ported from the scripts that lived unversioned beside the runs on the cluster
(`paired_delta.py` and `rescore/table.py`), minus their experiment-specific
constants: a problem-index split is an argument here, never a hard-coded list.

The statistic is the paired hierarchical bootstrap. Problems are resampled with
replacement and then that problem's rollouts within each side, exactly as
`limite_evals_core.stats.bootstrap_interval` does for a single run -- but the
problem is drawn *once* and read on both sides, so the problem-level variance
that dominates a per-side interval cancels in the delta instead of being
counted twice. The per-side intervals are still computed the single-run way,
which is how the pairing is checked against the summaries the runs already
carry before any new number is read off it.

What this module does not do is decide comparability. It reads whatever two
run directories it is given; whether their numbers may sit in one table is
`RunManifest.comparable_with`'s verdict, carried per run by `collect`'s index.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from limite_evals_core.schema import PROTOCOL_METRICS

RESAMPLES = 2000
CONFIDENCE = 0.95
COLUMNS = PROTOCOL_METRICS.metrics


def load(run_dir: Path) -> dict:
    """`{"tasks": {task: {problem: {column: [float, ...]}}}, "rows": [...]}`.

    Correctness columns are read from `metrics`, where a protocol that bound no
    artefact is absent rather than false -- a problem with no verdict at a
    column contributes nothing to it, which is D-107 applied to a delta. A
    rollout flagged `format_not_evaluable` is left out of the `format_ok`
    denominator: a completion whose think block ran into the token cap never
    made a formatting decision to be scored on.
    """
    rows = [json.loads(line) for line in (Path(run_dir) / "metrics" / "samples.jsonl").open()]
    tasks: dict = {}
    source_group_tasks: set[str] = set()
    for row in rows:
        per_problem = tasks.setdefault(row["task"], {}).setdefault(row["problem_index"], {})
        if row.get("source_group_id") is not None:
            source_group_tasks.add(row["task"])
        metric_columns = set(row.get("metrics", {}))
        columns = metric_columns | {"truncated"}
        if metric_columns & (set(COLUMNS) - {"format_ok", "truncated"}):
            columns.add("format_ok")
        for column in columns:
            if column == "format_ok":
                if row.get("format_not_evaluable"):
                    continue
                value = float(row[column])
            elif column == "truncated":
                value = float(row[column])
            else:
                metric = row.get("metrics", {}).get(column)
                if metric is None:
                    continue
                value = float(metric)
            per_problem.setdefault(column, []).append(value)
    return {"tasks": tasks, "rows": rows, "source_group_tasks": source_group_tasks}


def mean_of_groups(groups: list[list[float]]) -> float:
    scored = [group for group in groups if group]
    if not scored:
        return 0.0
    return sum(sum(group) / len(group) for group in scored) / len(scored)


def interval(groups: list[list[float]], *, seed: int = 0) -> dict:
    """Single-side interval, identical in construction to the repository's own."""
    point = mean_of_groups(groups)
    scored = [list(group) for group in groups if group]
    if len(scored) < 2:
        return {"value": point, "low": point, "high": point}
    rng = random.Random(seed)
    n = len(scored)
    draws = []
    for _ in range(RESAMPLES):
        total = 0.0
        for _ in range(n):
            group = scored[rng.randrange(n)]
            k = len(group)
            total += sum(group[rng.randrange(k)] for _ in range(k)) / k
        draws.append(total / n)
    draws.sort()
    tail = (1.0 - CONFIDENCE) / 2.0
    return {
        "value": point,
        "low": draws[min(int(tail * RESAMPLES), RESAMPLES - 1)],
        "high": draws[min(int((1.0 - tail) * RESAMPLES), RESAMPLES - 1)],
    }


def paired_delta(groups_a: list[list[float]], groups_b: list[list[float]], *, seed: int = 0) -> dict:
    """`a - b`, drawing each problem once and reading both sides at it.

    `significant` means the interval excludes zero, at this module's fixed
    confidence -- the same 95% every interval in the repository quotes.
    """
    pairs = [(list(a), list(b)) for a, b in zip(groups_a, groups_b) if a and b]
    point = mean_of_groups([a for a, _ in pairs]) - mean_of_groups([b for _, b in pairs])
    if len(pairs) < 2:
        return {"value": point, "low": point, "high": point, "significant": False}
    rng = random.Random(seed)
    n = len(pairs)
    draws = []
    for _ in range(RESAMPLES):
        total_a = 0.0
        total_b = 0.0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            ka, kb = len(a), len(b)
            total_a += sum(a[rng.randrange(ka)] for _ in range(ka)) / ka
            total_b += sum(b[rng.randrange(kb)] for _ in range(kb)) / kb
        draws.append((total_a - total_b) / n)
    draws.sort()
    tail = (1.0 - CONFIDENCE) / 2.0
    low = draws[min(int(tail * RESAMPLES), RESAMPLES - 1)]
    high = draws[min(int((1.0 - tail) * RESAMPLES), RESAMPLES - 1)]
    return {"value": point, "low": low, "high": high, "significant": (low > 0.0) or (high < 0.0)}


def compare(a: dict, b: dict, task: str, column: str, indices=None) -> dict:
    """Both sides and their paired delta at one column, over the shared problems."""
    if task in a.get("source_group_tasks", set()) or task in b.get("source_group_tasks", set()):
        raise ValueError(
            f"{task}: row-level comparison of a source-group metric is unsupported; "
            "compare its aggregated summaries so fixed variants are not redrawn"
        )
    a_task, b_task = a["tasks"].get(task, {}), b["tasks"].get(task, {})
    keys = sorted(set(a_task) & set(b_task))
    if indices is not None:
        keys = [key for key in keys if key in set(indices)]
    groups_a = [a_task[key].get(column, []) for key in keys]
    groups_b = [b_task[key].get(column, []) for key in keys]
    return {
        "problems": len(keys),
        "a": interval(groups_a),
        "b": interval(groups_b),
        "delta": paired_delta(groups_a, groups_b),
    }


def efficiency(run: dict, task: str) -> dict:
    """Tokens spent per `exact`-correct rollout, and the rates that qualify it."""
    rows = [row for row in run["rows"] if row["task"] == task]
    if not rows:
        return {}
    correct = sum(bool(row.get("metrics", {}).get("exact")) for row in rows)
    tokens = sum(row["completion_tokens"] for row in rows)
    return {
        "rollouts": len(rows),
        "total_completion_tokens": tokens,
        "mean_completion_tokens": tokens / len(rows),
        "exact_correct": correct,
        "tokens_per_correct": (tokens / correct) if correct else None,
        "truncated_rate": sum(bool(row["truncated"]) for row in rows) / len(rows),
    }


def termination(run: dict, task: str | None = None) -> dict:
    """How the completions ended, plus the think-block discipline where one exists."""
    rows = [row for row in run["rows"] if task is None or row["task"] == task]
    if not rows:
        return {}
    return {
        "rollouts": len(rows),
        "opens_think": sum("<think>" in row["completion"] for row in rows) / len(rows),
        "closes_think": sum("</think>" in row["completion"] for row in rows) / len(rows),
        "finish_reason": dict(Counter(row["finish_reason"] for row in rows)),
        "stop_reason": dict(Counter(str(row["stop_reason"]) for row in rows)),
        "truncated": sum(bool(row["truncated"]) for row in rows) / len(rows),
        "mean_completion_tokens": sum(row["completion_tokens"] for row in rows) / len(rows),
    }


def compare_runs(
    a_dir: Path,
    b_dir: Path,
    *,
    tasks: list[str] | None = None,
    columns: tuple[str, ...] | None = None,
    splits: dict[str, dict[str, list[int]]] | None = None,
) -> dict:
    """The whole comparison: every supported shared taskset, `a` minus `b`.

    `splits` restricts a taskset to named problem-index subsets --
    `{taskset: {split_name: [indices]}}` -- which is how a contamination or
    difficulty split is read without this module carrying anyone's index list.
    Source-group tasksets are listed under `unavailable_tasks` because comparing
    their row-level samples would redraw fixed variants.
    """
    a, b = load(a_dir), load(b_dir)
    shared = [task for task in a["tasks"] if task in b["tasks"]]
    if tasks is not None:
        shared = [task for task in tasks if task in shared]
    source_group_tasks = a["source_group_tasks"] | b["source_group_tasks"]
    unavailable_tasks = {
        task: (
            "row-level comparison of a source-group metric is unsupported; "
            "compare its aggregated summaries so fixed variants are not redrawn"
        )
        for task in shared
        if task in source_group_tasks
    }
    shared = [task for task in shared if task not in source_group_tasks]
    out: dict = {
        "a_run": Path(a_dir).name,
        "b_run": Path(b_dir).name,
        "resamples": RESAMPLES,
        "confidence": CONFIDENCE,
        "tasks": {},
        "unavailable_tasks": unavailable_tasks,
        "splits": {},
        "termination": {"a": termination(a), "b": termination(b)},
        "efficiency": {},
    }
    task_columns: dict[str, tuple[str, ...]] = {}
    for task in shared:
        observed = {
            column for run in (a, b) for problem in run["tasks"].get(task, {}).values() for column in problem
        }
        task_columns[task] = columns or (
            *(column for column in COLUMNS if column in observed),
            *sorted(observed - set(COLUMNS)),
        )
        out["tasks"][task] = {column: compare(a, b, task, column) for column in task_columns[task]}
        out["efficiency"][task] = {"a": efficiency(a, task), "b": efficiency(b, task)}
    for task, named in (splits or {}).items():
        if task not in shared:
            continue
        out["splits"][task] = {
            name: {column: compare(a, b, task, column, indices) for column in task_columns[task]}
            for name, indices in named.items()
        }
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    splits = json.loads(Path(args.splits).read_text()) if args.splits else None
    result = compare_runs(
        args.a_dir,
        args.b_dir,
        tasks=args.tasksets.split(",") if args.tasksets else None,
        splits=splits,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {args.out}")
    for task, reason in result["unavailable_tasks"].items():
        print(f"  {task:16s} unavailable: {reason}")
    for task, columns in result["tasks"].items():
        for column in ("exact", "reference"):
            body = columns.get(column)
            if body is None or body["delta"]["value"] == body["delta"]["low"] == body["delta"]["high"] == 0.0:
                continue
            delta = body["delta"]
            print(
                f"  {task:16s} {column:9s} delta {100 * delta['value']:+.1f} "
                f"({100 * delta['low']:+.1f} to {100 * delta['high']:+.1f}) "
                f"{'EXCLUDES ZERO' if delta['significant'] else 'includes zero'}"
            )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limite-compare",
        description="paired hierarchical bootstrap between two runs' stored rollouts",
    )
    parser.add_argument("a_dir", type=Path, help="a run directory carrying metrics/samples.jsonl")
    parser.add_argument("b_dir", type=Path, help="the run it is compared against (delta is a - b)")
    parser.add_argument("--tasksets", help="comma-separated subset; default is every shared taskset")
    parser.add_argument(
        "--splits",
        type=Path,
        help="a JSON file `{taskset: {split_name: [problem indices]}}` to compare subsets too",
    )
    parser.add_argument("--out", type=Path, help="write the full comparison as JSON here")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
