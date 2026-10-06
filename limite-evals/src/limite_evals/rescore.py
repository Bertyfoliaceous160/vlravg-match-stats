"""Score a finished run's rollouts again, without re-serving the checkpoint.

A completion is expensive and a verdict is cheap, and the two do not have the
same lifetime. `metrics/samples.jsonl` keeps every rollout's completion beside
its gold, so when a grader binding changes -- a published grader that could not
be bound becomes bindable, or a binding is corrected -- the run's numbers can be
recovered from what it already generated instead of from the GPU again.

Three things here are load-bearing rather than incidental.

**The run directory is never written to.** `report.reserve` claims a directory
once and refuses one that already carries a summary, and a rescore is not the
run: it scores the same completions under a different version of this code. It
writes to its own directory, and the original stays the record of what was
reported.

**`relaxation_bases` is re-derived, never carried over.** It is the one manifest
field a rescore can invalidate on its own. `lenient` relaxes `exact` where a
published grader is bound and `reference` where none is, so a binding that
appears between the run and the rescore silently changes what two columns
measure. Copying the recorded string would state the old base beside numbers
computed against the new one, and `comparable_with` would then read equal on two
runs that are not.

**The run id carries a suffix.** The rest of the manifest is true of these
samples -- the checkpoint, the rendering, the sampling policy and the dataset
pins all described how the completions were produced, and re-scoring changes
none of them -- so a rescored summary would otherwise compare equal to the run
it came from while reporting different numbers.

What it cannot do is recover anything the completions do not contain. A
rendering change, a stop-string change or a sampling change is a new run; this
only re-runs the scoring path.

The truncation policy is the one scoring-path decision it does *not* take from
this checkout: it defaults to the policy the run recorded, because restating an
old run's numbers under a rule it was never computed with is the thing a rescore
would otherwise do silently to every run in an output directory.

**An E42-era run is refused outright**, and that refusal is the reason this file
is worth reading twice. A rescore's premise is that the completion is the durable
thing and the verdict is cheap; under the E42-era instrument the completion is
*not* the model's output. The tokenizer deleted every `<think>` and `</think>`
before the text was ever written down, and no token ids were kept, so there is
nothing on disk to recover them from. Rescoring such a file under think-aware
rules would produce a confident number about a checkpoint that appears never to
have opened a think block -- the tokenizer's behaviour, reported under the
model's name. Regeneration is the only honest path, and the run directory is left
untouched: it is the record of what was reported at the time.

Its converse is the property that makes this file worth having. A corrected-era
`samples.jsonl` carries the delimiters intact, so every scoring change in this
checkout -- the think-aware hashed lens, the not-evaluable format rule, the
answer-region constraint checks -- reaches an existing baseline through this
path, with no GPU time at all.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from limite_evals import (
    aggregate,
    report,
    suites,
    tasksets,
    taxonomy,
)
from limite_evals.runner import _record
from limite_evals_core.answer_scoring import answer_scorer_revisions
from limite_evals_core.ladder import format_not_evaluable
from limite_evals_core.ladder import score as score_ladder
from limite_evals_core.protocols import score as score_protocols
from limite_evals_core.schema import (
    SCORER_VERSION,
    RunManifest,
    RunSummary,
    SampleResult,
    TruncationPolicy,
)

#: Appended to the run id so a rescored manifest cannot compare equal to the run
#: whose completions it scored. Every other field of that manifest is still true.
RESCORE_SUFFIX = "-rescore"

DEFAULT_WORKERS = 16


def summary_path(run_dir: Path) -> Path:
    """The run's own summary, whether it finished or was interrupted.

    A partial run is worth rescoring on exactly the same terms as a complete one:
    its manifest describes the tasksets that did run, which is what its samples
    contain.
    """
    for name in ("summary.json", "partial-summary.json"):
        candidate = run_dir / "metrics" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{run_dir} carries no summary to read a manifest from")


def read_manifest(run_dir: Path) -> RunManifest:
    """The run's own manifest, refused where its rollouts may not be rescored.

    The quarantine is checked here rather than after the fan-out, because the
    refusal follows from the manifest alone and belongs on the near side of a
    sixteen-process pool -- the same rule the CLI already follows for every
    refusal that needs no GPU.
    """
    manifest = RunSummary.model_validate_json(summary_path(run_dir).read_text()).manifest
    aggregate.assert_scoreable(manifest)
    return manifest


def read_samples(run_dir: Path) -> Iterator[str]:
    """The rollout rows, unparsed. Parsing happens in the worker that scores them."""
    with (run_dir / "metrics" / "samples.jsonl").open() as handle:
        yield from handle


def rescore_sample(spec: suites.TasksetSpec, sample: SampleResult) -> SampleResult:
    """One rollout scored again from its completion and gold alone.

    Everything describing how the completion was produced is copied rather than
    recomputed. A rescore cannot observe why a completion ended or how long it
    was; those are facts about the run, and `truncated` in particular has to
    survive intact, because it is what decision 10 reads at the aggregation
    boundary to fail a truncated completion at every correctness metric.
    """
    produced = sample.model_dump(
        include={
            "task",
            "problem_index",
            "rollout_index",
            "source_group_id",
            "source_variant_id",
            "gold",
            "completion",
            "finish_reason",
            "stop_reason",
            "template_sha256",
            "truncated",
            "completion_tokens",
        }
    ) | {
        # Re-derived rather than copied, on the same terms as the verdicts: it
        # is read from the stored completion and the run's own `truncated`, and
        # it is this checkout's rule that decides what those two mean.
        "format_not_evaluable": format_not_evaluable(
            sample.completion, truncated=sample.truncated
        ),
        "scorer_version": SCORER_VERSION,
    }
    answer = sample.gold
    scored = score_protocols(sample.completion, answer, tasksets.taskset_protocols(spec))
    ladder = score_ladder(
        sample.completion,
        answer,
        answer_format=spec.answer_format,
        truncated=sample.truncated,
    )
    return SampleResult(
        **produced,
        protocols={name: _record(result) for name, result in scored.items()},
        metrics={
            name: float(result.correct)
            for name, result in scored.items()
            if result.correct is not None
        },
        strict=ladder.strict,
        lenient=ladder.lenient,
        permissive=ladder.permissive,
        format_ok=ladder.format_ok,
        comparison_timeout=ladder.comparison_timeout,
        strict_answer=ladder.strict_answer,
        lenient_answer=ladder.lenient_answer,
        # Re-derived rather than copied, for the reason the verdicts are: the
        # taxonomy is this checkout's, and a rescore exists to restate a run's
        # rollouts under it. `truncated` is the run's own record and is copied,
        # so the `C` against `E` split stays a fact about how the rollout ended.
        failure_case=taxonomy.classify(
            sample.completion, answer, truncated=sample.truncated
        ),
    )


#: Resolved once per worker process rather than per row: binding a taskset's
#: `exact` re-reads, re-hashes and executes the vendored grader, and a rescore
#: hands one worker thousands of rows of the same taskset.
_SPECS: dict[str, suites.TasksetSpec] = {}


def _load_specs(suite: str) -> None:
    global _SPECS
    _SPECS = {spec.name: spec for spec in suites.resolve(suite)}


def _rescore_row(row: str) -> str:
    """One row in, one row out. JSON across the process boundary because a
    `SampleResult` carries a completion that is cheaper to move as bytes."""
    sample = SampleResult.model_validate_json(row)
    return rescore_sample(_SPECS[sample.task], sample).model_dump_json()


def rescore_run(
    run_dir: Path,
    out_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    truncation_policy: TruncationPolicy | None = None,
) -> RunSummary:
    """Every rollout in `run_dir` scored again, summarised and written to `out_dir`.

    Scoring is CPU-bound and independent per rollout -- a published grader
    executes sympy over one completion at a time -- so it fans out across
    processes. The order rows come back in does not matter: aggregation groups by
    taskset and problem and sorts within a group.

    `out_dir` must not exist: rescoring never overwrites a source or prior result.

    `truncation_policy` defaults to **the run's own**, not to the CLI's default,
    and the asymmetry is the point. A rescore exists to move the graders and hold
    everything else still; the truncation policy is not a grader, and inheriting
    the instrument's current default would silently restate an old run's numbers
    under a rule it was never computed with. Passing one explicitly is how you ask
    for that, and it is then recorded in the rescored manifest.
    """
    if out_dir.exists():
        raise FileExistsError(f"rescore directory already exists: {out_dir}")
    manifest = read_manifest(run_dir)
    _load_specs(manifest.suite)

    with ProcessPoolExecutor(
        workers, initializer=_load_specs, initargs=(manifest.suite,)
    ) as pool:
        samples = [
            SampleResult.model_validate_json(row)
            for row in pool.map(_rescore_row, read_samples(run_dir), chunksize=8)
        ]

    # Suite order restricted to what these samples contain, which for a partial
    # run is fewer tasksets than the suite names.
    scored = tuple(_SPECS[name] for name in dict.fromkeys(sample.task for sample in samples))
    summary = aggregate.summarize(
        manifest.model_copy(
            update={
                "run_id": manifest.run_id + RESCORE_SUFFIX,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "relaxation_bases": _relaxation_bases(scored),
                "truncation_policy": truncation_policy or manifest.truncation_policy,
                "pins": manifest.pins.model_copy(
                    update={
                        "native_scorer_revision": "",
                        "answer_scorer_revision": answer_scorer_revisions(spec.name for spec in scored),
                    }
                ),
            }
        ),
        samples,
        metric_sets=_metric_sets(scored),
        answer_formats={spec.name: spec.answer_format for spec in scored},
    )

    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics" / "summary.json").write_text(summary.model_dump_json(indent=2))
    (out_dir / "reports" / "summary.md").write_text(report.render_markdown(summary))
    report.write_samples(out_dir / "metrics" / "samples.jsonl", samples)
    report.write_ladder_plot(out_dir / "plots" / "ladder.png", summary)
    return summary


def _metric_sets(specs: tuple[suites.TasksetSpec, ...]):
    """The CLI's own mapping, imported at call time.

    `limite_evals.cli` builds an argument parser and reaches the engine at import;
    a rescore needs neither, and importing it lazily keeps this module usable
    where the serving stack is not installed. What a taskset reports is one
    decision and it stays in one place.
    """
    from limite_evals.cli import _metric_sets as build

    return build(specs)


def _relaxation_bases(specs: tuple[suites.TasksetSpec, ...]) -> str:
    from limite_evals.cli import _relaxation_bases as build

    return build(specs)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out or (args.run_dir.parent / (args.run_dir.name + RESCORE_SUFFIX)))
    try:
        summary = rescore_run(
            args.run_dir,
            out_dir,
            workers=args.workers,
            truncation_policy=args.truncation_policy,
        )
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    print(f"{summary.manifest.run_id}: {len(summary.tasks)} tasksets -> {out_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="score a finished run's stored rollouts again, under this checkout's graders"
    )
    parser.add_argument("run_dir", type=Path, help="a run directory carrying metrics/samples.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        help="where to write the rescored run; default is the run directory's name "
        f"with `{RESCORE_SUFFIX}` appended, beside it. Never the run directory itself",
    )
    parser.add_argument(
        "--truncation-policy",
        choices=("score", "fail"),
        help="what a completion that ran out of generation budget scores. Defaults to "
        "whatever the run being rescored recorded, so a rescore moves the graders and "
        "nothing else; pass one to compute the other quantity instead",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser


if __name__ == "__main__":
    main()
