"""The cross-run index: protocol identity on every row, grouping, and the merge."""

from __future__ import annotations

import csv
import json
import subprocess
import sys

import pytest

from limite_evals.collect import (
    CSV_COLUMNS,
    collect,
    comparison_groups,
    merge_summaries,
    write_index,
)
from limite_evals_core.schema import (
    LADDER,
    PROTOCOL_METRICS,
    Fingerprint,
    Interval,
    MetricSet,
    NativeMetric,
    Pins,
    ProtocolRecord,
    RunManifest,
    RunSummary,
    Sampling,
    TaskSummary,
)


def test_native_metric_provenance_range_and_absence_survive_collection(tmp_path) -> None:
    metric_set = MetricSet(
        name="physics",
        metrics=("exact", "eed"),
        native_metrics=(
            NativeMetric(
                name="exact",
                minimum=0.0,
                maximum=1.0,
                presentation="percent",
                provenance="official accuracy",
                unavailable_reason="official accuracy is not published",
            ),
            NativeMetric(
                name="eed",
                minimum=0.0,
                maximum=100.0,
                presentation="points",
                provenance="official EED implementation",
            ),
        ),
    )
    task = TaskSummary(
        task="phybench",
        problems=100,
        group_size=1,
        metric_set=metric_set,
        eed=Interval(value=50.0, low=0.0, high=100.0),
    )
    write_run_dir(
        tmp_path,
        RunSummary(
            manifest=manifest(suite="physics-extended-v1"),
            tasks=[task],
        ),
    )

    entries = collect(tmp_path)["runs"][0]["tasksets"][0]["metrics"]
    assert entries["eed"]["value"] == 50.0
    assert entries["eed"]["provenance"] == "official EED implementation"
    assert entries["exact"]["value"] is None
    assert entries["exact"]["unavailable_reason"] == "official accuracy is not published"

FINGERPRINT = Fingerprint(
    architecture="LimiteForCausalLM",
    artifact_sha256="a" * 64,
    template_sha256="b" * 64,
    max_model_len=8192,
    dtype="bfloat16",
)


def manifest(**overrides) -> RunManifest:
    fields = dict(
        run_id="r001",
        created_at="2026-08-13T00:00:00Z",
        checkpoint="/work/ckpt",
        stage="sft",
        profile="chat",
        suite="math-extended",
        truncation_policy="score",
        relaxation_bases="math500=exact",
        sampling=Sampling(temperature=0.6, top_p=0.95, max_tokens=4096),
        pins=Pins(
            verifiers="d30a3f48",
            research_environments="f9c43a74",
            math_verify="0.8.0",
            dataset_revision="math500=abc123",
        ),
        fingerprint=FINGERPRINT,
    )
    return RunManifest(**{**fields, **overrides})


def protocol_task(name: str = "math500", *, exact_available: bool = True) -> TaskSummary:
    records = [
        ProtocolRecord(
            protocol="exact",
            anchor="hendrycks/math" if exact_available else None,
            unavailable_reason=None if exact_available else "no published grader",
        ),
        ProtocolRecord(protocol="reference", anchor="math-verify"),
        ProtocolRecord(
            protocol="lenient",
            anchor="math-verify",
            relaxes="exact" if exact_available else "reference",
            waivers=("boxed_missing",),
        ),
    ]
    metrics = {
        "reference": Interval(value=0.5, low=0.4, high=0.6),
        "lenient": Interval(value=0.6, low=0.5, high=0.7),
        "truncated": Interval(value=0.1, low=0.05, high=0.2),
    }
    if exact_available:
        metrics["exact"] = Interval(value=0.45, low=0.35, high=0.55)
    return TaskSummary(
        task=name,
        problems=10,
        group_size=4,
        metric_set=PROTOCOL_METRICS,
        protocols=records,
        **metrics,
    )


def k_task(name: str = "math500", *, exact_available: bool = True) -> TaskSummary:
    """The same taskset with the k columns aggregation writes for a group of four.

    `pass@1` is the headline's own interval rather than a second number, which
    is what `_k_intervals` produces: the same estimator at k=1, drawn at the
    same seed. `pass@32` and `pass^32` are declared by the metric set and are
    **not** here, because four rollouts cannot answer them -- which is the state
    D-107 governs.
    """
    task = protocol_task(name, exact_available=exact_available)
    measured = dict(task.metrics)
    measured["pass@1"] = measured["exact" if exact_available else "reference"]
    measured["pass@4"] = Interval(value=0.7, low=0.6, high=0.8)
    measured["pass^4"] = Interval(value=0.2, low=0.1, high=0.3)
    return TaskSummary(
        task=name,
        problems=10,
        group_size=4,
        metric_set=PROTOCOL_METRICS,
        protocols=task.protocols,
        **measured,
    )


def write_run_dir(root, summary: RunSummary) -> None:
    run_dir = root / summary.manifest.run_id / "metrics"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))


def test_every_row_carries_its_profile_and_scoring_protocol(tmp_path) -> None:
    """A relaxation is never indexed without the base it relaxed."""
    write_run_dir(tmp_path, RunSummary(manifest=manifest(), tasks=[protocol_task()]))
    index = collect(tmp_path)

    (run,) = index["runs"]
    assert run["profile"] == "chat"
    entries = run["tasksets"][0]["metrics"]
    assert entries["lenient"]["protocol"] == "lenient (relaxes exact: boxed_missing)"
    assert entries["exact"]["protocol"] == "exact"
    assert entries["exact"]["anchor"] == "hendrycks/math"


def test_an_unavailable_protocol_is_absent_with_its_reason(tmp_path) -> None:
    summary = RunSummary(
        manifest=manifest(relaxation_bases="aime24=reference"),
        tasks=[protocol_task("aime24", exact_available=False)],
    )
    write_run_dir(tmp_path, summary)
    index = collect(tmp_path)

    entry = index["runs"][0]["tasksets"][0]["metrics"]["exact"]
    assert entry["value"] is None
    assert entry["unavailable_reason"] == "no published grader"


def test_comparison_groups_split_exactly_where_the_schema_refuses() -> None:
    same_a = manifest(run_id="a")
    same_b = manifest(run_id="b", profile="base-kshot")  # profile is not a blocker
    greedy = manifest(run_id="c", sampling_policy="greedy")
    failed = manifest(run_id="d", truncation_policy="fail")
    newer_engine = manifest(run_id="e", expected_vllm_version="0.27.1")
    assert comparison_groups([same_a, same_b, greedy, failed, newer_engine]) == [0, 0, 1, 2, 3]


def test_csv_preserves_rendering_conditions_within_a_comparison_group(tmp_path) -> None:
    """Metric compatibility does not collapse different rendering conditions."""
    runs = [
        manifest(run_id="limite-before"),
        manifest(
            run_id="limite-after",
            fingerprint=FINGERPRINT.model_copy(update={"template_sha256": "c" * 64}),
        ),
        manifest(
            run_id="qwen-base",
            checkpoint="/work/qwen",
            profile="base-kshot",
            fingerprint=FINGERPRINT.model_copy(
                update={
                    "architecture": "Qwen2ForCausalLM",
                    "artifact_sha256": "d" * 64,
                    "template_sha256": "e" * 64,
                }
            ),
        ),
    ]
    for run in runs:
        write_run_dir(tmp_path, RunSummary(manifest=run, tasks=[protocol_task()]))

    index = collect(tmp_path)
    _, csv_path = write_index(index, tmp_path)
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))

    expected = {
        run.run_id: (run.checkpoint, run.profile, run.fingerprint.template_sha256)
        for run in runs
    }
    assert {row["run_id"] for row in rows} == set(expected)
    assert {row["comparison_group"] for row in rows} == {"0"}
    for row in rows:
        assert (row["checkpoint"], row["profile"], row["template_sha256"]) == expected[row["run_id"]]
    assert len([row for row in rows if row["metric"] == "exact"]) == len(runs)


def test_the_index_is_written_as_json_and_long_format_csv(tmp_path) -> None:
    write_run_dir(tmp_path, RunSummary(manifest=manifest(), tasks=[protocol_task()]))
    json_path, csv_path = write_index(collect(tmp_path), tmp_path)

    run = json.loads(json_path.read_text())["runs"][0]
    assert run["comparison_group"] == 0
    assert run["expected_vllm_version"] == "0.26.0"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) == set(CSV_COLUMNS)
    assert rows[0]["expected_vllm_version"] == "0.26.0"
    # Every declared metric has a row, measured or not, and the unmeasured one
    # carries its reason rather than a zero -- the k columns the metric set
    # declares among them, since they are metrics of the same table.
    assert {row["metric"] for row in rows} == set(PROTOCOL_METRICS.metrics) | {
        "pass@1",
        "pass@4",
        "pass@32",
        "pass^4",
        "pass^32",
    }
    permissive = next(row for row in rows if row["metric"] == "permissive")
    assert permissive["value"] == "" and permissive["unavailable_reason"] == "not measured"


def test_indexing_an_out_root_drags_in_no_plotting_library() -> None:
    """The index reads the aggregation, never a rendering surface.

    `collect` is what gets run on a compute node the moment a matrix finishes,
    and a compute node mounts the shared home read-only -- which is why
    `AGENTS.md` lists `MPLCONFIGDIR` among the exports at all. Importing
    `report` for one sentence made the cross-run indexer depend on matplotlib
    and on a cache variable it never needed: it degrades to a temporary
    directory rather than failing, which is precisely how it would come back
    unnoticed. Asserted in a subprocess because pytest has imported `report`
    long before this runs, so an in-process check would pass either way.
    """
    probe = (
        "import limite_evals.collect, sys;"
        "print(sorted({'matplotlib', 'limite_evals.report'} & set(sys.modules)))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert loaded.stdout.strip() == "[]"


def test_a_k_row_names_the_protocol_it_was_computed_on(tmp_path) -> None:
    """D-06 computes the k columns on the headline alone, and the headline is per taskset.

    `exact` where a published grader is bound and `reference` where none is, so
    two tasksets of one run report their k at two anchors. Prose in one report
    cannot carry that across a matrix; the `protocol` and `anchor` columns can.
    """
    write_run_dir(
        tmp_path,
        RunSummary(
            manifest=manifest(pass_at_k="aime24=1+4+32,math500=1+4+32"),
            tasks=[k_task(), k_task("aime24", exact_available=False)],
        ),
    )
    math500, aime24 = collect(tmp_path)["runs"][0]["tasksets"]

    assert math500["metrics"]["pass@4"]["protocol"] == "exact"
    assert math500["metrics"]["pass@4"]["anchor"] == "hendrycks/math"
    assert aime24["metrics"]["pass@4"]["protocol"] == "reference"
    assert aime24["metrics"]["pass@4"]["anchor"] == "math-verify"


def test_a_k_above_the_group_size_is_absent_with_its_reason_never_zero(tmp_path) -> None:
    """D-107 on the k columns, in the report's own wording rather than a second one."""
    write_run_dir(tmp_path, RunSummary(manifest=manifest(), tasks=[k_task()]))
    json_path, csv_path = write_index(collect(tmp_path), tmp_path)

    entry = json.loads(json_path.read_text())["runs"][0]["tasksets"][0]["metrics"]["pass@32"]
    assert entry["value"] is None and entry["low"] is None and entry["high"] is None
    assert (
        entry["unavailable_reason"]
        == "this taskset drew 4 rollouts per problem, and `pass@32` needs 32"
    )
    with csv_path.open() as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert rows["pass^32"]["value"] == ""
    assert rows["pass^32"]["unavailable_reason"].endswith("`pass^32` needs 32")
    # The row exists at all: a k nobody asked and a k somebody forgot are the
    # same absence to a reader who has only the rows that are present.
    assert rows["pass@32"]["protocol"] == "exact"


def test_a_k_row_carries_the_interval_around_its_value(tmp_path) -> None:
    """An estimate that arrives without its interval is the point estimate this instrument avoids."""
    write_run_dir(tmp_path, RunSummary(manifest=manifest(), tasks=[k_task()]))
    metrics = collect(tmp_path)["runs"][0]["tasksets"][0]["metrics"]

    assert (metrics["pass@4"]["value"], metrics["pass@4"]["low"], metrics["pass@4"]["high"]) == (
        0.7,
        0.6,
        0.8,
    )
    # `pass@1` is the headline column itself -- same estimator, same interval,
    # same scoring identity -- so the two rows differ in nothing but the name.
    assert metrics["pass@1"] == metrics["exact"]


def test_a_metric_set_declaring_no_k_set_gains_no_rows(tmp_path) -> None:
    """Every run written before D-06 indexes exactly the metrics it reported then."""
    task = TaskSummary(
        task="math500",
        problems=10,
        group_size=4,
        metric_set=LADDER,
        strict=Interval(value=0.5, low=0.4, high=0.6),
    )
    write_run_dir(tmp_path, RunSummary(manifest=manifest(), tasks=[task]))
    metrics = collect(tmp_path)["runs"][0]["tasksets"][0]["metrics"]

    assert set(metrics) == set(LADDER.metrics)


def _fragment(
    run_id: str,
    task_name: str,
    *,
    max_tokens: int = 4096,
    pins: Pins | None = None,
    **overrides,
) -> RunSummary:
    return RunSummary(
        manifest=manifest(
            run_id=run_id,
            relaxation_bases=f"{task_name}=exact",
            sampling=Sampling(temperature=0.6, top_p=0.95, max_tokens=max_tokens),
            pins=pins
            or Pins(
                verifiers="d30a3f48",
                research_environments="f9c43a74",
                math_verify="0.8.0",
                dataset_revision=f"{task_name}=rev-{task_name}",
            ),
            **overrides,
        ),
        tasks=[protocol_task(task_name)],
    )


def test_merge_recombines_the_per_taskset_fields(tmp_path) -> None:
    merged = merge_summaries(
        [
            _fragment("r-m500", "math500"),
            _fragment(
                "r-gsm",
                "gsm8k",
                max_tokens=2048,
                pins=Pins(
                    verifiers="d30a3f48",
                    research_environments="f9c43a74",
                    math_verify="0.8.0",
                    dataset_revision="gsm8k=rev-gsm8k",
                    task_prompt_revision="gsm8k=openai-prompt",
                    native_scorer_revision="gsm8k=native-v1",
                ),
            ),
        ]
    )

    assert merged.manifest.run_id == "r-m500+r-gsm"
    assert [task.task for task in merged.tasks] == ["math500", "gsm8k"]
    assert merged.manifest.pins.dataset_revision == "gsm8k=rev-gsm8k,math500=rev-math500"
    assert merged.manifest.pins.task_prompt_revision == "gsm8k=openai-prompt"
    assert merged.manifest.pins.native_scorer_revision == "gsm8k=native-v1"
    assert merged.manifest.relaxation_bases == "gsm8k=exact,math500=exact"
    # The largest effective budget, exactly as the CLI records it for one run.
    assert merged.manifest.sampling.max_tokens == 4096


def test_merge_recombines_the_k_set_rather_than_carrying_the_first_fragment(tmp_path) -> None:
    """`pass_at_k` is per taskset, and `comparable_with` gates the merged value."""
    merged = merge_summaries(
        [
            _fragment("r-m500", "math500", pass_at_k="math500=1+4+32"),
            _fragment("r-gsm", "gsm8k", pass_at_k="gsm8k=1"),
        ]
    )

    assert merged.manifest.pass_at_k == "gsm8k=1,math500=1+4+32"


def test_merge_refuses_two_fragments_that_disagree_about_a_taskset_k_set() -> None:
    with pytest.raises(ValueError, match="pass_at_k disagrees at math500"):
        merge_summaries(
            [
                _fragment("a", "math500", pass_at_k="math500=1+4"),
                _fragment("b", "gsm8k", pass_at_k="math500=1+4+32"),
            ]
        )


def test_a_row_carries_the_k_set_it_was_measured_at(tmp_path) -> None:
    """A comparability key the index groups on has to be on the row it grouped."""
    write_run_dir(
        tmp_path, RunSummary(manifest=manifest(pass_at_k="math500=1+4+32"), tasks=[protocol_task()])
    )
    json_path, csv_path = write_index(collect(tmp_path), tmp_path)

    assert json.loads(json_path.read_text())["runs"][0]["pass_at_k"] == "math500=1+4+32"
    with csv_path.open() as handle:
        assert {row["pass_at_k"] for row in csv.DictReader(handle)} == {"math500=1+4+32"}


def test_runs_declaring_different_k_sets_are_different_comparison_groups() -> None:
    assert comparison_groups(
        [
            manifest(run_id="a", pass_at_k="math500=1+4+32"),
            manifest(run_id="b", pass_at_k="math500=1+4+32"),
            manifest(run_id="c", pass_at_k="math500=1+4"),
        ]
    ) == [0, 0, 1]


def test_merge_refuses_manifests_that_differ_beyond_the_taskset_list() -> None:
    with pytest.raises(ValueError, match="different checkpoint"):
        merge_summaries(
            [_fragment("a", "math500"), _fragment("b", "gsm8k", checkpoint="/work/other")]
        )


def test_merge_refuses_a_taskset_scored_twice() -> None:
    with pytest.raises(ValueError, match="math500 appears in both"):
        merge_summaries([_fragment("a", "math500"), _fragment("b", "math500")])


def test_collect_merges_on_request_and_groups_the_result(tmp_path) -> None:
    write_run_dir(tmp_path, _fragment("r-m500", "math500"))
    write_run_dir(tmp_path, _fragment("r-gsm", "gsm8k"))
    index = collect(tmp_path, merges=[["r-m500", "r-gsm"]])

    merged = next(run for run in index["runs"] if run.get("merged_from"))
    assert merged["run_id"] == "r-m500+r-gsm"
    assert merged["merged_from"] == ["r-m500", "r-gsm"]
    assert len(merged["tasksets"]) == 2


def test_collect_refuses_a_merge_of_unknown_runs(tmp_path) -> None:
    write_run_dir(tmp_path, _fragment("r-m500", "math500"))
    with pytest.raises(ValueError, match="no-such-run"):
        collect(tmp_path, merges=[["r-m500", "no-such-run"]])


def test_a_partial_summary_is_indexed_as_incomplete(tmp_path) -> None:
    run_dir = tmp_path / "r-partial" / "metrics"
    run_dir.mkdir(parents=True)
    summary = RunSummary(manifest=manifest(run_id="r-partial"), tasks=[protocol_task()])
    (run_dir / "partial-summary.json").write_text(summary.model_dump_json())
    index = collect(tmp_path)

    assert index["runs"][0]["complete"] is False
