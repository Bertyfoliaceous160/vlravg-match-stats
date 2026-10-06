"""The paired bootstrap: real gaps exclude zero, identity does not, absence counts nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from limite_evals.compare import compare, compare_runs, interval, load, paired_delta
from limite_evals_core.schema import SampleResult


def _sample(task: str, problem: int, rollout: int, correct: bool, **overrides) -> SampleResult:
    fields = dict(
        task=task,
        problem_index=problem,
        rollout_index=rollout,
        gold="42",
        completion="\\boxed{42}",
        finish_reason="stop",
        truncated=False,
        format_ok=True,
        completion_tokens=100,
        metrics={
            "exact": float(correct),
            "reference": float(correct),
            "lenient": float(correct),
            "permissive": float(correct),
        },
    )
    return SampleResult(**{**fields, **overrides})


def write_run_dir(root: Path, name: str, samples: list[SampleResult]) -> Path:
    run_dir = root / name / "metrics"
    run_dir.mkdir(parents=True)
    (run_dir / "samples.jsonl").write_text("".join(sample.model_dump_json() + "\n" for sample in samples))
    return root / name


def _samples(correct_problems: set[int], *, task: str = "math500") -> list[SampleResult]:
    return [
        _sample(task, problem, rollout, problem in correct_problems)
        for problem in range(8)
        for rollout in range(4)
    ]


def test_a_real_gap_excludes_zero(tmp_path) -> None:
    a_dir = write_run_dir(tmp_path, "a", _samples(set(range(8))))
    b_dir = write_run_dir(tmp_path, "b", _samples(set()))
    body = compare(load(a_dir), load(b_dir), "math500", "exact")

    assert body["problems"] == 8
    assert body["delta"]["value"] == 1.0
    assert body["delta"]["significant"] is True


def test_identical_runs_include_zero(tmp_path) -> None:
    a_dir = write_run_dir(tmp_path, "a", _samples({0, 1, 2}))
    b_dir = write_run_dir(tmp_path, "b", _samples({0, 1, 2}))
    body = compare(load(a_dir), load(b_dir), "math500", "exact")

    assert body["delta"]["value"] == 0.0
    assert body["delta"]["significant"] is False


def test_the_interval_reproduces_the_point_estimate(tmp_path) -> None:
    run = load(write_run_dir(tmp_path, "a", _samples({0, 1})))
    groups = [run["tasks"]["math500"][problem]["exact"] for problem in sorted(run["tasks"]["math500"])]
    assert interval(groups)["value"] == 2 / 8


def test_an_absent_column_contributes_nothing(tmp_path) -> None:
    """D-107 applied to a delta: no verdict is not a wrong answer."""
    samples = [
        _sample("aime24", problem, rollout, True, metrics={"reference": 1.0})
        for problem in range(4)
        for rollout in range(2)
    ]
    a_dir = write_run_dir(tmp_path, "a", samples)
    b_dir = write_run_dir(tmp_path, "b", samples)
    body = compare(load(a_dir), load(b_dir), "aime24", "exact")

    assert body["delta"] == {"value": 0.0, "low": 0.0, "high": 0.0, "significant": False}


def test_compare_runs_reads_two_directories_end_to_end(tmp_path) -> None:
    a_dir = write_run_dir(tmp_path, "a", _samples(set(range(6))))
    b_dir = write_run_dir(tmp_path, "b", _samples({0}))
    result = compare_runs(a_dir, b_dir, splits={"math500": {"first_half": [0, 1, 2, 3]}})

    assert result["a_run"] == "a" and result["b_run"] == "b"
    assert result["tasks"]["math500"]["exact"]["delta"]["significant"] is True
    assert result["splits"]["math500"]["first_half"]["exact"]["problems"] == 4
    assert result["efficiency"]["math500"]["a"]["rollouts"] == 32
    assert result["termination"]["a"]["truncated"] == 0.0


def test_paired_delta_drops_a_problem_only_one_side_scored() -> None:
    delta = paired_delta([[True], []], [[False], [True]])
    # One pair survives; too few to resample, so the interval is the point.
    assert delta["value"] == 1.0
    assert delta["significant"] is False


def test_native_scalar_comparison_preserves_point_units(tmp_path) -> None:
    a = [
        _sample("phybench", problem, 0, True, metrics={"eed": value})
        for problem, value in enumerate((0.0, 50.0, 100.0))
    ]
    b = [
        _sample("phybench", problem, 0, True, metrics={"eed": value})
        for problem, value in enumerate((0.0, 25.0, 50.0))
    ]
    body = compare(
        load(write_run_dir(tmp_path, "a", a)),
        load(write_run_dir(tmp_path, "b", b)),
        "phybench",
        "eed",
    )

    assert body["a"]["value"] == 50.0
    assert body["b"]["value"] == 25.0
    assert body["delta"]["value"] == 25.0


def test_compare_runs_includes_native_scalar_columns_without_a_protocol_flag(tmp_path) -> None:
    a = [
        _sample("phybench", problem, 0, True, metrics={"eed": value})
        for problem, value in enumerate((50.0, 100.0))
    ]
    b = [
        _sample("phybench", problem, 0, True, metrics={"eed": value})
        for problem, value in enumerate((25.0, 50.0))
    ]

    result = compare_runs(
        write_run_dir(tmp_path, "a", a),
        write_run_dir(tmp_path, "b", b),
    )

    assert set(result["tasks"]["phybench"]) == {"eed", "truncated"}
    assert result["tasks"]["phybench"]["eed"]["delta"]["value"] == 37.5


def test_source_group_metrics_are_refused_in_row_level_compare(tmp_path) -> None:
    grouped = [
        _sample(
            "abench_phy_b",
            variant,
            0,
            True,
            source_group_id="g0",
            source_variant_id=str(variant),
            metrics={"dynamic_accuracy": 1.0},
        )
        for variant in range(4)
    ]
    a = load(write_run_dir(tmp_path, "a", grouped))
    b = load(write_run_dir(tmp_path, "b", grouped))

    with pytest.raises(ValueError, match="source-group metric"):
        compare(a, b, "abench_phy_b", "dynamic_accuracy")


def test_compare_runs_skips_source_group_task_without_blocking_supported_tasks(
    tmp_path,
) -> None:
    grouped = [
        _sample(
            "abench_phy_b",
            variant,
            0,
            True,
            source_group_id="g0",
            source_variant_id=str(variant),
            metrics={"dynamic_accuracy": 1.0},
        )
        for variant in range(4)
    ]
    supported_a = [
        _sample("phybench", problem, 0, True, metrics={"eed": value})
        for problem, value in enumerate((50.0, 100.0))
    ]
    supported_b = [
        _sample("phybench", problem, 0, True, metrics={"eed": value})
        for problem, value in enumerate((25.0, 50.0))
    ]

    result = compare_runs(
        write_run_dir(tmp_path, "a", grouped + supported_a),
        write_run_dir(tmp_path, "b", grouped + supported_b),
    )

    assert set(result["tasks"]) == {"phybench"}
    assert result["tasks"]["phybench"]["eed"]["delta"]["value"] == 37.5
    assert set(result["unavailable_tasks"]) == {"abench_phy_b"}
    assert "aggregated summaries" in result["unavailable_tasks"]["abench_phy_b"]
