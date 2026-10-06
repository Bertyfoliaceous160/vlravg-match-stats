"""Scoring a finished run again: what it recomputes, and what it must not claim."""

from __future__ import annotations

import json

import pytest

from limite_evals import rescore
from limite_evals.aggregate import summarize
from limite_evals.report import write_run
from limite_evals_core.schema import (
    SCORER_VERSION,
    Fingerprint,
    InstrumentEraMismatch,
    Pins,
    RunManifest,
    RunSummary,
    SampleResult,
    Sampling,
)
FINGERPRINT = Fingerprint(
    architecture="LimiteForCausalLM",
    artifact_sha256="a" * 64,
    template_sha256="b" * 64,
    max_model_len=8192,
    dtype="bfloat16",
    vllm_version="0.26.0",
    engine_venv="/home/user/paradigma/project/limite-post-train/.venv",
)

PINS = Pins(
    verifiers="d30a3f48",
    research_environments="f9c43a74",
    math_verify="0.9.0",
    dataset_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
)

MANIFEST = RunManifest(
    run_id="r001",
    created_at="2026-08-07T00:00:00Z",
    checkpoint="/work/ckpt",
    stage="sft",
    profile="chat",
    suite="math-extended",
    # Deliberately a lie about the tasksets below, so the re-derivation is
    # visible: a rescore that carried this string over would keep it.
    relaxation_bases="aime24=reference,math500=reference",
    sampling=Sampling(
        temperature=0.0, top_p=1.0, max_tokens=4096, stop=[], skip_special_tokens=True
    ),
    pins=PINS,
    fingerprint=FINGERPRINT,
)


def _sample(task: str, problem: int, *, gold: str, completion: str, **overrides) -> SampleResult:
    fields = {
        "task": task,
        "problem_index": problem,
        "rollout_index": 0,
        "gold": gold,
        "completion": completion,
        "finish_reason": "stop",
        "template_sha256": "b" * 64,
    }
    return SampleResult(**{**fields, **overrides})


def _samples() -> list[SampleResult]:
    """Rows carrying a completion and a gold and nothing else true about them.

    Every verdict field is left at its default, so anything the rescored rows
    carry was computed from the two strings rather than copied.
    """
    return [
        _sample("math500", problem, gold="42", completion="The answer is \\boxed{42}.")
        for problem in range(4)
    ] + [
        _sample("aime24", problem, gold="42", completion="Thus \\boxed{7}.")
        for problem in range(4)
    ]


def _written(tmp_path):
    samples = _samples()
    return write_run(summarize(MANIFEST, samples, resamples=50), samples, root=tmp_path)


def _truncated_samples() -> list[SampleResult]:
    """One of math500's four rollouts ran out of budget with an answer in it."""
    samples = _samples()
    samples[0] = _sample(
        "math500",
        0,
        gold="42",
        completion="The answer is \\boxed{42}.",
        finish_reason="length",
        truncated=True,
        completion_tokens=4096,
    )
    return samples


def _written_with_truncation(tmp_path, root=None):
    samples = _truncated_samples()
    return write_run(
        summarize(MANIFEST, samples, resamples=50), samples, root=root or tmp_path
    )


def test_a_verdict_is_recomputed_from_the_completion(tmp_path) -> None:
    run_dir = _written(tmp_path)

    rescore.rescore_run(run_dir, tmp_path / "out", workers=2)
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metrics" / "samples.jsonl").read_text().splitlines()
    ]

    correct = [row for row in rows if row["task"] == "math500"]
    wrong = [row for row in rows if row["task"] == "aime24"]
    assert all(row["metrics"]["reference"] == 1.0 for row in correct)
    assert all(row["metrics"]["reference"] == 0.0 for row in wrong)
    # The stored rows carried False at every rung; these came from the text.
    assert all(row["strict"] for row in correct)
    assert all(row["strict_answer"] == "42" for row in correct)


def test_how_the_completion_was_produced_is_carried_not_invented(tmp_path) -> None:
    """A rescore cannot observe the engine, so it must copy what the run recorded.

    `truncated` above all: decision 10 reads it at the aggregation boundary to
    fail a cut-off completion at every correctness metric, and a rescore that
    dropped it would raise every number by forgiving exactly the rollouts that
    ran out of context.
    """
    run_dir = _written_with_truncation(tmp_path)

    summary = rescore.rescore_run(run_dir, tmp_path / "out", workers=2)
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metrics" / "samples.jsonl").read_text().splitlines()
    ]

    truncated = next(row for row in rows if row["task"] == "math500" and row["problem_index"] == 0)
    assert truncated["truncated"] is True
    assert summary.manifest.truncation_policy == "fail"
    assert truncated["finish_reason"] == "length"
    assert truncated["completion_tokens"] == 4096
    # Scored correct on its own text, and failed anyway because it was cut off.
    assert truncated["metrics"]["reference"] == 1.0
    math500 = next(task for task in summary.tasks if task.task == "math500")
    assert math500.reference.value == pytest.approx(0.75)


def test_the_manifest_says_it_is_a_rescore_and_re_derives_the_bases(tmp_path) -> None:
    run_dir = _written(tmp_path)

    summary = rescore.rescore_run(run_dir, tmp_path / "out", workers=2)

    assert summary.manifest.run_id == "r001" + rescore.RESCORE_SUFFIX
    # Re-derived from this checkout's bindings rather than copied: math500 binds a
    # published grader, so its relaxations relax `exact`, not `reference`.
    assert summary.manifest.relaxation_bases == "aime24=reference,math500=exact"
    # And the changed base is what stops the two comparing, which is the point:
    # `lenient(exact)` and `lenient(reference)` under one column heading are two
    # quantities, so the rescored numbers must not read as the run's own.
    #
    # The base is the *only* reason, and the k set deliberately is not a second
    # one. This rescore declares `math500=1+4+32` where the run it re-reads
    # predates the field and carries nothing, and that is an absent declaration
    # rather than a different one: the k columns are additional readings of the
    # same rollouts, so every column the two runs share means what it always
    # meant. `RunManifest.comparable_with` is where the distinction from the
    # relaxation base is argued.
    assert summary.manifest.comparable_with(MANIFEST) == [
        "different relaxation base: aime24=reference,math500=exact "
        "vs aime24=reference,math500=reference"
    ]


def test_a_rescore_keeps_the_run_own_truncation_policy(tmp_path) -> None:
    """A rescore moves the graders and holds everything else still.

    The manifest here states `fail`, and inheriting the instrument's current
    default instead would restate this run's numbers under a rule they were never
    computed with -- silently, for every run in an output directory.
    """
    run_dir = _written_with_truncation(tmp_path)
    assert rescore.read_manifest(run_dir).truncation_policy == "fail"

    summary = rescore.rescore_run(run_dir, tmp_path / "out", workers=2)

    assert summary.manifest.truncation_policy == "fail"
    math500 = next(task for task in summary.tasks if task.task == "math500")
    assert math500.reference.value == pytest.approx(0.75)


def test_asking_for_the_other_policy_recomputes_and_records_it(tmp_path) -> None:
    """The same stored rollouts, the other quantity, said so in the manifest."""
    run_dir = _written_with_truncation(tmp_path)

    summary = rescore.rescore_run(
        run_dir, tmp_path / "out", workers=2, truncation_policy="score"
    )

    assert summary.manifest.truncation_policy == "score"
    math500 = next(task for task in summary.tasks if task.task == "math500")
    assert math500.reference.value == pytest.approx(1.0)


def test_the_run_directory_is_left_exactly_as_it_was(tmp_path) -> None:
    run_dir = _written(tmp_path)
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    rescore.rescore_run(run_dir, tmp_path / "out", workers=2)

    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_a_rescored_directory_is_not_rescored_over(tmp_path) -> None:
    run_dir = _written(tmp_path)
    out = tmp_path / "out"
    rescore.main([str(run_dir), "--out", str(out), "--workers", "2"])

    with pytest.raises(SystemExit):
        rescore.main([str(run_dir), "--out", str(out), "--workers", "2"])


def test_a_partial_run_is_rescored_from_its_own_manifest(tmp_path) -> None:
    """An interrupted run carries `partial-summary.json` and no `summary.json`."""
    samples = [row for row in _samples() if row.task == "math500"]
    run_dir = _written(tmp_path)
    (run_dir / "metrics" / "summary.json").rename(run_dir / "metrics" / "partial-summary.json")

    manifest = rescore.read_manifest(run_dir)
    summary = rescore.rescore_run(run_dir, tmp_path / "out", workers=2)

    assert manifest.suite == "math-extended"
    assert [task.task for task in summary.tasks] == ["math500", "aime24"]
    assert len(samples) == 4


def test_a_summary_read_back_is_the_one_that_was_returned(tmp_path) -> None:
    run_dir = _written(tmp_path)

    summary = rescore.rescore_run(run_dir, tmp_path / "out", workers=2)

    written = RunSummary.model_validate_json(
        (tmp_path / "out" / "metrics" / "summary.json").read_text()
    )
    assert written == summary
    assert (tmp_path / "out" / "reports" / "summary.md").exists()
    assert (tmp_path / "out" / "plots" / "ladder.png").exists()


# --- the era boundary: what may be rescored, and what may not ----------------


#: A run written by the E42-era instrument: it recorded neither the vocabulary
#: it served through nor the decode its graders read completions back with,
#: because neither field existed yet. Built by taking them off rather than by
#: declaring an era, since the era is derived and cannot be declared.
E42_MANIFEST = MANIFEST.model_copy(
    update={
        "run_id": "res339-ev0-baseline-002",
        "sampling": Sampling(temperature=0.0, top_p=1.0, max_tokens=4096, stop=[]),
    }
)


def _think_samples() -> list[SampleResult]:
    """Corrected-era rollouts: the delimiters are in the text, so the lenses see them.

    One of each shape the new semantics turn on -- boxed answers after closed
    blocks and a completion that hit the cap still reasoning -- so a rescore of this file
    exercises the changed scorers rather than merely reproducing old verdicts.
    """
    return [
        _sample(
            "math500",
            0,
            gold="42",
            completion="<think>maybe 7</think> The answer is \\boxed{42}.",
        ),
        _sample(
            "math500",
            1,
            gold="42",
            completion="<think>I am still working, 42 maybe",
            finish_reason="length",
            truncated=True,
        ),
        _sample(
            "gsm8k",
            0,
            gold="42",
            completion="<think>so far \\boxed{18} here</think>\n\\boxed{42}",
        ),
    ]


def _written_with_think(tmp_path):
    samples = _think_samples()
    return write_run(
        summarize(
            MANIFEST.model_copy(update={"suite": "math-extended"}),
            samples,
            answer_formats={"math500": "boxed", "gsm8k": "boxed"},
            resamples=50,
        ),
        samples,
        root=tmp_path,
    )


def test_a_corrected_era_run_rescores_cleanly_under_the_new_scorers(tmp_path) -> None:
    """The property the whole quarantine rests on being worth having.

    A corrected-era `samples.jsonl` carries the reasoning delimiters, so every
    changed scoring rule reaches an existing baseline through this path and no
    GPU time is spent: the boxed lens reads the answer region and ignores a
    provisional box inside the reasoning block, and the rollout that hit the cap still reasoning is marked not evaluable
    rather than scored as a formatting failure.
    """
    run_dir = _written_with_think(tmp_path)

    summary = rescore.rescore_run(run_dir, tmp_path / "out", workers=2)
    rows = {
        (row["task"], row["problem_index"]): row
        for row in (
            json.loads(line)
            for line in (tmp_path / "out" / "metrics" / "samples.jsonl").read_text().splitlines()
        )
    }

    assert summary.manifest.instrument_era == "corrected"
    assert summary.manifest.scorer_version == SCORER_VERSION

    boxed = rows[("math500", 0)]
    assert (boxed["strict_answer"], boxed["format_ok"]) == ("42", True)
    assert boxed["format_not_evaluable"] is False

    mid_reasoning = rows[("math500", 1)]
    assert mid_reasoning["format_not_evaluable"] is True
    assert mid_reasoning["format_ok"] is False

    gsm8k = rows[("gsm8k", 0)]
    assert gsm8k["metrics"]["reference"] == 1.0
    assert gsm8k["strict_answer"] == "42"

    # And the not-evaluable rollout left one denominator and nothing else.
    math500 = next(task for task in summary.tasks if task.task == "math500")
    assert math500.problems == 2
    assert math500.format_not_evaluable_rate == 0.5
    assert math500.metrics["format_ok"].value == 1.0


def test_every_rescored_row_carries_the_scorer_that_wrote_it(tmp_path) -> None:
    """Otherwise the case letters it writes become an unversioned cache."""
    run_dir = _written_with_think(tmp_path)

    rescore.rescore_run(run_dir, tmp_path / "out", workers=2)
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "metrics" / "samples.jsonl").read_text().splitlines()
    ]

    assert all(row["scorer_version"] == SCORER_VERSION for row in rows)
    assert all(row["failure_case"] is not None for row in rows)


def test_an_e42_era_run_is_refused_before_anything_is_scored(tmp_path) -> None:
    """The quarantine, at the boundary that costs nothing to refuse from.

    Refused on the manifest alone, so it happens before a sixteen-process pool
    reads a single completion -- and refused rather than warned about, because
    there is no reading of those completions that is worth producing.
    """
    samples = _samples()
    run_dir = write_run(
        summarize(MANIFEST, samples, resamples=50).model_copy(
            update={"manifest": E42_MANIFEST}
        ),
        samples,
        root=tmp_path,
    )

    with pytest.raises(InstrumentEraMismatch, match="destroyed at decode time"):
        rescore.read_manifest(run_dir)
    with pytest.raises(InstrumentEraMismatch, match="res339-ev0-baseline-002"):
        rescore.rescore_run(run_dir, tmp_path / "out", workers=2)

    # Enforcement is in the tooling: nothing on disk was touched, and the
    # refusal wrote no output directory to be mistaken for a rescore.
    assert (run_dir / "metrics" / "samples.jsonl").exists()
    assert not (tmp_path / "out").exists()


def test_a_re_summarisation_of_an_e42_run_is_refused_too(tmp_path) -> None:
    """The rescore path is not the only way to compute numbers from old rollouts.

    `aggregate.summarize` is the one function every summary in this repository
    is written by, so the refusal is there as well -- otherwise a caller holding
    the samples and the manifest could reach around the rescore entrypoint.
    """
    with pytest.raises(InstrumentEraMismatch, match="Regenerating the rollouts"):
        summarize(E42_MANIFEST, _samples(), resamples=50)
