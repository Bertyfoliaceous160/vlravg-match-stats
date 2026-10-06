"""BeyondAIME's frozen data, rendering, scoring, and aggregation contract."""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import version
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from limite_evals import aggregate, cli, profiles, suites, tasksets, templates
from limite_evals.reference_models import find as find_reference_model
from limite_evals_core import ladder, protocols
from limite_evals_core.equivalence import Comparison
from limite_evals_core.protocols import Unavailable
from limite_evals_core.protocols import score as score_protocols
from limite_evals_core.schema import SampleResult

DATASET_REVISION = "c705198ae1043810b1e1693bd879250b51a7a523"
PARQUET_SHA256 = "969750458ae4796e783b6b5ae004cf48587f10683bd5b5ac00220c50561e1df6"


def _spec() -> suites.TasksetSpec:
    return suites.taskset("math-extended", "beyondaime")


def test_beyondaime_is_a_math_extended_taskset_only() -> None:
    """BeyondAIME was the eighth taskset when it landed and is no longer the last
    one, which is exactly the fact this assertion has to keep tracking rather than
    pin: `apex-shortlist` and `hmmt26` were appended after it. What has not moved
    is its position -- eighth, directly after `olympiadbench` -- and its absence
    from the frozen `math-standard` four."""
    standard = suites.MATH_STANDARD
    extended = suites.resolve("math-extended")

    assert len(standard) == 4
    assert extended[: len(standard)] == standard
    assert [spec.name for spec in extended] == [
        "aime24",
        "aime25",
        "math500",
        "gsm8k",
        "aime26",
        "hmmt25",
        "olympiadbench",
        "beyondaime",
        "apex-shortlist",
        "hmmt26",
    ]
    assert extended[7].name == "beyondaime"
    assert "beyondaime" not in {spec.name for spec in standard}


def test_beyondaime_spec_and_prompt_bytes_are_pinned() -> None:
    spec = _spec()
    prompt, gold = tasksets.render_row(spec, {"problem": "What is 6 times 7?", "answer": 42})

    assert spec.dataset == "ByteDance-Seed/BeyondAIME"
    assert spec.split == "test"
    assert spec.revision == DATASET_REVISION
    assert spec.verifiers_id == "beyondaime-v1"
    assert spec.group_size == 32
    assert spec.answer_format == "boxed"
    assert spec.instruction is suites.MATH500_INSTRUCTION
    assert spec.instruction_position == "append"
    assert spec.prompt_revision == "limite-evals:beyondaime-math500-boxed-v1"
    assert spec.licence is None
    assert spec.caveat is not None
    assert "Limite-owned" in spec.caveat
    assert "not a replication of upstream Table 2" in spec.caveat
    assert "verified per-row difficulty" in spec.caveat
    assert prompt == (
        "What is 6 times 7?\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    assert gold == "42"


def test_beyondaime_manifest_pins_include_data_and_prompt_revisions() -> None:
    spec = _spec()

    assert cli._dataset_revisions((spec,)) == f"beyondaime={DATASET_REVISION}"
    assert cli._task_prompt_revisions((spec,)) == ("beyondaime=limite-evals:beyondaime-math500-boxed-v1")


def test_beyondaime_reports_reference_avg_at_32_without_pass_columns() -> None:
    spec = _spec()
    metric_set = spec.metrics

    assert metric_set is suites.BEYONDAIME_METRICS
    assert metric_set.headline == "reference"
    assert metric_set.metrics == (
        "reference",
        "exact",
        "lenient",
        "permissive",
        "format_ok",
        "truncated",
    )
    assert metric_set.pass_at_k == ()

    samples = [
        SampleResult(
            task="beyondaime",
            problem_index=0,
            rollout_index=rollout,
            gold="42",
            completion="\\boxed{42}" if rollout < 16 else "\\boxed{41}",
            finish_reason="stop",
            metrics={
                "reference": float(rollout < 16),
                "lenient": float(rollout < 16),
                "permissive": float(rollout < 16),
            },
            strict=rollout < 16,
            lenient=rollout < 16,
            permissive=rollout < 16,
            format_ok=True,
        )
        for rollout in range(32)
    ]
    summary = aggregate.summarize_task(
        "beyondaime", samples, metric_set=metric_set, answer_format="boxed", resamples=50
    )

    assert summary.group_size == 32
    assert summary.reference.value == 0.5
    assert not any(name.startswith("pass") for name in summary.metrics)


def test_beyondaime_exact_is_explicitly_unavailable_and_reference_is_strict() -> None:
    bindings = tasksets.taskset_protocols(_spec())

    assert isinstance(bindings.exact, Unavailable)
    assert "publishes problems and gold answers" in bindings.exact.reason
    assert "no executable grader" in bindings.exact.reason
    assert "extraction" in bindings.exact.reason
    assert "equivalence" in bindings.exact.reason

    completion = "<think>Earlier \\boxed{7}</think>First \\boxed{41}; finally \\boxed{42}."
    scored = score_protocols(completion, "42", bindings)
    observed = ladder.score(completion, "42")

    assert scored["exact"].correct is None
    assert scored["exact"].unavailable_reason == bindings.exact.reason
    assert scored["reference"].correct is True
    assert observed.strict_answer == "42"
    assert observed.strict is True
    assert observed.format_ok is True


def test_beyondaime_relaxations_use_the_same_math_verify_comparator() -> None:
    bindings = tasksets.taskset_protocols(_spec())

    lenient = score_protocols("Working through it, the final answer is 1/2", "\\frac{1}{2}", bindings)
    permissive = score_protocols(
        "<think>The candidate was \\boxed{42}.</think>I could not finish.", "42", bindings
    )

    assert lenient["reference"].correct is False
    assert lenient["lenient"].correct is True
    assert lenient["lenient"].relaxes == "reference"
    assert permissive["reference"].correct is False
    assert permissive["lenient"].correct is False
    assert permissive["permissive"].correct is True
    assert permissive["permissive"].relaxes == "reference"


def test_beyondaime_parse_failures_are_wrong_without_a_timeout() -> None:
    bindings = tasksets.taskset_protocols(_spec())
    completion = r"\boxed{\not valid latex {{{}"

    scored = score_protocols(completion, "42", bindings)
    observed = ladder.score(completion, "42")

    assert scored["reference"].correct is False
    assert scored["lenient"].correct is False
    assert scored["permissive"].correct is False
    assert observed.comparison_timeout is False


def test_beyondaime_timeout_is_recorded_separately_and_scores_wrong(monkeypatch) -> None:
    monkeypatch.setattr(protocols, "equivalent", lambda _gold, _candidate: False)
    monkeypatch.setattr(
        ladder,
        "compare",
        lambda _gold, _candidate: Comparison(equal=False, timed_out=True),
    )

    completion = "\\boxed{42}"
    scored = score_protocols(completion, "42", tasksets.taskset_protocols(_spec()))
    observed = ladder.score(completion, "42")

    assert scored["reference"].correct is False
    assert scored["lenient"].correct is False
    assert scored["permissive"].correct is False
    assert (observed.strict, observed.lenient, observed.permissive) == (False, False, False)
    assert observed.format_ok is True
    assert observed.comparison_timeout is True


def test_beyondaime_uses_the_existing_math_verify_090_pin() -> None:
    assert version("math-verify") == "0.9.0"


def test_beyondaime_has_both_base_profile_mappings() -> None:
    assert profiles.EXEMPLAR_SETS["beyondaime"] == "math"
    assert templates.KSHOT_TEMPLATES["beyondaime"] == "base-kshot-math"
    assert templates.for_profile("base-kshot", "beyondaime") == "base-kshot-math"


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("WeiboAI/VibeThinker-3B", "77bd2cced09193c8b9a59a32bd8577bbd1f3e01c"),
        ("WeiboAI/VibeThinker-1.5B", "9614bcb0264f03f18f2fa2406c4a9754635f51de"),
    ],
)
def test_requested_vibethinker_models_are_admitted_at_immutable_revisions(
    repo_id: str, revision: str
) -> None:
    contract = find_reference_model(repo_id)

    assert contract is not None
    assert contract.identity == f"{repo_id}@{revision}"
    assert contract.architecture == "Qwen2ForCausalLM"
    assert contract.model_type == "qwen2"
    assert contract.runtime_status == "native"
    assert cli.token_budget(131072) == 126976


def test_pinned_beyondaime_dataset_cardinality_schema_and_parquet_digest() -> None:
    spec = _spec()
    try:
        parquet = Path(
            hf_hub_download(
                repo_id=spec.dataset,
                filename="data/test.parquet",
                repo_type="dataset",
                revision=spec.revision,
                token=os.environ.get("HF_TOKEN") or None,
            )
        )
        rows = tasksets.load_raw_rows(spec)
    except Exception as exc:  # noqa: BLE001 - offline parity is an explicit skip
        pytest.skip(f"BeyondAIME not available at the pinned revision: {exc}")

    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == PARQUET_SHA256
    assert len(rows) == 100
    assert {tuple(sorted(row)) for row in rows} == {("answer", "problem")}
    assert all(isinstance(row["problem"], str) for row in rows)
    assert all(isinstance(row["answer"], int) for row in rows)
