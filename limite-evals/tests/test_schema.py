"""The comparability and fingerprint rules, which are the schema's real content."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from limite_evals_core.schema import (
    LADDER,
    Fingerprint,
    InstrumentEraMismatch,
    MetricSet,
    NativeMetric,
    Pins,
    RunManifest,
    SampleResult,
    Sampling,
    TaskSummary,
)

BASE_FINGERPRINT = Fingerprint(
    architecture="LimiteForCausalLM",
    artifact_sha256="a" * 64,
    template_sha256="b" * 64,
    max_model_len=8192,
    dtype="bfloat16",
    vllm_version="0.26.0",
)

BASE_PINS = Pins(
    verifiers="d30a3f48",
    research_environments="f9c43a74",
    math_verify="0.8.0",
    dataset_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
)


def test_native_metric_and_source_identity_round_trip_without_moving_legacy_defaults() -> None:
    legacy = SampleResult(
        task="legacy",
        problem_index=0,
        rollout_index=0,
        gold="42",
        completion="42",
    )
    assert legacy.source_group_id is None
    assert legacy.source_variant_id is None
    assert LADDER.native_metrics == ()

    metric_set = MetricSet(
        name="physics",
        metrics=("exact", "eed"),
        native_metrics=(
            NativeMetric(
                name="exact",
                minimum=0.0,
                maximum=1.0,
                presentation="percent",
                provenance="official exact scorer",
                unavailable_reason="the benchmark publishes no exact accuracy",
            ),
            NativeMetric(
                name="eed",
                minimum=0.0,
                maximum=100.0,
                presentation="points",
                provenance="official EED scorer",
            ),
        ),
    )
    restored = MetricSet.model_validate_json(metric_set.model_dump_json())
    assert restored.native("eed").maximum == 100.0
    assert restored.native("exact").unavailable_reason == (
        "the benchmark publishes no exact accuracy"
    )

    identified = legacy.model_copy(
        update={"source_group_id": "group-7", "source_variant_id": "3"}
    )
    assert SampleResult.model_validate_json(identified.model_dump_json()) == identified


def _manifest(**overrides) -> RunManifest:
    fields = {
        "run_id": "r001",
        "created_at": "2026-08-07T00:00:00Z",
        "checkpoint": "/work/ckpt",
        "stage": "pretrain",
        "profile": "base-kshot",
        "suite": "math-standard",
        "sampling": Sampling(temperature=0.0, max_tokens=4096, skip_special_tokens=True),
        "pins": BASE_PINS,
        "fingerprint": BASE_FINGERPRINT,
    }
    return RunManifest(**{**fields, **overrides})


def test_a_different_rendering_profile_stays_comparable() -> None:
    """Comparing a base run against a chat run is the point of the repository."""
    base = _manifest(stage="pretrain", profile="base-kshot")
    chat = _manifest(run_id="r002", stage="posttrain", profile="chat")
    assert base.comparable_with(chat) == []


def test_a_manifest_written_before_the_policy_existed_reads_as_avg_at_k() -> None:
    """Every run predating the field drew the suite's own policy, so that is what it says."""
    assert _manifest().sampling_policy == "avg-k"


def test_a_greedy_run_is_not_comparable_with_an_avg_at_k_one() -> None:
    """One draw from the mode and the mean of a distribution are two quantities."""
    greedy = _manifest(run_id="r004", sampling_policy="greedy")
    assert _manifest().comparable_with(greedy) == [
        "different sampling policy: avg-k vs greedy"
    ]


def test_sampling_parameters_and_mtp_gate_comparability() -> None:
    baseline = _manifest()
    hotter = _manifest(
        run_id="r002",
        sampling=Sampling(temperature=0.9, top_p=1.0, max_tokens=4096, skip_special_tokens=True),
    )
    mtp = _manifest(
        run_id="r003",
        speculative_decoding={"method": "qwen3_5_mtp", "num_speculative_tokens": 2},
        compilation_config="mode0-full-decode-only",
    )
    assert baseline.comparable_with(hotter) == ["different temperature: 0.0 vs 0.9"]
    assert baseline.comparable_with(mtp) == [
        "different speculative decoding: none vs qwen3_5_mtp:2",
        "different compilation config: default vs mode0-full-decode-only",
    ]


def test_extended_sampling_parameters_gate_comparability() -> None:
    baseline = _manifest()
    model_default = _manifest(
        run_id="r002",
        sampling=Sampling(
            temperature=0.0,
            top_p=1.0,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_tokens=4096,
            skip_special_tokens=True,
        ),
    )
    assert baseline.comparable_with(model_default) == [
        "different top-k: none vs 20",
        "different min-p: none vs 0.0",
        "different presence penalty: none vs 0.0",
        "different repetition penalty: none vs 1.0",
    ]


def test_expected_engine_version_is_a_public_version_comparability_key() -> None:
    baseline = _manifest()
    newer = _manifest(run_id="r002", expected_vllm_version="0.27.1")
    local_build = _manifest(run_id="r003", expected_vllm_version="0.26.0+cu129")

    assert baseline.expected_vllm_version == "0.26.0"
    assert baseline.comparable_with(newer) == ["different expected vLLM version: 0.26.0 vs 0.27.1"]
    assert baseline.comparable_with(local_build) == []




def test_native_max_is_not_comparable_with_historical_reference_even_at_the_same_cap() -> None:
    """The coarse maximum alone cannot identify a taskset-reference measurement."""
    native = _manifest(run_id="r-native", generation_budget_policy="native-max")
    assert _manifest().comparable_with(native) == [
        "different generation budget policy: reference vs native-max"
    ]


def test_a_manifest_written_before_the_truncation_policy_reads_as_the_gate() -> None:
    """Every run predating the field was computed with truncation failing.

    The instrument's own default is the other one, and the disagreement is
    deliberate: here the field's job is to say what an older summary already
    means, so defaulting it forward would restate numbers nobody recomputed.
    """
    assert _manifest().truncation_policy == "fail"


def test_a_changed_truncation_policy_blocks_comparison() -> None:
    """The same completions read tens of points apart across the two policies.

    A column computed with truncated completions graded on their text and one
    computed with them failed are not one measurement at two confidence levels,
    so a run quoting one beside the other is what this refuses.
    """
    graded = _manifest(run_id="r005", truncation_policy="score")
    assert _manifest().comparable_with(graded) == [
        "different truncation policy: fail vs score"
    ]


def test_a_moved_pin_blocks_comparison() -> None:
    moved = _manifest(run_id="r003", pins=BASE_PINS.model_copy(update={"math_verify": "0.9.0"}))
    reasons = _manifest().comparable_with(moved)
    assert reasons == ["different math_verify pin: 0.8.0 vs 0.9.0"]


def test_a_revised_native_scorer_is_not_comparable_with_an_unrecorded_one() -> None:
    revised = _manifest(
        run_id="r-native-v1",
        pins=BASE_PINS.model_copy(
            update={"native_scorer_revision": "phybench=eed-upstream-plus-extractor-v1"}
        ),
    )

    assert _manifest().comparable_with(revised) == [
        "different native scorer revision pin: unrecorded vs "
        "phybench=eed-upstream-plus-extractor-v1"
    ]


def test_a_revised_task_prompt_is_not_comparable_with_an_unrecorded_one() -> None:
    """Unrecorded GSM8K cannot compare with the boxed prompt/scorer cutover."""
    revised = _manifest(
        run_id="r-gsm8k-boxed",
        pins=Pins(
            verifiers=BASE_PINS.verifiers,
            research_environments=BASE_PINS.research_environments,
            math_verify=BASE_PINS.math_verify,
            dataset_revision=BASE_PINS.dataset_revision,
            task_prompt_revision="gsm8k=limite-evals:gsm8k-math500-boxed-v1",
        ),
    )

    assert _manifest().comparable_with(revised) == [
        "different task prompt revision pin: unrecorded vs "
        "gsm8k=limite-evals:gsm8k-math500-boxed-v1"
    ]


@pytest.mark.parametrize(
    "previous_revision",
    [
        "openai/grade-school-math@3101c7d5072418e28b9008a6636bde82a006892c:question-lf",
        (
            "openai/grade-school-math@3101c7d5072418e28b9008a6636bde82a006892c:"
            "readme-answer-format+question-lf"
        ),
    ],
)
def test_boxed_gsm8k_is_not_comparable_with_either_previous_prompt(
    previous_revision: str,
) -> None:
    """Both prior GSM8K renderings belong to retired hashed scoring contracts."""
    previous = _manifest(
        run_id="r-gsm8k-previous",
        pins=BASE_PINS.model_copy(
            update={"task_prompt_revision": f"gsm8k={previous_revision}"}
        ),
    )
    boxed = _manifest(
        run_id="r-gsm8k-boxed",
        pins=BASE_PINS.model_copy(
            update={"task_prompt_revision": "gsm8k=limite-evals:gsm8k-math500-boxed-v1"}
        ),
    )

    assert previous.comparable_with(boxed) == [
        "different task prompt revision pin: "
        f"gsm8k={previous_revision} vs gsm8k=limite-evals:gsm8k-math500-boxed-v1"
    ]


def test_a_different_format_policy_blocks_comparison() -> None:
    reasons = _manifest().comparable_with(_manifest(run_id="r004", format_policy="lenient"))
    assert reasons == ["different format policy: strict vs lenient"]


def test_eager_serving_mode_blocks_comparison() -> None:
    """Eager serving bypasses vLLM's compile/capture path and is a new measurement."""
    eager = _manifest(run_id="r004", enforce_eager=True)
    assert _manifest().comparable_with(eager) == ["different vLLM eager mode: False vs True"]


def test_tensor_parallel_size_blocks_comparison() -> None:
    sharded = _manifest(run_id="r-tp2", tensor_parallel_size=2)
    assert _manifest().comparable_with(sharded) == [
        "different tensor parallel size: 1 vs 2"
    ]


def test_a_changed_relaxation_base_blocks_comparison() -> None:
    """D-108's base is per taskset, so `lenient` alone does not name a quantity.

    `lenient` relaxed against a benchmark's published grader and `lenient` relaxed
    against prime-rl's verifier are two measurements under one column heading.
    Two runs whose bases differ have `lenient` columns that cannot be subtracted,
    and nothing else in the manifest would say so -- the pins, the fingerprint and
    the suite can all match exactly.
    """
    published = _manifest(relaxation_bases="math500=exact")
    verifier = _manifest(run_id="r005", relaxation_bases="math500=reference")
    assert published.comparable_with(verifier) == [
        "different relaxation base: math500=exact vs math500=reference"
    ]
    assert published.comparable_with(_manifest(run_id="r006", relaxation_bases="math500=exact")) == []


def test_a_run_that_recorded_no_base_is_not_paired_with_one_that_did() -> None:
    """A result file written before protocols existed has no base to state. It is
    not thereby comparable with every base: it is comparable with none of them."""
    assert _manifest(relaxation_bases="math500=exact").comparable_with(_manifest(run_id="r007")) == [
        "different relaxation base: math500=exact vs unrecorded"
    ]
    # And two such runs still compare with each other, which is what keeps the
    # existing math-standard numbers quotable against each other.
    assert _manifest().comparable_with(_manifest(run_id="r008")) == []


def test_a_changed_k_set_blocks_comparison() -> None:
    """Which k a headline was asked at is a measurement, not a presentation.

    Two runs that declared different sets report different columns of the same
    rollouts, and the absence of a `pass@32` cannot say which of them happened:
    it reads the same on a run that never asked for one and on a run that asked
    and drew too few rollouts to answer. The manifest is where that is settled.
    """
    asked = _manifest(pass_at_k="math500=1+4+32")
    fewer = _manifest(run_id="r009", pass_at_k="math500=1+4")
    assert asked.comparable_with(fewer) == ["different k set: math500=1+4+32 vs math500=1+4"]
    assert asked.comparable_with(_manifest(run_id="r010", pass_at_k="math500=1+4+32")) == []


def test_an_unrecorded_k_set_is_no_declaration_rather_than_a_different_one() -> None:
    """The opposite answer to the one the relaxation base gets, and on purpose.

    A base decides what the shared `lenient` column means, so a run that
    recorded none has a number nobody can read. A k set adds columns and leaves
    every shared one exactly as it was, so a run that reported no `pass@k` still
    has an `exact` and a `reference` that may be quoted beside one that did.
    Refusing here would put every run written before the columns existed out of
    reach of every run after them, which is the pre-to-post comparison the
    instrument exists to make, and it would do so naming a pair of columns that
    does not exist on one side.
    """
    assert _manifest(pass_at_k="math500=1+4+32").comparable_with(_manifest(run_id="r011")) == []
    assert _manifest().comparable_with(_manifest(run_id="r012", pass_at_k="math500=1+4")) == []


def test_a_changed_context_length_is_a_fatal_fingerprint_mismatch() -> None:
    observed = BASE_FINGERPRINT.model_copy(update={"max_model_len": 4096})
    mismatches = observed.mismatches(BASE_FINGERPRINT)
    assert mismatches == {"max_model_len": (8192, 4096)}
    assert Fingerprint.fatal(mismatches) == mismatches


def test_a_changed_engine_version_is_recorded_but_not_fatal() -> None:
    observed = BASE_FINGERPRINT.model_copy(update={"vllm_version": "0.27.0"})
    mismatches = observed.mismatches(BASE_FINGERPRINT)
    assert "vllm_version" in mismatches
    assert Fingerprint.fatal(mismatches) == {}


def test_an_unobserved_field_is_not_a_mismatch() -> None:
    """An engine that does not report a field must not read as a disagreement."""
    observed = BASE_FINGERPRINT.model_copy(update={"vllm_version": None})
    assert observed.mismatches(BASE_FINGERPRINT) == {}


LEGACY_TASK = (
    '{"task": "aime24", "problems": 30, "group_size": 32, '
    '"strict": {"value": 0.1, "low": 0.0, "high": 0.2, "confidence": 0.95}}'
)


def test_a_task_summary_written_before_metric_sets_still_loads() -> None:
    """The five rungs were declared fields; they are now whatever the taskset declared.

    Keeping them at the top level of the object rather than under a nested map
    is what lets an existing result file load and an existing reader keep
    indexing `task["strict"]` off the JSON. The metric set defaults to the
    ladder, which is what such a file was written by.
    """
    task = TaskSummary.model_validate_json(LEGACY_TASK)

    assert task.metric_set == LADDER
    assert task.metrics["strict"].value == 0.1
    assert task.strict is task.metrics["strict"]
    assert json.loads(task.model_dump_json())["strict"]["value"] == 0.1


def test_a_metric_that_is_not_an_interval_is_still_an_error() -> None:
    """Otherwise "allow extras" would mean "accept anything".

    A summary that silently swallows a mistyped key is worse than one that
    refuses to load: the number goes missing from the report without saying so.
    """
    with pytest.raises(ValidationError):
        TaskSummary(task="aime24", problems=30, group_size=32, strict=0.1)


def test_the_ladder_is_a_declaration_like_any_other() -> None:
    """Decision 3's order and decision 10's subset, now stated rather than assumed.

    `format_ok` and `truncated` stay out of the correctness list: they describe
    what the completion looked like, and forcing them on a truncated rollout
    would erase the fact being reported.
    """
    assert LADDER.headline == "strict"
    assert LADDER.metrics == ("strict", "lenient", "permissive", "format_ok", "truncated")
    assert LADDER.correctness == ("strict", "lenient", "permissive")
    assert LADDER.row_label == "Rung"


# --- the instrument era: derived, stamped, and refused across ----------------


def test_the_era_is_derived_from_what_the_run_recorded() -> None:
    """Two witnesses, either sufficient, and neither settable from outside.

    The fingerprint's tokenizer and terminator are the pair the E42 correction
    added, and the decode flag is the third field it added. A run that recorded
    any of them was written by the corrected instrument; a run that recorded
    none of them was not.
    """
    served_through_a_pin = _manifest(
        sampling=Sampling(temperature=0.0, max_tokens=4096),
        fingerprint=BASE_FINGERPRINT.model_copy(
            update={"tokenizer": "paradigma/limite-tokenizer@abc123", "eos_token_ids": [151666]}
        ),
    )
    recorded_its_decode = _manifest()
    recorded_neither = _manifest(sampling=Sampling(temperature=0.0, max_tokens=4096))

    assert served_through_a_pin.instrument_era == "corrected"
    assert recorded_its_decode.instrument_era == "corrected"
    assert recorded_neither.instrument_era == "e42"


def test_a_baseline_served_its_own_tokenizer_is_not_filed_as_e42() -> None:
    """Why the fingerprint alone cannot decide it.

    A reference baseline keeps the tokenizer it was trained against and pins no
    terminator, so it records no vocabulary at all -- and it is every bit as
    corrected-era as a Limite run beside it. Reading only the fingerprint would
    quarantine those runs, refusing to rescore completions whose delimiters were
    never at risk in the first place.
    """
    baseline = _manifest(checkpoint="Qwen/Qwen3-8B")

    assert baseline.fingerprint.records_vocabulary is False
    assert baseline.instrument_era == "corrected"


def test_the_era_is_written_into_the_summary_and_cannot_be_set_from_outside() -> None:
    """A computed field, so a file somebody edited cannot lie about its instrument."""
    manifest = _manifest()
    assert json.loads(manifest.model_dump_json())["instrument_era"] == "corrected"

    forged = json.loads(manifest.model_dump_json())
    forged["sampling"]["skip_special_tokens"] = None
    forged["instrument_era"] = "corrected"
    assert RunManifest.model_validate(forged).instrument_era == "e42"


def test_two_eras_are_refused_rather_than_reported() -> None:
    """The one difference that is not advice.

    Every other reason `comparable_with` returns is a difference in how the same
    text was scored. This is a difference in what text there was, so it raises --
    and it appears in the reason list too, for a caller reading those instead.
    """
    corrected = _manifest()
    e42 = _manifest(run_id="r002", sampling=Sampling(temperature=0.0, max_tokens=4096))

    with pytest.raises(InstrumentEraMismatch, match="deleted between the engine and the grader"):
        corrected.assert_comparable_with(e42)
    with pytest.raises(InstrumentEraMismatch):
        e42.assert_comparable_with(corrected)

    assert "different instrument era: corrected vs e42" in corrected.comparable_with(e42)[0]


def test_one_era_compares_normally_on_both_sides() -> None:
    """The refusal is about the boundary, not about the marker existing."""
    corrected = _manifest()
    also_corrected = _manifest(run_id="r002", profile="chat")
    e42 = _manifest(sampling=Sampling(temperature=0.0, max_tokens=4096))
    also_e42 = _manifest(run_id="r002", sampling=Sampling(temperature=0.0, max_tokens=4096))

    corrected.assert_comparable_with(also_corrected)
    e42.assert_comparable_with(also_e42)
    assert corrected.comparable_with(also_corrected) == []
    assert e42.comparable_with(also_e42) == []
