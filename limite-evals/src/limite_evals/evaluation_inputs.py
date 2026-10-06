"""Freeze the ordered inputs and scoring bindings actually used by a run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from limite_evals import tasksets
from limite_evals.checkpoint import ProblemIdentity, TaskContract, text_sha256
from limite_evals.checkpoint_schema import CheckpointMetadata, first_difference
from limite_evals.profiles import Profile
from limite_evals.suites import TasksetSpec
from limite_evals.templates import Rendering
from limite_evals_core.protocols import TasksetProtocols, Unavailable
from limite_evals_core.schema import PROTOCOL_METRICS, RunManifest, RunSummary

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)


@dataclass(frozen=True)
class PreparedTask:
    """One loaded/rendered taskset, shared by contract creation and generation.

    The lists retain their original dataset order throughout resume. `identity`
    holds hashes and scorer declarations; prompt and gold text stay in memory
    until their completed sample is written into the journal.
    """

    problems: list[tuple[str, str]]
    bindings: TasksetProtocols | None
    budget: int
    task_contract: TaskContract
    identity: dict


def prepare(
    spec: TasksetSpec, profile: Profile, rendering: Rendering, *, budget: int, limit: int | None
) -> PreparedTask:
    """Load and render once, before generation, preserving source order and pins."""
    rows = limit_rows(spec, tasksets.load_raw_rows(spec), limit)
    problems = [tasksets.render_row(spec, row, profile) for row in rows]
    bindings = tasksets.taskset_protocols(spec)
    declared = TaskContract(
        name=spec.name,
        group_size=spec.group_size,
        template_sha256=rendering.sha256(),
        problems=[
            ProblemIdentity(gold_sha256=text_sha256(gold))
            for _, gold in problems
        ],
    )
    identity = {
        "ordered_inputs_sha256": _digest(problems),
        # Raw rows also cover source IDs and benchmark-specific fields that are
        # intentionally absent from the rendered prompt and SampleResult schema.
        "ordered_source_rows_sha256": _digest([asdict(row) if is_dataclass(row) else row for row in rows]),
        "max_tokens": budget,
        "answer_format": spec.answer_format,
        "metric_set": (spec.metrics or PROTOCOL_METRICS).model_dump(mode="json"),
        "protocols": {
            "exact": (
                {"unavailable": bindings.exact.reason}
                if isinstance(bindings.exact, Unavailable)
                else {"anchor": bindings.exact.name}
            ),
            "reference": {"anchor": bindings.reference.name},
        },
    }
    return PreparedTask(
        problems=problems,
        bindings=bindings,
        budget=budget,
        task_contract=declared,
        identity=identity,
    )


def _digest(value: object) -> str:
    return text_sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def local_tokenizer_identity(model: str) -> str | None:
    """Hash the effective local tokenizer without changing historical digests.

    Hub sources already carry immutable revisions and require no extra fetch.
    The exporter's existing hash covers standard tokenizer assets. Native Llama
    tokenizers can also load tokenizer.model, and Transformers can select a
    versioned tokenizer JSON named in tokenizer_config.json.
    """
    source = Path(model)
    if not source.is_dir():
        return None
    names = {"tokenizer.model"}
    config_path = source / "tokenizer_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if "fast_tokenizer_files" in config:
            from transformers.tokenization_utils_base import get_fast_tokenizer_file

            names.add(get_fast_tokenizer_file(config["fast_tokenizer_files"]))
    additional = {}
    for name in sorted(names - set(TOKENIZER_FILES)):
        path = source / name
        if path.is_file():
            with path.open("rb") as handle:
                additional[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    standard = {}
    for name in TOKENIZER_FILES:
        path = source / name
        if path.is_file():
            with path.open("rb") as handle:
                standard[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return _digest({"standard": standard, "additional": additional})


def manifest_identity(manifest: RunManifest) -> dict:
    """Compare declared measurement fields before serving and after aggregation.

    Attempt identity/time, aggregation stamps, and runtime observations have
    separate authoritative records. Runtime evidence is checked against saved
    fingerprints; scorer versions and metric declarations live in the task
    contract. Everything else must match the declaration exactly.
    """
    return manifest.model_dump(
        mode="json",
        exclude={
            "run_id": True,
            "created_at": True,
            "scorer_version": True,
            "pass_at_k": True,
            "fingerprint": {"vllm_version", "engine_venv", "tokenizer", "eos_token_ids"},
        },
    )


def validate_summary_manifest(summary: RunSummary, metadata: CheckpointMetadata) -> None:
    """Require a final summary to describe this checkpoint's frozen measurement.

    Coverage and journal validity belong to `SampleJournal.validate_complete`.
    This boundary also serves scheduler restarts that skip a completed run,
    without loading datasets or starting an engine again.
    """
    expected = metadata.contract.get("manifest")
    if difference := first_difference(expected, manifest_identity(summary.manifest), "manifest"):
        raise ValueError(f"terminal summary contract mismatch: {difference}")
    if summary.manifest.run_id != metadata.attempt_id:
        raise ValueError("terminal summary attempt identity mismatch")
    if [task.task for task in summary.tasks] != [task.name for task in metadata.tasks]:
        raise ValueError("terminal summary task order/coverage mismatch")
    for task, declared in zip(summary.tasks, metadata.tasks, strict=True):
        expected_metrics = metadata.contract["tasksets"][declared.name]["metric_set"]
        if task.metric_set.model_dump(mode="json") != expected_metrics:
            raise ValueError(f"terminal summary metric-set mismatch: {task.task}")
        if summary.manifest.scorer_version != declared.scorer_version:
            raise ValueError(f"terminal summary scorer-version mismatch: {task.task}")
        if task.problems != len(declared.problems) or task.group_size != declared.group_size:
            raise ValueError(f"terminal summary sample coverage mismatch: {task.task}")
    expected_k = ",".join(
        f"{task.task}={'+'.join(map(str, task.metric_set.pass_at_k))}"
        for task in sorted(summary.tasks, key=lambda task: task.task)
        if task.metric_set.pass_at_k
    )
    if summary.manifest.pass_at_k != expected_k:
        raise ValueError("terminal summary pass-at-k declaration mismatch")
    observed = summary.manifest.fingerprint.model_dump(exclude={"template_sha256", "engine_venv"})
    for fingerprint in metadata.fingerprints.values():
        expected = fingerprint.model_dump(exclude={"template_sha256", "engine_venv"})
        if difference := first_difference(expected, observed, "fingerprint"):
            raise ValueError(f"terminal summary runtime mismatch: {difference}")


def limit_rows(spec: TasksetSpec, rows: list, limit: int | None) -> list:
    """Apply the dataset-order limit."""
    del spec
    return rows if limit is None else rows[:limit]
