"""Versioned recovery metadata and durable publication of small JSON documents."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from limite_evals_core.schema import SCORER_VERSION, Fingerprint

CHECKPOINT_FILE = "checkpoint.json"
LOCK_FILE = ".checkpoint.lock"
SAMPLES_FILE = Path("metrics/samples.jsonl")
SampleKey = tuple[str, int, int]


class CheckpointError(ValueError):
    """An interrupted evaluation cannot be safely imported or extended."""


class ProblemIdentity(BaseModel):
    """Expected row identity without copying answer text into metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    gold_sha256: str
    source_group_id: str | None = None
    source_variant_id: str | None = None


class TaskContract(BaseModel):
    """Declared task order, rollout count, and identities checked on every sample."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    group_size: int = Field(gt=0)
    template_sha256: str
    problems: list[ProblemIdentity]
    scorer_version: str = SCORER_VERSION


class CheckpointMetadata(BaseModel):
    """One immutable evaluation contract and this attempt's recovery state.

    ``import_complete=False`` means no new generation is allowed: the intact
    ``source_attempt`` remains authoritative if copying is interrupted.
    Fingerprints are observed engine evidence, published before its samples.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1]
    logical_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    source_attempt: str | None
    import_complete: bool
    contract: dict[str, Any]
    tasks: list[TaskContract]
    fingerprints: dict[str, Fingerprint]
    slurm_job_id: str | None
    slurm_restart_count: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_lineage_and_tasks(self) -> CheckpointMetadata:
        if not self.import_complete and self.source_attempt is None:
            raise ValueError("incomplete import requires source_attempt")
        if self.source_attempt is not None and not Path(self.source_attempt).is_absolute():
            raise ValueError("source_attempt must be absolute")
        names = [task.name for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("duplicate task names in checkpoint contract")
        templates = {task.template_sha256 for task in self.tasks}
        for key, fingerprint in self.fingerprints.items():
            if key not in templates or fingerprint.template_sha256 != key:
                raise ValueError(f"unexpected fingerprint template {key}")
        return self


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_text(value: object, *, sort_keys: bool = True) -> str:
    """Reject non-finite JSON numbers; optionally preserve declared report order."""
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, allow_nan=False)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object, *, sort_keys: bool = True) -> None:
    """Publish JSON only after file sync, then sync the containing directory.

    The caller owns the destination's writer lock or its fresh run directory.
    Readers see either the preceding complete document or the new one. Reports
    use ``sort_keys=False`` to preserve their declared metric order.
    """
    contents = json_text(value, sort_keys=sort_keys) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def first_difference(expected: object, observed: object, path: str) -> str | None:
    """Identify a changed contract field without including its possibly large value."""
    if type(expected) is not type(observed):
        return path
    if isinstance(expected, dict):
        for key in sorted(expected.keys() | observed.keys()):
            child = f"{path}.{key}"
            if key not in expected or key not in observed:
                return child
            if difference := first_difference(expected[key], observed[key], child):
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return path
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            if difference := first_difference(left, right, f"{path}[{index}]"):
                return difference
        return None
    return path if expected != observed else None
