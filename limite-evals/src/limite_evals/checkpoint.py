"""Append-only scored samples with exclusive writing and immutable-source recovery."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from pydantic import BaseModel

from limite_evals.checkpoint_schema import (
    CHECKPOINT_FILE,
    LOCK_FILE,
    SAMPLES_FILE,
    CheckpointError,
    CheckpointMetadata,
    ProblemIdentity,
    SampleKey,
    TaskContract,
    atomic_json,
    first_difference,
    json_text,
    sync_directory,
    text_sha256,
)
from limite_evals_core.schema import Fingerprint, ProtocolSample, SampleResult

__all__ = [
    "CHECKPOINT_FILE",
    "CheckpointError",
    "CheckpointMetadata",
    "CheckpointSnapshot",
    "ProblemIdentity",
    "SampleJournal",
    "TaskContract",
    "atomic_json",
    "read_checkpoint",
    "read_source",
    "text_sha256",
]


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _json(contents: str | bytes) -> object:
    return json.loads(contents, parse_constant=_reject_nonfinite)


def _complete_fields(value: object, model: type[BaseModel]) -> None:
    # Defaults support old offline reports; a new checkpoint must actually carry
    # every saved field rather than invent scores or identities during recovery.
    if not isinstance(value, dict) or value.keys() != model.model_fields.keys():
        raise ValueError(f"record must contain all and only {model.__name__} fields")


def _read_metadata(run_dir: Path) -> CheckpointMetadata:
    try:
        value = _json((run_dir / CHECKPOINT_FILE).read_bytes())
        _complete_fields(value, CheckpointMetadata)
        for task in value["tasks"]:
            _complete_fields(task, TaskContract)
            for problem in task["problems"]:
                _complete_fields(problem, ProblemIdentity)
        for fingerprint in value["fingerprints"].values():
            _complete_fields(fingerprint, Fingerprint)
        metadata = CheckpointMetadata.model_validate(value)
        if metadata.attempt_id != run_dir.name:
            raise ValueError("attempt_id does not match source directory")
        return metadata
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise CheckpointError(f"invalid checkpoint metadata in {run_dir}: {error}") from error


@contextmanager
def _locked_metadata(run_dir: Path) -> Iterator[CheckpointMetadata]:
    try:
        lock = (run_dir / LOCK_FILE).open("rb")
    except FileNotFoundError as error:
        raise CheckpointError(f"source has no initialized checkpoint: {run_dir}") from error
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CheckpointError(f"source checkpoint has an active writer: {run_dir}") from error
        yield _read_metadata(run_dir)


def read_checkpoint(run_dir: Path) -> CheckpointMetadata:
    """Read validated metadata under its shared lock; use read_source for samples."""
    with _locked_metadata(Path(run_dir)) as metadata:
        return metadata


def _expected_keys(tasks: list[TaskContract]) -> list[SampleKey]:
    return [
        (task.name, problem, rollout)
        for task in tasks
        for problem in range(len(task.problems))
        for rollout in range(task.group_size)
    ]


def _validate_sample(
    sample: SampleResult, tasks: dict[str, TaskContract], fingerprints: dict[str, Fingerprint]
) -> SampleKey:
    key = sample.task, sample.problem_index, sample.rollout_index
    task = tasks.get(sample.task)
    if (
        task is None
        or not 0 <= sample.problem_index < len(task.problems)
        or not 0 <= sample.rollout_index < task.group_size
    ):
        raise CheckpointError(f"unexpected sample key {key}")
    problem = task.problems[sample.problem_index]
    expected = {
        "gold_sha256": problem.gold_sha256,
        "source_group_id": problem.source_group_id,
        "source_variant_id": problem.source_variant_id,
        "template_sha256": task.template_sha256,
        "scorer_version": task.scorer_version,
    }
    observed = {name: getattr(sample, name) for name in expected if name != "gold_sha256"}
    observed["gold_sha256"] = text_sha256(sample.gold)
    if difference := first_difference(expected, observed, "sample"):
        raise CheckpointError(f"incompatible {difference} for key {key}")
    if task.template_sha256 not in fingerprints:
        raise CheckpointError(f"sample {key} has no observed engine fingerprint")
    return key


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Validated source records in declared order, available while its lock is held."""

    metadata: CheckpointMetadata
    samples: list[SampleResult]

    def validate_complete(self) -> list[SampleResult]:
        """Return the bank only after import and exact declared coverage are complete."""
        expected = _expected_keys(self.metadata.tasks)
        actual = [(row.task, row.problem_index, row.rollout_index) for row in self.samples]
        if actual != expected:
            missing = set(expected) - set(actual)
            raise CheckpointError(
                f"checkpoint coverage is incomplete or duplicated: {len(missing)} missing keys"
            )
        if not self.metadata.import_complete:
            raise CheckpointError("checkpoint import is incomplete")
        return self.samples


@contextmanager
def read_source(run_dir: Path) -> Iterator[CheckpointSnapshot]:
    """Validate and hold an existing source's nonblocking shared lock.

    No file in the source is created or changed. Only a final record without a
    newline is uncommitted; every complete record must pass schema and identity
    checks. Import lineage is deliberately followed by SampleJournal.create.
    """
    run_dir = Path(run_dir)
    with _locked_metadata(run_dir) as metadata:
        rows: dict[SampleKey, SampleResult] = {}
        tasks = {task.name: task for task in metadata.tasks}
        try:
            with (run_dir / SAMPLES_FILE).open("rb") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.endswith(b"\n"):
                        break
                    try:
                        record = _json(line)
                        _complete_fields(record, SampleResult)
                        if isinstance(record["protocols"], dict):
                            for verdict in record["protocols"].values():
                                _complete_fields(verdict, ProtocolSample)
                        sample = SampleResult.model_validate(record, strict=True)
                        key = _validate_sample(sample, tasks, metadata.fingerprints)
                        if key in rows:
                            raise CheckpointError(f"duplicate sample key {key}")
                        rows[key] = sample
                    except (ValueError, TypeError) as error:
                        raise CheckpointError(
                            f"invalid sample record {line_number} in {run_dir}: {error}"
                        ) from error
        except OSError as error:
            raise CheckpointError(f"cannot read checkpoint samples in {run_dir}: {error}") from error
        yield CheckpointSnapshot(
            metadata, [rows[key] for key in _expected_keys(metadata.tasks) if key in rows]
        )


class SampleJournal:
    """One serialized sample writer, owned for the lifetime of the create context."""

    def __init__(self, run_dir: Path, metadata: CheckpointMetadata, handle: TextIO):
        self.run_dir = run_dir
        self.metadata = metadata
        self._handle = handle
        self._rows: dict[SampleKey, SampleResult] = {}
        self._tasks = {task.name: task for task in metadata.tasks}
        self._order = _expected_keys(metadata.tasks)
        self._mutex = threading.RLock()

    @classmethod
    @contextmanager
    def create(
        cls,
        run_dir: Path,
        *,
        contract: dict,
        tasks: list[TaskContract],
        logical_id: str | None = None,
        source: Path | None = None,
    ) -> Iterator[SampleJournal]:
        """Initialize a reserved, fresh run, optionally importing an intact source.

        The caller reserves the directory and its metrics child. Existing
        destinations are never reopened. Interrupted imports retain their source
        chain; locks protect every consulted source through copying. Fresh runs
        and resumed runs share the same live append path after initialization.
        """
        run_dir = Path(run_dir).resolve()
        source = Path(source).resolve() if source is not None else None
        with ExitStack() as stack:
            lock = stack.enter_context((run_dir / LOCK_FILE).open("xb"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle = stack.enter_context((run_dir / SAMPLES_FILE).open("x", encoding="utf-8"))
            sync_directory(run_dir / "metrics")
            if (run_dir / CHECKPOINT_FILE).exists():
                raise FileExistsError(f"checkpoint destination already initialized: {run_dir}")
            inherited = None
            requested_logical = logical_id or os.environ.get("LIMITE_EVAL_LOGICAL_ID")
            next_source = source
            seen = {run_dir}
            while next_source is not None:
                if next_source in seen:
                    raise CheckpointError("cycle in checkpoint source lineage")
                seen.add(next_source)
                inherited = stack.enter_context(read_source(next_source))
                prior = inherited.metadata
                expected = {"contract": contract, "tasks": [task.model_dump(mode="json") for task in tasks]}
                observed = {
                    "contract": prior.contract,
                    "tasks": [task.model_dump(mode="json") for task in prior.tasks],
                }
                if difference := first_difference(expected, observed, "checkpoint"):
                    raise CheckpointError(f"incompatible {difference}")
                if requested_logical is not None and prior.logical_id != requested_logical:
                    raise CheckpointError("incompatible checkpoint.logical_id")
                requested_logical = prior.logical_id
                next_source = None if prior.import_complete else Path(prior.source_attempt).resolve()
            metadata = CheckpointMetadata(
                version=1,
                logical_id=requested_logical or run_dir.name,
                attempt_id=run_dir.name,
                source_attempt=str(source) if source is not None else None,
                import_complete=source is None,
                contract=_json(json_text(contract)),
                tasks=tasks,
                fingerprints={} if inherited is None else inherited.metadata.fingerprints,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                slurm_restart_count=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            )
            atomic_json(run_dir / CHECKPOINT_FILE, metadata.model_dump(mode="json"))
            journal = cls(run_dir, metadata, handle)
            if inherited is not None:
                # Until import_complete is durable, recovery uses the intact
                # source. One sync for the copied bank avoids per-row fsync cost.
                for sample in inherited.samples:
                    journal._append(sample, synchronize=False)
                handle.flush()
                os.fsync(handle.fileno())
                journal._publish(metadata.model_copy(update={"import_complete": True}))
            yield journal

    @property
    def logical_id(self) -> str:
        return self.metadata.logical_id

    @property
    def completed_keys(self) -> set[SampleKey]:
        with self._mutex:
            return set(self._rows)

    @property
    def samples(self) -> list[SampleResult]:
        with self._mutex:
            return [self._rows[key] for key in self._order if key in self._rows]

    @property
    def fingerprints(self) -> dict[str, Fingerprint]:
        return dict(self.metadata.fingerprints)

    def _publish(self, metadata: CheckpointMetadata) -> None:
        atomic_json(self.run_dir / CHECKPOINT_FILE, metadata.model_dump(mode="json"))
        self.metadata = metadata

    def record_fingerprint(self, template_sha256: str, fingerprint: Fingerprint) -> None:
        """Publish observed runtime identity before accepting samples from it.

        Only the environment's filesystem location may differ on restart. Every
        recorded runtime field, including absence, remains part of the comparison.
        """
        with self._mutex:
            if template_sha256 not in {task.template_sha256 for task in self.metadata.tasks}:
                raise CheckpointError(f"unexpected fingerprint template {template_sha256}")
            if fingerprint.template_sha256 != template_sha256:
                raise CheckpointError("incompatible fingerprint.template_sha256")
            prior = self.metadata.fingerprints.get(template_sha256)
            if prior is not None:
                expected = prior.model_dump(exclude={"engine_venv"})
                observed = fingerprint.model_dump(exclude={"engine_venv"})
                if difference := first_difference(expected, observed, "fingerprint"):
                    raise CheckpointError(f"incompatible {difference}")
            fingerprints = {**self.metadata.fingerprints, template_sha256: fingerprint.model_copy(deep=True)}
            self._publish(self.metadata.model_copy(update={"fingerprints": fingerprints}))

    def append(self, sample: SampleResult) -> None:
        """Return only after the complete scored record has been flushed and synced."""
        with self._mutex:
            if not self.metadata.import_complete:
                raise CheckpointError("cannot append new samples before checkpoint import completes")
            self._append(sample, synchronize=True)

    def _append(self, sample: SampleResult, *, synchronize: bool) -> None:
        try:
            contents = json_text(sample.model_dump(mode="json"))
            saved = SampleResult.model_validate(_json(contents), strict=True)
        except ValueError as error:
            raise CheckpointError(f"invalid sample record: {error}") from error
        key = _validate_sample(saved, self._tasks, self.metadata.fingerprints)
        if key in self._rows:
            raise CheckpointError(f"duplicate sample key {key}")
        self._handle.write(contents + "\n")
        if synchronize:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        self._rows[key] = saved

    def validate_complete(self) -> list[SampleResult]:
        """Return the durable bank in declared order, refusing missing sample keys."""
        return CheckpointSnapshot(self.metadata, self.samples).validate_complete()
