"""Recovery through real files, locks, and abrupt process termination."""

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from limite_evals.checkpoint import (
    CHECKPOINT_FILE,
    CheckpointError,
    ProblemIdentity,
    SampleJournal,
    TaskContract,
    atomic_json,
    read_source,
    text_sha256,
)
from limite_evals_core.schema import SCORER_VERSION, Fingerprint, ProtocolSample, SampleResult


def task(name="aime", group_size=2):
    return TaskContract(
        name=name,
        group_size=group_size,
        template_sha256="template",
        problems=[ProblemIdentity(gold_sha256=text_sha256("42"))],
    )


def sample(rollout=0, **updates):
    values = dict(
        task="aime",
        problem_index=0,
        rollout_index=rollout,
        gold="42",
        completion="answer",
        template_sha256="template",
        scorer_version=SCORER_VERSION,
    )
    values.update(updates)
    return SampleResult(**values)


def fingerprint(**updates):
    values = dict(
        architecture="test",
        artifact_sha256="weights",
        template_sha256="template",
        max_model_len=2048,
        dtype="bfloat16",
        vllm_version="test",
    )
    values.update(updates)
    return Fingerprint(**values)


def reserve(root, name):
    path = root / name
    (path / "metrics").mkdir(parents=True)
    return path


def create(path, **kwargs):
    return SampleJournal.create(path, contract={"model": "weights", "seed": 17}, tasks=[task()], **kwargs)


def populate(path, samples):
    with create(path) as journal:
        journal.record_fingerprint("template", fingerprint())
        for row in samples:
            journal.append(row)


def test_resume_preserves_all_verdicts_and_original_keys(tmp_path):
    source = reserve(tmp_path, "first")
    saved = sample(strict=False, truncated=True, finish_reason="length")
    populate(source, [saved])
    original = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with create(reserve(tmp_path, "second"), source=source) as journal:
        assert journal.logical_id == "first"
        assert journal.samples == [saved]
        assert journal.completed_keys == {("aime", 0, 0)}
        assert journal.fingerprints == {"template": fingerprint()}
        journal.append(sample(1, strict=True))
        assert journal.validate_complete() == [saved, sample(1, strict=True)]
    assert all(path.read_bytes() == contents for path, contents in original.items())


def test_multiple_restarts_make_cumulative_progress(tmp_path):
    first = reserve(tmp_path, "first")
    populate(first, [sample(1)])
    second = reserve(tmp_path, "second")
    with create(second, source=first) as journal:
        journal.append(sample(0))
    with create(reserve(tmp_path, "third"), source=second) as journal:
        assert journal.validate_complete() == [sample(0), sample(1)]
        assert journal.metadata.source_attempt == str(second.resolve())


def test_contract_written_before_samples_and_engine_required(tmp_path):
    path = reserve(tmp_path, "first")
    with create(path) as journal:
        metadata = json.loads((path / CHECKPOINT_FILE).read_text())
        assert metadata["contract"] == {"model": "weights", "seed": 17}
        assert metadata["import_complete"] is True
        with pytest.raises(CheckpointError, match="fingerprint"):
            journal.append(sample())
        assert (path / "metrics/samples.jsonl").read_bytes() == b""
        journal.record_fingerprint("template", fingerprint())
        journal.append(sample())
        with pytest.raises(CheckpointError, match="missing"):
            journal.validate_complete()


@pytest.mark.parametrize("field,value", [("seed", 18), ("model", "changed")])
def test_incompatible_contract_refused_before_import(tmp_path, field, value):
    source = reserve(tmp_path, "first")
    populate(source, [sample()])
    contract = {"model": "weights", "seed": 17, field: value}
    with pytest.raises(CheckpointError, match=field):
        with SampleJournal.create(
            reserve(tmp_path, "second"), contract=contract, tasks=[task()], source=source
        ):
            pytest.fail("incompatible contract entered generation")


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"problem_index": -1}, "unexpected"),
        ({"rollout_index": 2}, "unexpected"),
        ({"gold": "wrong"}, "gold"),
        ({"source_group_id": "wrong"}, "source_group_id"),
        ({"template_sha256": "other"}, "template"),
        ({"scorer_version": "old"}, "scorer"),
    ],
)
def test_invalid_sample_identity_refused(tmp_path, updates, match):
    path = reserve(tmp_path, "first")
    with create(path) as journal:
        journal.record_fingerprint("template", fingerprint())
        with pytest.raises(CheckpointError, match=match):
            journal.append(sample(**updates))
        assert journal.samples == []


@pytest.mark.parametrize("suffix,match", [(b"{bad}\n", "record"), (b"\n", "record")])
def test_malformed_complete_record_refuses_instead_of_regenerating(tmp_path, suffix, match):
    path = reserve(tmp_path, "first")
    populate(path, [sample()])
    with (path / "metrics/samples.jsonl").open("ab") as handle:
        handle.write(suffix)
    with pytest.raises(CheckpointError, match=match):
        with read_source(path):
            pytest.fail("damaged source accepted")


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_complete_record_refused(tmp_path, conflicting):
    path = reserve(tmp_path, "first")
    populate(path, [sample()])
    duplicate = sample(completion="different") if conflicting else sample()
    with (path / "metrics/samples.jsonl").open("a") as handle:
        handle.write(duplicate.model_dump_json() + "\n")
    with pytest.raises(CheckpointError, match="duplicate"):
        with read_source(path):
            pytest.fail("duplicate source accepted")


def test_read_source_does_not_create_missing_lock_or_accept_active_writer(tmp_path):
    legacy = reserve(tmp_path, "legacy")
    with pytest.raises(CheckpointError, match="checkpoint"):
        with read_source(legacy):
            pytest.fail("historical source accepted")
    assert list(legacy.iterdir()) == [legacy / "metrics"]
    path = reserve(tmp_path, "first")
    with create(path):
        with pytest.raises(CheckpointError, match="active writer"):
            with read_source(path):
                pytest.fail("changing bank imported")


def test_existing_destination_cannot_be_reopened(tmp_path):
    path = reserve(tmp_path, "first")
    populate(path, [sample()])
    before = (path / "metrics/samples.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        with create(path):
            pytest.fail("destination reused")
    assert (path / "metrics/samples.jsonl").read_bytes() == before


def test_recorded_engine_contract_must_match_except_environment_location(tmp_path):
    path = reserve(tmp_path, "first")
    with create(path) as journal:
        journal.record_fingerprint("template", fingerprint(engine_venv="/old/env"))
        journal.append(sample())
        journal.record_fingerprint("template", fingerprint(engine_venv="/new/env"))
        with pytest.raises(CheckpointError, match="dtype"):
            journal.record_fingerprint("template", fingerprint(dtype="float16"))


def _writer_killed_after_commit(path, connection):
    with create(path) as journal:
        journal.record_fingerprint("template", fingerprint())
        journal.append(sample())
        with (path / "metrics/samples.jsonl").open("ab") as handle:
            handle.write(b'{"task":"unfinished')
            handle.flush()
        connection.send("committed")
        connection.recv()


def test_sigkill_keeps_committed_sample_and_discards_only_uncommitted_tail(tmp_path):
    source = reserve(tmp_path, "first")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_writer_killed_after_commit, args=(source, child))
    process.start()
    try:
        assert parent.poll(20), "writer did not reach its durable sample boundary"
        assert parent.recv() == "committed"
    finally:
        process.kill()
        process.join(timeout=10)
        parent.close()
        child.close()
    assert process.exitcode == -9
    before = (source / "metrics/samples.jsonl").read_bytes()
    with create(reserve(tmp_path, "second"), source=source) as journal:
        assert journal.samples == [sample()]
        journal.append(sample(1))
        assert journal.validate_complete() == [sample(), sample(1)]
    assert (source / "metrics/samples.jsonl").read_bytes() == before


def test_unsupported_metadata_version_and_missing_fields_fail(tmp_path):
    source = reserve(tmp_path, "first")
    populate(source, [sample()])
    metadata = json.loads((source / CHECKPOINT_FILE).read_text())
    metadata["version"] = 999
    atomic_json(source / CHECKPOINT_FILE, metadata)
    with pytest.raises(CheckpointError, match="version"):
        with read_source(source):
            pytest.fail("unsupported version accepted")


def test_scheduler_binding_and_manual_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("LIMITE_EVAL_LOGICAL_ID", "campaign")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    path = reserve(tmp_path, "attempt-r2")
    with create(path) as journal:
        assert journal.logical_id == "campaign"
        assert journal.metadata.slurm_job_id == "1234"
        assert journal.metadata.slurm_restart_count == 2
    monkeypatch.delenv("LIMITE_EVAL_LOGICAL_ID")
    with create(reserve(tmp_path, "manual"), source=path) as journal:
        assert journal.logical_id == "campaign"


def test_atomic_metadata_failure_preserves_previous_document(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    atomic_json(path, {"complete": False})

    def interrupted_replace(source, destination):
        raise OSError("interrupted before atomic publication")

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted"):
        atomic_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": False}


def _kill_before_import_publication(source, destination, connection):
    replace = os.replace

    def before_publish(temporary, target):
        if (
            Path(target).name == CHECKPOINT_FILE
            and json.loads(Path(temporary).read_text())["import_complete"]
        ):
            connection.send("imported-not-published")
            connection.recv()
        replace(temporary, target)

    os.replace = before_publish
    with create(destination, source=source):
        raise AssertionError("import was meant to be interrupted")


def test_killed_import_retains_source_lineage_and_can_resume(tmp_path):
    source = reserve(tmp_path, "first")
    populate(source, [sample(), sample(1)])
    interrupted = reserve(tmp_path, "second")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_kill_before_import_publication, args=(source, interrupted, child))
    process.start()
    try:
        assert parent.poll(20), "import did not reach its metadata publication boundary"
        assert parent.recv() == "imported-not-published"
    finally:
        process.kill()
        process.join(timeout=10)
        parent.close()
        child.close()
    assert process.exitcode == -9
    metadata = json.loads((interrupted / CHECKPOINT_FILE).read_text())
    assert metadata["import_complete"] is False
    assert metadata["source_attempt"] == str(source.resolve())
    with create(reserve(tmp_path, "third"), source=interrupted) as journal:
        assert journal.validate_complete() == [sample(), sample(1)]
        assert journal.logical_id == "first"


@pytest.mark.parametrize("field,value", [("gold", "changed"), ("problem_index", "0"), ("metrics", None)])
def test_schema_and_identity_checked_when_loading_source(tmp_path, field, value):
    source = reserve(tmp_path, "first")
    populate(source, [sample()])
    record = sample().model_dump()
    record[field] = value
    (source / "metrics/samples.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(CheckpointError, match="record|gold"):
        with read_source(source):
            pytest.fail("invalid stored record accepted")


def test_partial_schema_is_not_a_committed_sample(tmp_path):
    source = reserve(tmp_path, "first")
    populate(source, [])
    record = sample().model_dump()
    del record["protocols"]
    (source / "metrics/samples.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(CheckpointError, match="record"):
        with read_source(source):
            pytest.fail("missing stored scores were filled with schema defaults")


def test_missing_nested_scorer_verdict_is_not_filled_with_default(tmp_path):
    source = reserve(tmp_path, "first")
    row = sample(protocols={"exact": ProtocolSample(protocol="exact", correct=True)})
    populate(source, [row])
    record = row.model_dump()
    del record["protocols"]["exact"]["correct"]
    (source / "metrics/samples.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(CheckpointError, match="record"):
        with read_source(source):
            pytest.fail("missing stored scorer verdict was filled with null")


def test_declared_task_problem_rollout_order_and_group_ids_survive(tmp_path):
    tasks = [
        TaskContract(
            name="z",
            group_size=2,
            template_sha256="template",
            problems=[
                ProblemIdentity(gold_sha256=text_sha256("42"), source_group_id="g", source_variant_id=variant)
                for variant in ["second", "first"]
            ],
        ),
        task(name="a", group_size=1),
    ]
    rows = [
        sample(
            task="z", problem_index=problem, rollout=rollout, source_group_id="g", source_variant_id=variant
        )
        for problem, variant in enumerate(["second", "first"])
        for rollout in range(2)
    ] + [sample(task="a")]
    source = reserve(tmp_path, "first")
    with SampleJournal.create(source, contract={}, tasks=tasks) as journal:
        journal.record_fingerprint("template", fingerprint())
        for row in reversed(rows):
            journal.append(row)
        assert journal.validate_complete() == rows
    with SampleJournal.create(
        reserve(tmp_path, "second"), contract={}, tasks=tasks, source=source
    ) as journal:
        assert journal.validate_complete() == rows


def test_observed_engine_field_cannot_disappear_on_restart(tmp_path):
    source = reserve(tmp_path, "first")
    populate(source, [sample()])
    with create(reserve(tmp_path, "second"), source=source) as journal:
        with pytest.raises(CheckpointError, match="vllm_version"):
            journal.record_fingerprint("template", fingerprint(vllm_version=None))
