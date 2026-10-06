from __future__ import annotations

import json
from pathlib import Path

import pytest

from limite_evals import resolve
from limite_evals.limite_models import LIMITE_MODELS
from limite_evals.reference_models import REFERENCE_MODELS

CONFIG = {
    "architectures": ["LimiteForCausalLM"],
    "model_type": "limite",
    "max_position_embeddings": 131072,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "pad_token_id": 151643,
}
GENERATION = {
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "pad_token_id": 151643,
}


def test_three_limite_releases_are_immutable_and_choose_the_profile() -> None:
    assert [(model.repo_id, model.profile) for model in LIMITE_MODELS] == [
        ("paradigma-inc/limite-1b-base", "base-kshot"),
        ("paradigma-inc/limite-1b-base-soup", "base-kshot"),
        ("paradigma-inc/limite-1b-violetto", "chat"),
    ]
    assert all(len(model.revision) == 40 for model in LIMITE_MODELS)


@pytest.mark.parametrize("model", LIMITE_MODELS, ids=lambda model: model.name)
def test_hub_resolution_validates_contract_and_returns_automatic_profile(monkeypatch, model) -> None:
    monkeypatch.setattr(
        resolve,
        "_hub_json",
        lambda _contract, filename: CONFIG if filename == "config.json" else GENERATION,
    )
    result = resolve.resolve(model.repo_id)
    assert result.model == model.repo_id
    assert result.revision == model.revision
    assert result.source_format == "limite-hub"
    assert result.profile == model.profile
    assert result.stage == model.stage
    assert result.architecture == "LimiteForCausalLM"
    assert result.max_model_len == 131072


def test_wrong_or_moving_hub_identity_is_refused() -> None:
    model = LIMITE_MODELS[0]
    with pytest.raises(resolve.CheckpointResolutionError, match="approved immutable revision"):
        resolve.detect_format(f"{model.repo_id}@{'0' * 40}")
    with pytest.raises(resolve.CheckpointResolutionError, match="not allowlisted"):
        resolve.detect_format("someone/moving-model")


@pytest.mark.parametrize("field", ["bos_token_id", "eos_token_id", "pad_token_id"])
def test_limite_token_contract_is_fatal(field: str) -> None:
    broken = {**CONFIG, field: 151645}
    with pytest.raises(resolve.CheckpointResolutionError, match=field):
        resolve._validate_config(broken, LIMITE_MODELS[0], "fixture")
    with pytest.raises(resolve.CheckpointResolutionError, match=field):
        resolve._validate_limite_generation({**GENERATION, field: 151645}, "fixture")


def test_standard_hf_cache_snapshot_is_accepted(tmp_path: Path) -> None:
    model = LIMITE_MODELS[0]
    snapshot = (
        tmp_path
        / "hub"
        / "models--paradigma-inc--limite-1b-base"
        / "snapshots"
        / model.revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(CONFIG))
    (snapshot / "generation_config.json").write_text(json.dumps(GENERATION))
    (snapshot / "model.safetensors").write_bytes(b"weights")

    result = resolve.resolve(str(snapshot))
    assert result.source_format == "limite-local"
    assert result.profile == "base-kshot"
    assert result.stage == "pretrain"
    assert result.revision == model.revision


def test_reference_models_define_the_reference_stage() -> None:
    assert REFERENCE_MODELS
    assert {model.stage for model in REFERENCE_MODELS} == {"reference"}


def test_hub_json_uses_hf_download_with_the_pinned_revision(tmp_path: Path, monkeypatch) -> None:
    payload = tmp_path / "config.json"
    payload.write_text(json.dumps(CONFIG))
    calls = []

    def download(*, repo_id, filename, revision, token):
        calls.append((repo_id, filename, revision, token))
        return str(payload)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model = LIMITE_MODELS[0]
    assert resolve._hub_json(model, "config.json") == CONFIG
    assert calls == [(model.repo_id, "config.json", model.revision, None)]


def test_raw_export_inputs_are_not_supported(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"trainer")
    with pytest.raises(resolve.CheckpointResolutionError, match="Exporters are intentionally unsupported"):
        resolve.detect_format(str(checkpoint))
