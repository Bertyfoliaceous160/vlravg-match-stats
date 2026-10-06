"""Resolve immutable Hugging Face checkpoints accepted by the evaluator.

Limite releases and reference baselines are already complete Hugging Face
artifacts. The evaluator never exports or rewrites weights: it validates a
pinned contract and passes the repo id plus commit to vLLM. Hugging Face's
normal cache remains authoritative, so cached files are reused automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from limite_evals.limite_models import LIMITE_MODELS, LimiteModel
from limite_evals.limite_models import find as find_limite_model
from limite_evals.profiles import ProfileName
from limite_evals.reference_models import REFERENCE_MODELS, ReferenceModel
from limite_evals.reference_models import find as find_reference_model
from limite_evals_core.schema import Stage

LIMITE_ARCHITECTURE = "LimiteForCausalLM"
LIMITE_MODEL_TYPE = "limite"
LIMITE_TOKEN_ID = 151643

CheckpointFormat = Literal["limite-hub", "limite-local", "reference-hub", "reference-local"]
HubContract: TypeAlias = LimiteModel | ReferenceModel

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_HASH_CHUNK = 1 << 20


class CheckpointResolutionError(RuntimeError):
    """The checkpoint is not an immutable artifact admitted by this instrument."""


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """The validated artifact passed to vLLM and recorded in the manifest."""

    model: str
    source: str
    source_format: CheckpointFormat
    stage: Stage
    sha256: str
    architecture: str
    revision: str | None
    max_model_len: int
    profile: ProfileName


def resolve(checkpoint: str) -> ResolvedCheckpoint:
    """Resolve an allowlisted Hub id or an immutable local Hub snapshot."""
    source_format = detect_format(checkpoint)
    if source_format in ("limite-hub", "reference-hub"):
        contract = _hub_contract(checkpoint)
        config = _hub_json(contract, "config.json")
        _validate_config(config, contract, checkpoint)
        if isinstance(contract, LimiteModel):
            _validate_limite_generation(
                _hub_json(contract, "generation_config.json"), checkpoint
            )
        return ResolvedCheckpoint(
            model=contract.repo_id,
            source=contract.identity,
            source_format=source_format,
            stage=contract.stage,
            sha256=_hub_sha256(contract.identity),
            architecture=contract.architecture,
            revision=contract.revision,
            max_model_len=_native_max_model_len(config, checkpoint),
            profile=contract.profile if isinstance(contract, LimiteModel) else "chat",
        )

    path = Path(checkpoint).expanduser()
    config = json.loads((path / "config.json").read_text())
    contract = _local_contract(path, config)
    if contract is None:
        raise CheckpointResolutionError(
            f"{path} is not an immutable local snapshot of an allowlisted Limite or reference model"
        )
    _validate_config(config, contract, str(path))
    if isinstance(contract, LimiteModel):
        generation_path = path / "generation_config.json"
        if not generation_path.is_file():
            raise CheckpointResolutionError(f"{path} has no generation_config.json")
        _validate_limite_generation(json.loads(generation_path.read_text()), str(path))
    return ResolvedCheckpoint(
        model=str(path),
        source=checkpoint,
        source_format=source_format,
        stage=contract.stage,
        sha256=artifact_sha256(path),
        architecture=contract.architecture,
        revision=contract.revision,
        max_model_len=_native_max_model_len(config, str(path)),
        profile=contract.profile if isinstance(contract, LimiteModel) else "chat",
    )


def detect_format(checkpoint: str) -> CheckpointFormat:
    """Recognise only immutable allowlisted Hugging Face artifacts."""
    path = Path(checkpoint).expanduser()
    if not path.exists():
        if _looks_like_a_path(checkpoint):
            raise CheckpointResolutionError(
                f"no such checkpoint path: {checkpoint}. It looks like a path rather than a Hub repo id"
            )
        contract = _hub_contract(checkpoint)
        return "limite-hub" if isinstance(contract, LimiteModel) else "reference-hub"

    if path.is_file():
        if path.suffix == ".pt":
            raise CheckpointResolutionError(
                f"{path} is a raw trainer checkpoint. Exporters are intentionally unsupported; "
                "evaluate one of the immutable Hugging Face releases instead."
            )
        raise CheckpointResolutionError(f"{path} is a file, not a checkpoint directory")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise CheckpointResolutionError(f"{path} has no config.json")
    config = json.loads(config_path.read_text())
    contract = _local_contract(path, config)
    if contract is None:
        raise CheckpointResolutionError(
            f"{path} is not an immutable local snapshot of an allowlisted Hub model. "
            "Use its pinned Hub id, or a cache snapshot path containing the approved commit."
        )
    return "limite-local" if isinstance(contract, LimiteModel) else "reference-local"


def _hub_contract(checkpoint: str) -> HubContract:
    repo_id, separator, revision = checkpoint.rpartition("@")
    if not separator:
        repo_id, revision = checkpoint, ""
    contract = find_limite_model(repo_id) or find_reference_model(repo_id)
    if contract is None:
        known = [model.repo_id for model in LIMITE_MODELS] + [
            model.repo_id for model in REFERENCE_MODELS
        ]
        raise CheckpointResolutionError(
            f"{checkpoint!r} is not allowlisted. Hub inputs must name one of: {', '.join(known)}"
        )
    if revision and revision != contract.revision:
        raise CheckpointResolutionError(
            f"{checkpoint!r} does not name the approved immutable revision {contract.identity}"
        )
    return contract


def _local_contract(path: Path, config: dict[str, Any]) -> HubContract | None:
    repo_id = config.get("_name_or_path")
    revision = _local_revision(path, config)
    if not isinstance(repo_id, str) or not repo_id:
        repo_id = _cache_repo_id(path)
    if not repo_id or not revision:
        return None
    contract = find_limite_model(repo_id) or find_reference_model(repo_id)
    return contract if contract is not None and contract.revision == revision else None


def _cache_repo_id(path: Path) -> str | None:
    """Recover ``org/repo`` from a standard Hugging Face cache snapshot path."""
    for parent in (path, *path.parents):
        if parent.name.startswith("models--"):
            parts = parent.name.removeprefix("models--").split("--")
            return "/".join(parts) if len(parts) >= 2 else None
    return None


def _local_revision(path: Path, config: dict[str, Any]) -> str | None:
    for key in ("_commit_hash", "commit_hash", "revision"):
        candidate = config.get(key)
        if isinstance(candidate, str) and _IMMUTABLE_REVISION.fullmatch(candidate):
            return candidate
    for parent in (path, *path.parents):
        if parent.parent.name == "snapshots" and _IMMUTABLE_REVISION.fullmatch(parent.name):
            return parent.name
    return None


def _hub_json(contract: HubContract, filename: str) -> dict[str, Any]:
    """Read immutable Hub JSON through the standard reusable HF cache."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=contract.repo_id,
            filename=filename,
            revision=contract.revision,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except EntryNotFoundError as exc:
        raise CheckpointResolutionError(f"{contract.identity} has no {filename}") from exc
    except Exception as exc:
        raise CheckpointResolutionError(
            f"could not read cached or remote {filename} for {contract.identity}: {exc}"
        ) from exc
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointResolutionError(f"invalid {filename} for {contract.identity}: {exc}") from exc


def _validate_config(config: dict[str, Any], contract: HubContract, source: str) -> None:
    architectures = config.get("architectures")
    if architectures != [contract.architecture] or config.get("model_type") != contract.model_type:
        raise CheckpointResolutionError(
            f"{source} config differs from its allowlist contract: expected "
            f"architectures={[contract.architecture]!r}, model_type={contract.model_type!r}; got "
            f"architectures={architectures!r}, model_type={config.get('model_type')!r}"
        )
    if config.get("trust_remote_code") is True:
        raise CheckpointResolutionError(f"{source} requests trust_remote_code")
    if isinstance(contract, ReferenceModel) and contract.runtime_status != "native":
        raise CheckpointResolutionError(
            f"{source} is registered as {contract.runtime_status} and is not launchable"
        )
    if isinstance(contract, LimiteModel):
        for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if config.get(field) != LIMITE_TOKEN_ID:
                raise CheckpointResolutionError(
                    f"{source} config has {field}={config.get(field)!r}; expected {LIMITE_TOKEN_ID}"
                )


def _validate_limite_generation(config: dict[str, Any], source: str) -> None:
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if config.get(field) != LIMITE_TOKEN_ID:
            raise CheckpointResolutionError(
                f"{source} generation_config has {field}={config.get(field)!r}; expected {LIMITE_TOKEN_ID}"
            )


def _native_max_model_len(config: dict[str, Any], source: str) -> int:
    value = config.get("max_position_embeddings")
    if not isinstance(value, bool) and isinstance(value, int) and value > 0:
        return value
    text_config = config.get("text_config")
    nested = text_config.get("max_position_embeddings") if isinstance(text_config, dict) else None
    if (
        config.get("model_type") in {"muse_glimmer", "gemma4", "qwen3_5"}
        and not isinstance(nested, bool)
        and isinstance(nested, int)
        and nested > 0
    ):
        return nested
    raise CheckpointResolutionError(
        f"{source} config has invalid max_position_embeddings={value!r}"
    )


def artifact_sha256(directory: Path) -> str:
    """Hash local config and top-level safetensors without loading weights in memory."""
    digest = hashlib.sha256()
    paths = [directory / "config.json", *sorted(directory.glob("*.safetensors"))]
    if len(paths) == 1:
        raise CheckpointResolutionError(f"{directory} has no top-level safetensors weights")
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                digest.update(chunk)
    return digest.hexdigest()


def _hub_sha256(identity: str) -> str:
    return hashlib.sha256(f"hf-hub:{identity}".encode()).hexdigest()


def _looks_like_a_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~")) or value.endswith((".pt", ".safetensors"))
