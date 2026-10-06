"""CPU contracts for the pinned OLMo Hybrid vLLM compatibility plugin."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import tomllib
import types
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "packages" / "olmo-hybrid-vllm-plugin"
PLUGIN_SRC = PLUGIN_ROOT / "src"
INSPECTED_SOURCE_SHA256 = {
    "model_executor/models/olmo_hybrid.py": "bdb1e965eda2f8445b7740a18f67bfda75d0b206a95a6067ef4bc41743f3c84f",
    "model_executor/models/utils.py": "a93f50ddf56cac4992dcb7699da48cf70207c8fb0e533beea8818c8f48b1e2d8",
    "model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py": (
        "4e5b144d109c6f13431e1113ca8cf28132df14e7988dfaad08008cd1e43b18a1"
    ),
}
sys.path.insert(0, str(PLUGIN_SRC))
from olmo_hybrid_vllm.patch import (  # noqa: E402
    EXPECTED_DISTRIBUTION_VERSION,
    EXPECTED_MODULE_VERSION,
    EXPECTED_STACKED_MAPPING,
    OlmoHybridCompatibilityError,
    apply_patch,
)


class FakeMapper:
    """Minimal native-mapper double retaining the production name semantics."""

    orig_to_new_stacked = EXPECTED_STACKED_MAPPING

    def _map_name_with_shard(self, name: str) -> tuple[str, int | str] | None:
        for source, (target, shard_id) in self.orig_to_new_stacked.items():
            if source in name:
                return name.replace(source, target, 1), shard_id
        return name, None


class FakeParameter:
    """Records the three-argument calls expected by the fused GDN loader."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, int]] = []

    def weight_loader(self, parameter: "FakeParameter", weight: Any, shard_id: int) -> None:
        assert parameter is self
        self.calls.append((weight, shard_id))


class FakeOlmoHybridModel:
    hf_to_vllm_mapper = FakeMapper()

    def __init__(self) -> None:
        self.conv = FakeParameter()
        self.native_seen: list[str] = []

    def named_parameters(self) -> Iterable[tuple[str, FakeParameter]]:
        return [("layers.0.linear_attn.conv1d.weight", self.conv)]

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        self.native_seen = [name for name, _ in weights]
        return set(self.native_seen)


def test_workspace_metadata_declares_a_discoverable_companion_plugin() -> None:
    root_metadata = tomllib.loads((PLUGIN_ROOT.parents[1] / "pyproject.toml").read_text())
    source_metadata = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text())

    assert "olmo-hybrid-vllm-plugin" in root_metadata["project"]["dependencies"]
    assert root_metadata["tool"]["uv"]["sources"]["olmo-hybrid-vllm-plugin"] == {"workspace": True}
    assert source_metadata["project"]["entry-points"]["vllm.general_plugins"] == {
        "olmo-hybrid-gdn-shards": "olmo_hybrid_vllm.register:register"
    }


def test_inspected_vllm_sources_are_still_the_pinned_wheel_contract() -> None:
    """The patch is valid only for the exact native mapper and fused loader."""
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None or not vllm_spec.submodule_search_locations:
        pytest.skip("vLLM is installed only in the Linux serving environment")
    vllm_root = Path(next(iter(vllm_spec.submodule_search_locations)))

    for relative_path, expected_sha256 in INSPECTED_SOURCE_SHA256.items():
        source_path = vllm_root / relative_path
        assert source_path.is_file(), f"missing inspected vLLM source: {relative_path}"
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == expected_sha256


def test_patch_forwards_gdn_convolution_shards_and_delegates_everything_else() -> None:
    assert apply_patch(
        FakeOlmoHybridModel,
        module_version=EXPECTED_MODULE_VERSION,
        distribution_version=EXPECTED_DISTRIBUTION_VERSION,
    )
    model = FakeOlmoHybridModel()
    weights = [
        ("layers.0.linear_attn.q_conv1d.weight", "q"),
        ("layers.0.linear_attn.k_conv1d.weight", "k"),
        ("layers.0.linear_attn.v_conv1d.weight", "v"),
        ("layers.0.linear_attn.a_proj.weight", "native"),
    ]

    assert model.load_weights(weights) == {
        "layers.0.linear_attn.conv1d.weight",
        "layers.0.linear_attn.a_proj.weight",
    }
    assert model.conv.calls == [("q", 0), ("k", 1), ("v", 2)]
    assert model.native_seen == ["layers.0.linear_attn.a_proj.weight"]
    assert not apply_patch(
        FakeOlmoHybridModel,
        module_version=EXPECTED_MODULE_VERSION,
        distribution_version=EXPECTED_DISTRIBUTION_VERSION,
    )


def test_patch_refuses_a_different_wheel_or_native_mapping() -> None:
    class WrongVersion(FakeOlmoHybridModel):
        pass

    with pytest.raises(OlmoHybridCompatibilityError, match="supports only vLLM"):
        apply_patch(
            WrongVersion,
            module_version="0.27.0",
            distribution_version=EXPECTED_DISTRIBUTION_VERSION,
        )

    class WrongMapping:
        hf_to_vllm_mapper = FakeMapper()

        def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
            return {name for name, _ in weights}

    WrongMapping.hf_to_vllm_mapper.orig_to_new_stacked = {}
    with pytest.raises(OlmoHybridCompatibilityError, match="native stacked mapping"):
        apply_patch(
            WrongMapping,
            module_version=EXPECTED_MODULE_VERSION,
            distribution_version=EXPECTED_DISTRIBUTION_VERSION,
        )


def test_register_loads_the_pinned_target_without_importing_a_real_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RegisteredModel:
        pass

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = EXPECTED_MODULE_VERSION
    fake_executor = types.ModuleType("vllm.model_executor")
    fake_models = types.ModuleType("vllm.model_executor.models")
    fake_olmo = types.ModuleType("vllm.model_executor.models.olmo_hybrid")
    fake_olmo.OlmoHybridModel = RegisteredModel
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", fake_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", fake_models)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.olmo_hybrid", fake_olmo)

    register = importlib.import_module("olmo_hybrid_vllm.register")
    monkeypatch.setattr(register, "version", lambda package: EXPECTED_DISTRIBUTION_VERSION)
    received: dict[str, Any] = {}

    def record_patch(model_cls: type[Any], **kwargs: str) -> bool:
        received["model_cls"] = model_cls
        received.update(kwargs)
        return True

    monkeypatch.setattr(register, "apply_patch", record_patch)
    register.register()

    assert received == {
        "model_cls": RegisteredModel,
        "module_version": EXPECTED_MODULE_VERSION,
        "distribution_version": EXPECTED_DISTRIBUTION_VERSION,
    }
