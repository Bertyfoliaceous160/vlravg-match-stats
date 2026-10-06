"""Strict compatibility patch for vLLM 0.28.0's OLMo Hybrid GDN loader.

vLLM's native ``WeightsMapper`` carries a shard id for each source GDN
``q_conv1d``, ``k_conv1d`` and ``v_conv1d`` tensor.  The fused convolution
parameter requires that integer when its custom weight loader is called.  This
patch extracts that id through the native mapper and forwards it explicitly;
every other weight remains on the original ``OlmoHybridModel.load_weights``
path.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

EXPECTED_DISTRIBUTION_VERSION = "0.28.0+cu129"
EXPECTED_MODULE_VERSION = "0.28.0"

# This is the complete native stacked mapping in the inspected vLLM wheel, not
# a replacement mapping.  Checking it turns an upstream loader change into a
# startup error rather than applying a plausible patch to different semantics.
EXPECTED_STACKED_MAPPING = {
    ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
    ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
    ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
    ".linear_attn.q_proj": (".linear_attn.in_proj_qkvg", 0),
    ".linear_attn.k_proj": (".linear_attn.in_proj_qkvg", 1),
    ".linear_attn.v_proj": (".linear_attn.in_proj_qkvg", 2),
    ".linear_attn.g_proj": (".linear_attn.in_proj_qkvg", 3),
    ".q_conv1d": (".conv1d", 0),
    ".k_conv1d": (".conv1d", 1),
    ".v_conv1d": (".conv1d", 2),
    ".gate_proj": (".gate_up_proj", 0),
    ".up_proj": (".gate_up_proj", 1),
}

_ORIGINAL_ATTR = "__olmo_hybrid_vllm_plugin_original_load_weights__"
_PATCHED_ATTR = "__olmo_hybrid_vllm_plugin_patched_load_weights__"
_GDN_CONV_SUFFIXES = (".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")


class OlmoHybridCompatibilityError(RuntimeError):
    """Raised when this narrowly pinned patch cannot prove its vLLM contract."""


def apply_patch(
    olmo_hybrid_model_cls: type[Any], *, module_version: str, distribution_version: str
) -> bool:
    """Patch exactly one vLLM model method and return whether it was newly patched.

    ``OlmoHybridModel.load_weights`` keeps native handling for every tensor
    except the three split GDN convolution tensors. For those tensors, the
    native mapper selects the fused parameter and the patch passes its integer
    shard id directly to that parameter's existing ``weight_loader``. This is
    needed to retain the mapper metadata at the custom GDN loader boundary.

    Args:
        olmo_hybrid_model_cls: vLLM's imported ``OlmoHybridModel`` class.
        module_version: ``vllm.__version__`` from the running engine process.
        distribution_version: Installed ``vllm`` distribution version.

    Raises:
        OlmoHybridCompatibilityError: If the inspected wheel version or loader
            API/mapping does not exactly match this compatibility contract.
    """
    if module_version != EXPECTED_MODULE_VERSION or distribution_version != EXPECTED_DISTRIBUTION_VERSION:
        raise OlmoHybridCompatibilityError(
            "olmo-hybrid-vllm-plugin supports only vLLM "
            f"module {EXPECTED_MODULE_VERSION!r} / distribution {EXPECTED_DISTRIBUTION_VERSION!r}; "
            f"found module {module_version!r} / distribution {distribution_version!r}."
        )

    previous = getattr(olmo_hybrid_model_cls, _ORIGINAL_ATTR, None)
    patched = getattr(olmo_hybrid_model_cls, _PATCHED_ATTR, None)
    if previous is not None or patched is not None:
        if previous is not None and patched is olmo_hybrid_model_cls.load_weights:
            return False
        raise OlmoHybridCompatibilityError(
            "OlmoHybridModel.load_weights is already marked by an incompatible "
            "olmo-hybrid-vllm-plugin patch."
        )

    original = olmo_hybrid_model_cls.load_weights
    if tuple(inspect.signature(original).parameters) != ("self", "weights"):
        raise OlmoHybridCompatibilityError(
            "vLLM 0.28.0 OlmoHybridModel.load_weights no longer has the expected "
            "(self, weights) API."
        )

    mapper = getattr(olmo_hybrid_model_cls, "hf_to_vllm_mapper", None)
    stacked_mapping = getattr(mapper, "orig_to_new_stacked", None)
    map_name_with_shard = getattr(mapper, "_map_name_with_shard", None)
    if stacked_mapping != EXPECTED_STACKED_MAPPING or not callable(map_name_with_shard):
        raise OlmoHybridCompatibilityError(
            "vLLM 0.28.0 OLMo Hybrid's native stacked mapping or mapper API differs "
            "from the inspected GDN compatibility contract."
        )

    def patched_load_weights(self: Any, weights: Iterable[tuple[str, Any]]) -> set[str]:
        """Forward native GDN convolution shard ids, then delegate all other weights."""
        params = dict(self.named_parameters())
        manually_loaded: set[str] = set()

        def remaining_weights() -> Iterable[tuple[str, Any]]:
            for name, loaded_weight in weights:
                if not name.endswith(_GDN_CONV_SUFFIXES):
                    yield name, loaded_weight
                    continue

                mapped = self.hf_to_vllm_mapper._map_name_with_shard(name)
                if mapped is None:
                    continue
                mapped_name, shard_id = mapped
                if not isinstance(shard_id, int) or shard_id not in (0, 1, 2):
                    raise OlmoHybridCompatibilityError(
                        f"GDN convolution mapping for {name!r} did not retain shard id 0, 1, or 2."
                    )
                try:
                    parameter = params[mapped_name]
                except KeyError as error:
                    raise OlmoHybridCompatibilityError(
                        f"Native GDN convolution mapping produced missing parameter {mapped_name!r}."
                    ) from error
                weight_loader = getattr(parameter, "weight_loader", None)
                if not callable(weight_loader):
                    raise OlmoHybridCompatibilityError(
                        f"Fused GDN convolution parameter {mapped_name!r} has no callable weight_loader."
                    )
                weight_loader(parameter, loaded_weight, shard_id)
                manually_loaded.add(mapped_name)

        return set(original(self, remaining_weights())) | manually_loaded

    setattr(olmo_hybrid_model_cls, _ORIGINAL_ATTR, original)
    setattr(olmo_hybrid_model_cls, _PATCHED_ATTR, patched_load_weights)
    olmo_hybrid_model_cls.load_weights = patched_load_weights
    return True
