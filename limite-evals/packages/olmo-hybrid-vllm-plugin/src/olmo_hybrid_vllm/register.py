"""vLLM general-plugin entry point for the OLMo Hybrid GDN loader patch."""

from __future__ import annotations

from importlib.metadata import version

from olmo_hybrid_vllm.patch import apply_patch


def register() -> None:
    """Install the pinned patch in every vLLM engine process, idempotently.

    General plugins run in the API process and in spawned engine workers. Any
    version or API mismatch raises a clear error before a checkpoint is loaded.
    """
    import vllm
    from vllm.model_executor.models.olmo_hybrid import OlmoHybridModel

    apply_patch(
        OlmoHybridModel,
        module_version=vllm.__version__,
        distribution_version=version("vllm"),
    )
