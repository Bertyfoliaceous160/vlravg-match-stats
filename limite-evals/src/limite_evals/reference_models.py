"""Pinned, text-only reference models admitted to the evaluation instrument.

This registry is intentionally a small allowlist rather than a convenience list
of popular Hub names.  A reference run is reproducible only when the exact Hub
revision, native architecture, and model type are all known before the engine
starts.  The resolver consumes these contracts for both Hub IDs and local Hub
snapshots; it never infers a revision from a moving branch.

``Qwen3.5-4B`` is conditional-generation rather than a conventional CausalLM,
but its text path was smoke-tested on the engine pin current when it entered
this registry's native evaluation queue.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

RuntimeStatus = Literal["native", "text-runtime-gated"]


class ReferenceDecoding(BaseModel):
    """Hub-published sampling defaults that define a reference measurement."""

    model_config = ConfigDict(frozen=True)

    temperature: float
    top_p: float
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None


@dataclass(frozen=True)
class ReferenceModel:
    """Immutable Hub identity and architecture contract for one reference model."""

    name: str
    repo_id: str
    revision: str
    architecture: str
    model_type: str
    stage: Literal["reference"] = "reference"
    runtime_status: RuntimeStatus = "native"
    decoding: ReferenceDecoding | None = None
    chat_template_kwargs: tuple[tuple[str, bool], ...] = ()

    @property
    def identity(self) -> str:
        """The Hub locator recorded in manifests and passed to vLLM with its revision."""
        return f"{self.repo_id}@{self.revision}"


# Every revision is an immutable 40-hex Git commit, not a branch or a tag.
# Reference checkpoints all carry their own immutable ``reference`` stage.
REFERENCE_MODELS: tuple[ReferenceModel, ...] = (
    ReferenceModel(
        "qwen3-4b", "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "Qwen3ForCausalLM", "qwen3"
    ),
    ReferenceModel(
        "qwen3-8b", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "Qwen3ForCausalLM", "qwen3"
    ),
    ReferenceModel(
        "qwen3.5-4b",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5",
    ),
    ReferenceModel(
        "qwen3.8-27b",
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5",
    ),
    ReferenceModel(
        "qwen3.8-27b-fp8",
        "Qwen/Qwen3.8-27B-FP8",
        "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5",
    ),
    ReferenceModel(
        "vibethinker-3b",
        "WeiboAI/VibeThinker-3B",
        "77bd2cced09193c8b9a59a32bd8577bbd1f3e01c",
        "Qwen2ForCausalLM",
        "qwen2",
    ),
    ReferenceModel(
        "vibethinker-1.5b",
        "WeiboAI/VibeThinker-1.5B",
        "9614bcb0264f03f18f2fa2406c4a9754635f51de",
        "Qwen2ForCausalLM",
        "qwen2",
    ),
    ReferenceModel(
        "phi-4-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "cfbefacb99257ffa30c83adab238a50856ac3083",
        "Phi3ForCausalLM",
        "phi3",
    ),
    ReferenceModel(
        "minicpm5-2b",
        "openbmb/MiniCPM5-2B",
        "a063f08de1bd09dfc9ae4cf3da35e6064949e533",
        "LlamaForCausalLM",
        "llama",
    ),
    ReferenceModel(
        "olmo-hybrid-7b-base",
        "allenai/Olmo-Hybrid-7B",
        "4f1cc566f9fdf3ce68da2ab6a788a83d89896dcf",
        "OlmoHybridForCausalLM",
        "olmo_hybrid",
    ),
    ReferenceModel(
        "olmo-hybrid-7b-think-sft",
        "allenai/Olmo-Hybrid-Think-SFT-7B",
        "79d1f7f613a9f98169e6c9f00880ab2df4383860",
        "OlmoHybridForCausalLM",
        "olmo_hybrid",
    ),
    ReferenceModel(
        "olmo-hybrid-7b-instruct-sft",
        "allenai/Olmo-Hybrid-Instruct-SFT-7B",
        "15ec262d91de5a2729e445a1c190c17573239a8a",
        "OlmoHybridForCausalLM",
        "olmo_hybrid",
    ),
    ReferenceModel(
        "olmo-hybrid-7b-instruct-dpo",
        "allenai/Olmo-Hybrid-Instruct-DPO-7B",
        "ec62da5af0106ab4e83a60e07c4796ff358fd204",
        "OlmoHybridForCausalLM",
        "olmo_hybrid",
    ),
    ReferenceModel(
        "olmo3-7b-base",
        "allenai/Olmo-3-1025-7B",
        "a81bae42db3975be1671e27b9c9a56da1a9f980f",
        "Olmo3ForCausalLM",
        "olmo3",
    ),
    ReferenceModel(
        "olmo3-7b-think-sft",
        "allenai/Olmo-3-7B-Think-SFT",
        "6ff857587e040d6d523a3d5f3a56e918f5401d66",
        "Olmo3ForCausalLM",
        "olmo3",
    ),
    ReferenceModel(
        "olmo3-7b-think-dpo",
        "allenai/Olmo-3-7B-Think-DPO",
        "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38",
        "Olmo3ForCausalLM",
        "olmo3",
    ),
    ReferenceModel(
        "olmo3-7b-think",
        "allenai/Olmo-3-7B-Think",
        "d97e442d7cc678210054dbcc9b440894d62c89a4",
        "Olmo3ForCausalLM",
        "olmo3",
    ),
    ReferenceModel(
        "olmo3-7b-instruct-sft",
        "allenai/Olmo-3-7B-Instruct-SFT",
        "e1452fc572d51966ff4aaeb25118b891eb93e549",
        "Olmo3ForCausalLM",
        "olmo3",
    ),
    ReferenceModel(
        "olmo3-7b-instruct-dpo",
        "allenai/Olmo-3-7B-Instruct-DPO",
        "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1",
        "Olmo3ForCausalLM",
        "olmo3",
    ),
    ReferenceModel(
        "olmo3-7b-instruct",
        "allenai/Olmo-3-7B-Instruct",
        "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        "Olmo3ForCausalLM",
        "olmo3",
    ),
    ReferenceModel(
        "muse-glimmer-30b",
        "meta-models/Muse-Glimmer-30B",
        "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        "MuseGlimmerForConditionalGeneration",
        "muse_glimmer",
        decoding=ReferenceDecoding(temperature=1.0, top_p=0.95, top_k=64),
    ),
    ReferenceModel(
        "gemma4-31b-thinking",
        "google/gemma-4-31B-it",
        "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "Gemma4ForConditionalGeneration",
        "gemma4",
        decoding=ReferenceDecoding(temperature=1.0, top_p=0.95, top_k=64),
        chat_template_kwargs=(("enable_thinking", True),),
    ),
    ReferenceModel(
        "qwen3.6-27b-thinking",
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5",
        decoding=ReferenceDecoding(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        ),
        chat_template_kwargs=(("enable_thinking", True),),
    ),
)

BY_REPO = {model.repo_id.casefold(): model for model in REFERENCE_MODELS}


def find(repo_id: str) -> ReferenceModel | None:
    """Return the admitted contract for ``repo_id`` without normalising its spelling."""
    return BY_REPO.get(repo_id.casefold())
