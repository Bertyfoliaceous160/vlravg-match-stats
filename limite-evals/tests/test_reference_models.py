"""The approved open-reference registry is data, not a moving Hub query."""

from __future__ import annotations

import re

from limite_evals.reference_models import REFERENCE_MODELS, find


def test_every_reference_contract_has_one_immutable_hub_identity() -> None:
    assert len(REFERENCE_MODELS) == 23
    assert {model.name for model in REFERENCE_MODELS} == {
        "qwen3-4b",
        "qwen3-8b",
        "qwen3.5-4b",
        "qwen3.8-27b",
        "qwen3.8-27b-fp8",
        "phi-4-mini-instruct",
        "minicpm5-2b",
        "olmo-hybrid-7b-base",
        "olmo-hybrid-7b-think-sft",
        "olmo-hybrid-7b-instruct-sft",
        "olmo-hybrid-7b-instruct-dpo",
        "olmo3-7b-base",
        "olmo3-7b-think-sft",
        "olmo3-7b-think-dpo",
        "olmo3-7b-think",
        "olmo3-7b-instruct-sft",
        "olmo3-7b-instruct-dpo",
        "olmo3-7b-instruct",
        "vibethinker-3b",
        "vibethinker-1.5b",
        "muse-glimmer-30b",
        "gemma4-31b-thinking",
        "qwen3.6-27b-thinking",
    }
    assert all(re.fullmatch(r"[0-9a-f]{40}", model.revision) for model in REFERENCE_MODELS)
    assert all(model.identity == f"{model.repo_id}@{model.revision}" for model in REFERENCE_MODELS)


def test_new_reasoning_models_pin_hub_decoding_and_template_defaults() -> None:
    muse = find("meta-models/Muse-Glimmer-30B")
    gemma = find("google/gemma-4-31B-it")
    qwen = find("Qwen/Qwen3.6-27B")

    assert muse is not None and muse.decoding.model_dump() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": None,
        "presence_penalty": None,
        "repetition_penalty": None,
    }
    assert gemma is not None and gemma.decoding == muse.decoding
    assert gemma.chat_template_kwargs == (("enable_thinking", True),)
    assert qwen is not None and qwen.decoding.model_dump() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert qwen.chat_template_kwargs == (("enable_thinking", True),)


def test_qwen35_records_its_conditional_generation_contract() -> None:
    qwen35 = find("Qwen/Qwen3.5-4B")
    assert qwen35 is not None
    assert qwen35.architecture == "Qwen3_5ForConditionalGeneration"
    assert qwen35.runtime_status == "native"
