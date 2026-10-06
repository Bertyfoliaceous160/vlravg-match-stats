# Limite evaluation harness

This directory contains the standalone evaluator for immutable Limite Hugging
Face releases and pinned reference models. It never exports or rewrites model
weights.

## Evaluation surface

### Suites

- `math-extended`

`math-extended` contains AIME 2024/2025/2026, MATH-500, GSM8K, HMMT February
2025/2026, OlympiadBench, BeyondAIME, and Apex Shortlist. The suite reports all
available scoring views: published `exact`, pinned `reference`, `lenient`,
`permissive`, format, truncation, and comparison-timeout diagnostics.

### Rendering profiles

- `chat`: the checkpoint's immutable chat template.
- `base-kshot`: committed few-shot mathematics templates and exemplars.

The allowlist selects stage and profile automatically: Limite Base and Base
Soup are `pretrain` with `base-kshot`; Violetto is `posttrain` with `chat`; and
every external model is `reference` with `chat`.

### Reference models

The pinned reference-model registry includes the comparison checkpoints. OLMo
Hybrid checkpoints use the compatibility package under
`packages/olmo-hybrid-vllm-plugin/`.

## Scope boundaries

The evaluator accepts only allowlisted immutable Hugging Face checkpoints, the
`math-extended` suite, and the `chat` and `base-kshot` profiles. It does not
export trainer checkpoints or include scheduler-specific launchers.

Cluster orchestration is deliberately outside the evaluator. A caller can run
the evaluator on one GPU, several GPUs, or under any scheduler without importing
a Kyoto-specific launch policy.

## Command line

The installed commands are `limite-eval`, `limite-rescore`, `limite-collect`,
and `limite-compare`. Their `--help` output is the canonical option reference;
for example, run `uv run limite-eval --help`. Evaluation runs default to seed
`0` and write under `outputs/eval/<run-id>/` unless overridden.

## Package layout

- `src/limite_evals/`: evaluator, runner, reports, checkpoint resolution and CLI.
- `packages/limite-evals-core/`: scoring protocols, result schema and statistics.
- `packages/olmo-hybrid-vllm-plugin/`: OLMo Hybrid vLLM compatibility.
- `src/limite_evals/templates/`: committed rendering templates.
- `src/limite_evals/exemplars/`: committed base-kshot exemplar sets.
- `tests/`: CPU contract tests; GPU and end-to-end validation run in the Linux
  serving environment.

## Checkpoint resolution

The three released Limite repositories and every reference model are pinned to
immutable Hub commits. Their stages are part of the checkpoint registry rather
than caller-supplied CLI metadata. `huggingface_hub` and vLLM use the standard
Hugging Face cache, so existing snapshots and weights are reused. The local
`limite-vllm` package registers `LimiteForCausalLM` in the serving process.
