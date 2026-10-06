# Evaluation contract

A result is comparable only when its manifest agrees on the model artefact,
rendering bytes, tokenizer/vocabulary, terminators, stop strings, suite,
tasksets, scorer version, dataset revisions, sampling policy, group size, and
generation budget.

The evaluation contract has one suite:

- `math-extended`: AIME 2024/2025/2026, MATH-500, GSM8K, HMMT February
  2025/2026, OlympiadBench, BeyondAIME, and Apex Shortlist.

It has two profiles:

- `chat`, using the checkpoint-family chat rendering;
- `base-kshot`, using committed static exemplars and templates.

Each taskset uses all available anchored answer protocols. Unavailable published
graders remain unavailable with a recorded reason; they are never silently
converted into zeroes.

Every run writes immutable per-run artefacts under `outputs/eval/` by default.
A run directory is never reused or merged. GPU count and scheduler are
operational inputs, not measurement semantics, and therefore belong to the
caller rather than this repository.

The checkpoint allowlist fixes the Hub revision, stage, and rendering profile.
Model weights are never converted by the evaluator.
