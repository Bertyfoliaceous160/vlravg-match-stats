# Architecture

The evaluator is split into three layers:

1. `limite_evals` resolves a checkpoint, selects a rendering, launches an
   OpenAI-compatible vLLM server, loads dataset rows, and records completions.
2. `limite_evals_core` binds published graders, computes the answer-protocol
   ladder, validates schemas, and aggregates statistics.
3. The OLMo Hybrid compatibility package patches only the registered OLMo Hybrid
   reference-model load path.

The public evaluation surface is deliberately small:

- suite: `math-extended`;
- profiles: checkpoint-selected `chat` or `base-kshot`;
- scoring lane: the complete mathematical answer-protocol set.

Templates, exemplars, dataset revisions, grader sources, and generation settings
are explicit inputs recorded in the run manifest. Reports and rescoring consume
stored samples; they do not need to regenerate completions.

Checkpoint resolution accepts only allowlisted immutable Hugging Face releases.
Limite Base and Base Soup select `pretrain`/`base-kshot`; Violetto selects
`posttrain`/`chat`; every external model selects `reference`/`chat`. vLLM
receives the repo id and commit and reuses the normal Hugging Face cache. The
parent `limite-vllm` package owns model registration.

Cluster scheduling is not part of the evaluator architecture. The same CLI must
be launchable by a local shell, Slurm, Kubernetes, or a provider-specific runner.
