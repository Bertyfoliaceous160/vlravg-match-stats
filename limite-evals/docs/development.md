# Development

Read the evaluator's [`AGENTS.md`](../AGENTS.md) before changing code.
Dependency management uses `uv`; `pyproject.toml` is the source of truth and
`uv.lock` must be regenerated whenever dependencies change.

Keep changes surgical and validate, at minimum:

```text
python -m compileall -q src packages tests
uvx ruff check --no-force-exclude src packages tests
uv lock --check
```

Run the CPU test suite once the checkpoint-resolver dependencies are available.
GPU tests must use a fresh run directory and preserve the command, environment,
Git revision, configuration, logs, and machine-readable results under
`outputs/eval/<unique-run-id>/` by default, or under the explicitly selected
`--out-root`.

The repository contains no cluster launcher. Allocate one or more GPUs using the
target environment's normal mechanism, then invoke the same evaluator CLI.
Never embed a personal partition, filesystem path, host name, or fixed GPU count
in the evaluator.

The parent `limite-vllm` project is the only local model-runtime dependency. Hub
artifacts are immutable inputs and use the standard Hugging Face cache.
