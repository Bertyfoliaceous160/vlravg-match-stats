"""The pipeline, end to end: resolve, render, serve, evaluate, aggregate, report.

Every step already exists as a library function; this module is the order they go
in and the manifest they are recorded under. Three things it decides that no
single step could:

**The run directory is claimed before anything is spent, and after the rendering
is named.** `serve` writes the chat template and the engine log into `artifacts/`
before there are any numbers, so the directory has to exist before the GPU does
anything -- and decision 14 says a run never overwrites an older one, so the
claim has to be the thing that fails if the identifier is taken. What comes
*before* the claim is which template each taskset will be served with, because
that question is answered from names alone and one of its answers is a refusal:
a suite that will not render under the requested profile, a taskset with no
committed k-shot template, an unknown profile. Claiming first meant a refused run
left an empty directory behind and burned its identifier under decision 14's
no-reuse rule, which is litter for the operator and a run id nobody can have
back. Reading the artifact's own bytes still happens after the claim, since it
needs a resolved checkpoint.

**One engine per distinct rendering, not per taskset.** The template is what a
profile varies, and under `base-kshot` the three math tasksets share a template
while GSM8K has its own. Grouping by template hash means two engine starts
instead of four, and one instead of four under `chat`. A cold start is minutes,
so this is not a micro-optimisation.

**The manifest's `template_sha256` covers the whole run's rendering**, not one
taskset's. It is the digest of the `{taskset: rendering hash}` map, so two runs
comparing equal there rendered every taskset the same way. The per-taskset hash
the engine was gated against is that taskset's own `Rendering.sha256()`, and it
is what `serve` checks; these are two jobs and deliberately two values.

**The rendering is now an input rather than a consequence of the profile.**
`--chat-template` names a committed template or a file, `--stop` replaces its
stop strings, and both are resolved before the engine starts so that the digest
covers the bytes actually served. Without that the chat profile hashed a
constant -- the same digest for every chat run of every checkpoint -- while the
artifact hash excluded the tokenizer, so two supervised-fine-tuned checkpoints
served with different templates produced manifests that compared equal (F-01).

**Each scored sample is durable before waiting for its siblings.** The frozen
ordered inputs and measurement contract precede generation. Recovery imports
the prior bank into a fresh attempt and submits only missing sample keys. Final
reporting consumes that journal without rewriting it; completeness is published
only after full coverage and the final report have been validated.

**Generation respects the checkpoint's native context.** `base-kshot` defaults
to greedy decoding with an 8192-token cap and an optional thinking prefix. Other
profiles default to sampled decoding and the remaining native context. The
immutable `config.json` supplies `max_position_embeddings`; `token_budget` keeps
4096 tokens for the rendered prompt and gives the remainder to every taskset.
The benchmark protocol budgets remain reference metadata, not serving defaults.
An explicit `--max-tokens` replaces native-max but stays capped at the same safe
ceiling. `RunManifest.generation_budget_policy` keeps these measurements apart.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections.abc import Callable, Set
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from limite_evals import (
    aggregate,
    profiles,
    reference_models,
    report,
    runner,
    suites,
    tasksets,
    templates,
)
from limite_evals import evaluation_inputs as inputs
from limite_evals.checkpoint import SampleJournal
from limite_evals.profiles import EXEMPLAR_DIR, EXEMPLAR_SETS, Profile
from limite_evals.resolve import (
    CheckpointResolutionError,
    ResolvedCheckpoint,
)
from limite_evals.resolve import resolve as resolve_checkpoint
from limite_evals.serve import (
    DEFAULT_DATA_PARALLEL_SIZE,
    DEFAULT_ENGINE_VENV,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_PORT,
    PINNED_VLLM_VERSION,
    ServeError,
    Server,
    error,
    info,
    ok,
    serve,
    warn,
)
from limite_evals.suites import TasksetSpec
from limite_evals.templates import Rendering, TemplateResolutionError
from limite_evals_core.answer_scoring import answer_scorer_revisions
from limite_evals_core.equivalence import MATH_VERIFY_VERSION
from limite_evals_core.protocols import Unavailable
from limite_evals_core.schema import (
    PROTOCOL_METRICS,
    Fingerprint,
    MetricSet,
    Pins,
    RunManifest,
    RunSummary,
    SampleResult,
    Sampling,
    SamplingPolicy,
    SpeculativeDecoding,
)

#: The revisions used by the evaluation contract. Transcribed rather
#: than read from the environment: they identify what produced a number, so a manifest has to
#: carry them even when this repository is installed as a wheel. The `verifiers`
#: value is checked against `[tool.uv.sources]` by `tests/test_cli.py`, which is
#: what keeps it from drifting away from what is actually installed.
VERIFIERS_PIN = "d30a3f48e5f14b06b3081b2102ec32cc3149b849"
RESEARCH_ENVIRONMENTS_PIN = "f9c43a74"

DEFAULT_DTYPE = "bfloat16"

#: Tokens held back from the generation budget for the prompt itself.
#:
#: The native context is a total sequence length, so assigning all of it to the
#: completion would reject every non-empty prompt. The largest rendered prompt in
#: the shipped tasksets is 2850 tokens; 4096 covers it and leaves the default
#: independent of the benchmark's historical generation budget.
CONTEXT_RESERVE = 4096


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_dir = run(args)
    except (
        CheckpointResolutionError,
        TemplateResolutionError,
        ServeError,
        FileExistsError,
        ValueError,
        KeyError,
    ) as failure:
        error(str(failure))
        return 1
    ok(f"wrote {run_dir}")
    return 0


def run(args: argparse.Namespace) -> Path:
    """The whole pipeline for one checkpoint, returning its run directory."""
    # Resolve per-run defaults on a copy: matrix cells reuse the parsed options.
    args = argparse.Namespace(**vars(args))
    resolved = resolve_checkpoint(args.checkpoint)
    profile = profiles.resolve(resolved.profile)
    if args.thinking and profile.name != "base-kshot":
        raise ValueError("--thinking is available only for a base-kshot checkpoint")
    if args.sampling is None:
        args.sampling = "greedy" if profile.name == "base-kshot" else "avg-k"
    if args.max_tokens is None and profile.name == "base-kshot":
        args.max_tokens = 8192
    specs = _specs(args.suite, args.tasksets)
    profile = _profile_with_reference_defaults(args.checkpoint, profile)
    specs, profile = _under_sampling_policy(specs, profile, args.sampling, args.temperature)
    requested = _requested_templates(specs, profile, args)
    run_id = args.run_id or _run_id(resolved.stage, profile.name)
    run_dir = report.reserve(run_id, root=Path(args.out_root))
    info(f"run {run_id} in {run_dir}")
    info(
        f"client concurrency {_client_concurrency(args)} across "
        f"{args.data_parallel_size} data-parallel replica(s), "
        f"{args.tensor_parallel_size} tensor-parallel shard(s) each"
    )

    context = served_context(resolved, args.max_model_len)
    args.max_model_len = context.max_model_len
    info(f"serving {resolved.model} ({resolved.source_format}, sha256 {resolved.sha256[:12]})")

    # Before the engine, because an artifact that carries no chat template must
    # be refused here rather than after the GPU has bound it: under `chat` the
    # engine would start, the fingerprint gate would pass, and every completion
    # request would answer 400 from inside the subprocess (F-12).
    renderings = _renderings(
        specs,
        profile,
        requested,
        resolved.model,
        resolved.revision,
        args,
    )

    prepared = {
        spec.name: inputs.prepare(
            spec,
            profile,
            renderings[spec.name],
            budget=token_budget(args.max_model_len, args.max_tokens),
            limit=args.limit,
        )
        for spec in specs
    }
    declared = {
        template_sha: Fingerprint(
            architecture=resolved.architecture,
            artifact_sha256=resolved.sha256,
            template_sha256=template_sha,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
        for template_sha in _serving_groups(specs, renderings)
    }
    manifest = _manifest(run_id, resolved, profile, specs, declared, renderings, args)
    contract = {
        "manifest": inputs.manifest_identity(manifest),
        "model_revision": resolved.revision,
        "source_format": resolved.source_format,
        "local_tokenizer_sha256": inputs.local_tokenizer_identity(resolved.model),
        "tasksets": {spec.name: prepared[spec.name].identity for spec in specs},
    }
    # One counter for the whole run, not one per taskset: a resend is a fact
    # about the transport this run used, and the tasksets share a client policy.
    retries = runner.TransportRetries()
    report.write_status(run_dir, completed=[], pending=[spec.name for spec in specs])
    report.write_transport_retries(run_dir, retries.as_dict())

    def summary_of(scored: tuple[TasksetSpec, ...], journal: SampleJournal) -> RunSummary:
        """The run over `scored`, with a manifest that describes those tasksets only.

        A partial run's manifest cannot be the whole run's: `max_tokens` records
        the largest *effective* budget and the dataset pin names every taskset,
        so stating them over tasksets that never ran would put revisions and a
        budget beside numbers they did not produce. Built from what was scored,
        the manifest is true of the samples beside it -- and it will not compare
        equal to a full run of the same suite, which is the correct answer.
        """
        names = {spec.name for spec in scored}
        return aggregate.summarize(
            _manifest(run_id, resolved, profile, scored, journal.fingerprints, renderings, args),
            [sample for sample in journal.samples if sample.task in names],
            metric_sets=_metric_sets(scored),
            answer_formats=_answer_formats(scored),
        )

    with SampleJournal.create(
        run_dir,
        contract=contract,
        tasks=[prepared[spec.name].task_contract for spec in specs],
        source=Path(args.resume_from) if args.resume_from else None,
    ) as journal:

        def completed(spec: TasksetSpec) -> bool:
            keys = journal.completed_keys
            return all(
                (spec.name, problem, rollout) in keys
                for problem in range(len(prepared[spec.name].problems))
                for rollout in range(spec.group_size)
            )

        def progress() -> list[TasksetSpec]:
            done = [spec for spec in specs if completed(spec)]
            saved = len(journal.completed_keys)
            info(
                f"checkpoint {run_id}: expected {expected}, reused {reused}, "
                f"new {saved - reused}, pending {expected - saved}"
            )
            report.write_status(
                run_dir,
                completed=[spec.name for spec in done],
                pending=[spec.name for spec in specs if spec not in done],
            )
            return done

        expected = sum(len(item.problems) * item.task_contract.group_size for item in prepared.values())
        reused = len(journal.completed_keys)
        info(f"evaluation {journal.logical_id}, attempt {run_id}; source {args.resume_from or 'none'}")
        progress()
        try:
            for template_sha, group in _serving_groups(specs, renderings).items():
                pending = [spec for spec in group if not completed(spec)]
                if not pending:
                    continue
                info(f"rendering {', '.join(spec.name for spec in pending)} at template {template_sha[:12]}")
                with serve(
                    resolved.model,
                    revision=resolved.revision,
                    declared=declared[template_sha],
                    rendering=renderings[pending[0].name],
                    taskset=pending[0].name,
                    artifact_sha256=resolved.sha256,
                    run_dir=run_dir,
                    engine_venv=Path(args.engine_venv),
                    port=args.port,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    data_parallel_size=args.data_parallel_size,
                    tensor_parallel_size=args.tensor_parallel_size,
                    enforce_eager=args.enforce_eager,
                    hf_overrides=context.hf_overrides,
                    mtp_speculative_tokens=args.mtp_speculative_tokens,
                    max_num_seqs=args.max_num_seqs,
                    expected_vllm_version=args.expected_vllm_version,
                ) as server:
                    journal.record_fingerprint(template_sha, server.fingerprint)
                    # Detect drift across rendering groups before accepting any
                    # response from the newly started engine.
                    _run_fingerprint(journal.fingerprints, renderings, specs)
                    for spec in pending:
                        _evaluate(
                            spec,
                            server,
                            profile,
                            renderings[spec.name],
                            args,
                            retries,
                            prepared=prepared[spec.name],
                            completed_keys={
                                (problem, rollout)
                                for task, problem, rollout in journal.completed_keys
                                if task == spec.name
                            },
                            on_sample=journal.append,
                        )
                        progress()
                        report.write_transport_retries(run_dir, retries.as_dict())
                        if retries.total:
                            warn(
                                f"{retries.total} rollout request(s) resent after a transport fault; "
                                "see artifacts/transport-retries.json"
                            )
        except Exception:
            report.write_transport_retries(run_dir, retries.as_dict())
            done = progress()
            if done:
                report.write_partial(run_dir, summary_of(tuple(done), journal))
                warn(f"{len(done)} of {len(specs)} tasksets finished; durable progress is in {run_dir}")
            raise

        samples = journal.validate_complete()
        summary = summary_of(specs, journal)
        inputs.validate_summary_manifest(summary, journal.metadata)
        result = report.write_run(summary, samples, root=Path(args.out_root), persist_samples=False)
        info(f"checkpoint {run_id}: reused {reused}, new {len(samples) - reused}, pending 0")
        return result


@dataclass(frozen=True)
class ServedContext:
    """The total context this run serves at."""

    max_model_len: int
    hf_overrides: dict[str, Any] | None


def served_context(resolved: ResolvedCheckpoint, max_model_len: int | None) -> ServedContext:
    """Use the artifact context unless the operator provides an explicit cap."""
    return ServedContext(
        max_model_len=resolved.max_model_len if max_model_len is None else max_model_len,
        hf_overrides=None,
    )


def token_budget(max_model_len: int, override: int | None = None) -> int:
    """Return native-max completion capacity, or a bounded explicit override.

    ``max_model_len`` is the model's total context capacity. The default leaves
    the fixed prompt reserve free: the budget is a property of the served model,
    not of the taskset, which is why no taskset declares one. ``override`` is the
    operator's explicit completion cap and cannot exceed the same safe ceiling.
    """
    ceiling = max_model_len - CONTEXT_RESERVE
    if ceiling <= 0:
        raise ValueError(
            f"max model length {max_model_len} is not larger than the {CONTEXT_RESERVE}-token prompt reserve"
        )
    return ceiling if override is None else min(override, ceiling)


def _client_concurrency(args: argparse.Namespace) -> int:
    """Resolve an explicit limit or scale the measured per-replica default."""
    if args.concurrency is not None:
        return args.concurrency
    return runner.DEFAULT_CONCURRENCY_PER_REPLICA * args.data_parallel_size


def _requested_templates(
    specs: tuple[TasksetSpec, ...], profile: Profile, args: argparse.Namespace
) -> dict[str, str]:
    """Which template each taskset will be served with, named but not yet read.

    The precedence is `--chat-template` over the profile's own default, and it is
    visible here rather than buried in the registry: a run that names a template
    serves that template for every taskset, which is the escape hatch for a
    rendering nobody has committed, while a run that names none gets the one its
    profile implies.

    Split out of `_renderings` and called **before the run directory is claimed**,
    because every refusal that follows from the names alone belongs on the near
    side of that claim: a suite that declares the requested profile out of scope,
    a taskset with no committed k-shot template, an unknown profile. Those runs
    used to `mkdir` an empty run directory first and then fail, which under
    decision 14's no-reuse rule burned the identifier permanently. Nothing here
    touches the filesystem or the artifact, so there is nothing to undo.

    Base and Soup receive committed k-shot renderings; Violetto and admitted
    references use the template carried by their immutable Hub artifact.
    """
    return {
        spec.name: args.chat_template or templates.for_profile(profile.name, spec.name)
        for spec in specs
    }


def _renderings(
    specs: tuple[TasksetSpec, ...],
    profile: Profile,
    requested: dict[str, str],
    artifact: str,
    artifact_revision: str | None,
    args: argparse.Namespace,
) -> dict[str, Rendering]:
    """The bytes each taskset is served with, resolved before the engine starts.

    `requested` is what `_requested_templates` already named; what is added here
    is reading the bytes, which is why this cannot run as early: a rendering that
    resolves to the artifact's own template needs a resolved checkpoint. `--stop`
    replaces the stop strings on whichever template won.

    Resolved for every taskset up front so that a failure to pin the rendering --
    a missing file, an artifact carrying no chat template -- happens before the
    run directory is filled and before any GPU time is spent.
    """
    resolved = {
        spec.name: templates.resolve(
            taskset=spec.name,
            template=requested[spec.name],
            stop=args.stop,
            artifact=artifact,
            artifact_revision=artifact_revision,
            prompt_identity=profile.prompt_identity(spec.name),
            chat_template_kwargs=dict(profile.chat_template_kwargs),
        )
        for spec in specs
    }
    if args.thinking:
        resolved = {
            name: replace(rendering, template=rendering.template + r"{{ '\n<think>\n' }}")
            for name, rendering in resolved.items()
        }
    unbounded = sorted(name for name, r in resolved.items() if not r.stop)
    if unbounded and profile.name != "chat":
        warn(
            f"{', '.join(unbounded)}: served with no stop strings under `{profile.name}`. "
            "A base checkpoint has no reliable terminator, so every rollout will run to the "
            "token cap and decision 10 will score it wrong. Name the boundary with `--stop`."
        )
    return resolved


def _evaluate(
    spec: TasksetSpec,
    server: Server,
    profile: Profile,
    rendering: Rendering,
    args: argparse.Namespace,
    retries: runner.TransportRetries | None = None,
    *,
    prepared: inputs.PreparedTask | None = None,
    completed_keys: Set[tuple[int, int]] = frozenset(),
    on_sample: Callable[[SampleResult], None] | None = None,
) -> list[SampleResult]:
    """One taskset against a bound engine, rendered with its own instruction.

    Rendering goes through the raw rows, not through `(problem, answer)` pairs,
    because a taskset whose reference builds its instruction per row needs fields
    the pair has already dropped. `--limit` still slices the dataset's own rows in
    the dataset's own order, before anything is rendered: it writes itself into
    the recorded dataset revision, so a change in *which* rows are scored would
    quietly change what a preflight measured.
    """
    if prepared is None:
        prepared = inputs.prepare(
            spec,
            profile,
            rendering,
            budget=token_budget(args.max_model_len, args.max_tokens),
            limit=args.limit,
        )
    problems, budget, bindings = prepared.problems, prepared.budget, prepared.bindings
    requested = args.max_tokens
    if requested is not None and budget < requested:
        warn(
            f"{spec.name}: generation budget clamped from {requested} to {budget}; "
            f"it does not fit a {args.max_model_len}-token context"
        )
    elif requested is None:
        info(
            f"{spec.name}: native-max generation budget {budget} tokens from "
            f"{args.max_model_len}-token context"
        )
    else:
        warn(
            f"{spec.name}: generation budget set to {budget}; this run is not "
            "comparable with a native-max run"
        )
    if isinstance(bindings.exact, Unavailable):
        warn(f"{spec.name}: `exact` is unavailable and reads as absent -- {bindings.exact.reason}")
    info(f"{spec.name}: {len(problems)} problems x {spec.group_size} rollouts at {budget} tokens")
    return asyncio.run(
        runner.run_taskset(
            spec,
            problems,
            base_url=server.base_url,
            model=server.model,
            profile=profile,
            rendering=rendering,
            bindings=bindings,
            max_tokens=budget,
            seed=args.seed,
            concurrency=_client_concurrency(args),
            timeout=args.request_timeout,
            retries=retries,
            completed_keys=completed_keys,
            on_sample=on_sample,
        )
    )


def _metric_sets(specs: tuple[TasksetSpec, ...]) -> dict[str, MetricSet]:
    """What each taskset reports: the protocols, or its own declaration.

    A taskset declares what it reports in a `metrics` field on its spec; read
    through `getattr` because only the tasksets that report something other than
    the protocols carry one. A constraint-checked taskset has no boxed answer to
    extract and therefore no anchor to bind, so it keeps its own declaration and
    is not coerced into columns that would be meaningless for it.

    Everything else reports `PROTOCOL_METRICS` rather than the ladder. That is
    the change D-103 asks for: a column now names the artefact that produced it,
    and `exact` -- the benchmark's own published grader -- is the headline
    wherever one is bound.
    """
    return {spec.name: getattr(spec, "metrics", None) or PROTOCOL_METRICS for spec in specs}


def _answer_formats(specs: tuple[TasksetSpec, ...]) -> dict[str, str]:
    """Which reference notation each taskset's strict lens reads.

    Read straight off the mathematical taskset specification.
    """
    return {spec.name: spec.answer_format for spec in specs}


def _serving_groups(
    specs: tuple[TasksetSpec, ...], renderings: dict[str, Rendering]
) -> dict[str, list[TasksetSpec]]:
    """Tasksets that render identically, so they can share one engine."""
    groups: dict[str, list[TasksetSpec]] = {}
    for spec in specs:
        groups.setdefault(renderings[spec.name].sha256(), []).append(spec)
    return groups


def _manifest(
    run_id: str,
    resolved: ResolvedCheckpoint,
    profile: Profile,
    specs: tuple[TasksetSpec, ...],
    served: dict[str, Fingerprint],
    renderings: dict[str, Rendering],
    args: argparse.Namespace,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=resolved.source,
        stage=resolved.stage,
        profile=profile.name,
        suite=args.suite,
        format_policy=args.format_policy,
        expected_vllm_version=args.expected_vllm_version,
        generation_budget_policy="native-max" if args.max_tokens is None else "override",
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        truncation_policy=args.truncation_policy,
        exemplars_sha256=_exemplars_sha256(profile, specs, renderings),
        relaxation_bases=_relaxation_bases(specs),
        sampling_policy=args.sampling,
        rope_scaling=None,
        speculative_decoding=(
            SpeculativeDecoding(method="qwen3_5_mtp", num_speculative_tokens=args.mtp_speculative_tokens)
            if args.mtp_speculative_tokens is not None
            else None
        ),
        compilation_config=(
            "mode0-full-decode-only" if args.mtp_speculative_tokens is not None else "default"
        ),
        sampling=Sampling(
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            min_p=profile.min_p,
            presence_penalty=profile.presence_penalty,
            repetition_penalty=profile.repetition_penalty,
            max_tokens=token_budget(args.max_model_len, args.max_tokens),
            stop=_stop_for_run(specs, renderings),
            seed=args.seed,
            # The detokenisation every completion was read back through, which is
            # the runner's constant rather than an argument: what text a grader
            # sees is not an axis this instrument offers, and recording it is how
            # a reader establishes which decode produced a number without reading
            # the runner at that revision.
            skip_special_tokens=runner.SKIP_SPECIAL_TOKENS,
        ),
        pins=Pins(
            verifiers=VERIFIERS_PIN,
            research_environments=RESEARCH_ENVIRONMENTS_PIN,
            math_verify=MATH_VERIFY_VERSION,
            dataset_revision=_dataset_revisions(specs, limit=args.limit),
            task_prompt_revision=_task_prompt_revisions(specs),
            native_scorer_revision="",
            answer_scorer_revision=answer_scorer_revisions(spec.name for spec in specs),
        ),
        fingerprint=_run_fingerprint(served, renderings, specs),
    )


def _stop_for_run(specs: tuple[TasksetSpec, ...], renderings: dict[str, Rendering]) -> list[str]:
    """Every stop string a run over these tasksets sends, for the manifest.

    `Sampling.stop` is one field and the stop set is per taskset -- the same
    rough edge `Sampling.max_tokens` carries, resolved the same way: openly.
    Recording one taskset's set as if it were the run's would put strings in the
    record that were never sent, which is worse than the coarseness it hides. So
    the field records the union, deduplicated and ordered by first appearance in
    suite order, and the authoritative per-taskset record stays the rendering
    digest, which hashes each taskset's own stop strings under decision 9.
    """
    seen: dict[str, None] = {}
    for spec in specs:
        seen.update(dict.fromkeys(renderings[spec.name].stop))
    return list(seen)


def _relaxation_bases(specs: tuple[TasksetSpec, ...]) -> str:
    """Which anchor each taskset's relaxations relaxed, in one comparable string.

    D-108 lets the base fall back per taskset, so `lenient` means
    `lenient(exact)` on MATH-500 and `lenient(reference)` on AIME. Those are not
    the same quantity, and two runs that resolved them differently are not
    comparable -- a published grader becoming bindable between two runs would
    otherwise change what a column measures while every other field of the
    manifest matched. Joined deterministically for the same reason
    `Pins.dataset_revision` is: string equality is what `comparable_with` needs.
    """
    return ",".join(
        f"{spec.name}={aggregate.relaxation_base(tasksets.taskset_protocols(spec))}"
        for spec in sorted(specs, key=lambda s: s.name)
    )


def _run_fingerprint(
    served: dict[str, Fingerprint],
    renderings: dict[str, Rendering],
    specs: tuple[TasksetSpec, ...],
) -> Fingerprint:
    """What the run served, with the rendering identified for the whole run.

    Every engine start in a run serves the same artifact in the same environment,
    so anything the starts disagree on other than the template is a drift that
    must not be averaged into one manifest -- it is raised instead.
    """
    fingerprints = list(served.values())
    first = fingerprints[0]
    for other in fingerprints[1:]:
        differences = {
            field: (getattr(first, field), getattr(other, field))
            for field in Fingerprint.model_fields
            if field not in {"template_sha256", "engine_venv"}
            and getattr(first, field) != getattr(other, field)
        }
        if differences:
            raise ServeError(f"the engine changed between tasksets: {differences}")
    return first.model_copy(update={"template_sha256": _rendering_sha256(renderings, specs)})


def _rendering_sha256(renderings: dict[str, Rendering], specs: tuple[TasksetSpec, ...]) -> str:
    payload = json.dumps({spec.name: renderings[spec.name].sha256() for spec in specs}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _exemplars_sha256(
    profile: Profile, specs: tuple[TasksetSpec, ...], renderings: dict[str, Rendering]
) -> str | None:
    """Decision 7's hash, over the exemplar files this run actually used.

    None for the profiles that use none, rather than the digest of nothing, so a
    reader can tell "no exemplars" from "exemplars whose hash I did not record".

    Each base-kshot taskset uses one committed static exemplar file.
    The file is selected from the template actually served, with the taskset's
    declared exemplar set as the fallback for an explicitly supplied compatible
    template.
    """
    if profile.name != "base-kshot":
        return None
    digest = hashlib.sha256()
    static = {
        templates.EXEMPLAR_SET_FOR_TEMPLATE.get(renderings[spec.name].name, EXEMPLAR_SETS[spec.name])
        for spec in specs
    }
    for name in sorted(static):
        digest.update(name.encode())
        digest.update((EXEMPLAR_DIR / f"{name}.json").read_bytes())
    return digest.hexdigest()


def _dataset_revisions(specs: tuple[TasksetSpec, ...], *, limit: int | None = None) -> str:
    """Every taskset's pinned revision in one comparable string.

    `Pins` carries one field and a suite has four revisions, so they are joined
    deterministically: two runs over the same tasksets at the same revisions
    produce the same string, which is what `RunManifest.comparable_with` needs.
    GSM8K's source taskset pins none, and that shows up as `unpinned` rather than as
    an empty gap.

    `--limit` lands here rather than beside the numbers because a truncated
    taskset **is different data**, and this field is what says which data was
    scored. Writing `@first8` into it makes `comparable_with` reject a preflight
    against a full run mechanically -- which is the whole point, since an
    eight-problem MATH-500 number is otherwise indistinguishable from a real one.
    """
    suffix = f"@first{limit}" if limit else ""
    return ",".join(
        f"{spec.name}={spec.revision or 'unpinned'}{suffix}" for spec in sorted(specs, key=lambda s: s.name)
    )


def _task_prompt_revisions(specs: tuple[TasksetSpec, ...]) -> str:
    """Source identities for tasksets with a separately revised prompt contract."""
    return ",".join(
        f"{spec.name}={spec.prompt_revision}"
        for spec in sorted(specs, key=lambda s: s.name)
        if spec.prompt_revision
    )


def _under_sampling_policy(
    specs: tuple[TasksetSpec, ...],
    profile: Profile,
    policy: SamplingPolicy,
    temperature: float | None = None,
) -> tuple[tuple[TasksetSpec, ...], Profile]:
    """Temperature, top-p and group size, which a policy sets together or not at all.

    Separating them is the trap. At temperature zero a group of 32 draws one
    sample 32 times: the run spends 32x what it measures, and its avg@32 column
    reports a spread of zero that is a property of the decoder rather than of the
    checkpoint. So `greedy` collapses the group in the same move that drops the
    temperature, and the manifest records the pair under one name -- which is
    what `RunManifest.comparable_with` then refuses to mix.

    The profile still owns the rendering. What a policy overrides is how the
    engine draws from it, never what it is shown.
    """
    if policy == "avg-k":
        return specs, profile if temperature is None else replace(profile, temperature=temperature)
    if temperature is not None:
        raise ValueError("--temperature cannot be combined with --sampling greedy")
    grouped = sorted({spec.group_size for spec in specs} - {1})
    if grouped:
        info(f"greedy: temperature 0, top-p 1, and group size {grouped} collapsed to 1")
    return (
        tuple(replace(spec, group_size=1) for spec in specs),
        replace(
            profile,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
        ),
    )


def _profile_with_reference_defaults(checkpoint: str, profile: Profile) -> Profile:
    """Apply immutable Hub decoding defaults for an admitted reference model."""
    repo_id = checkpoint.partition("@")[0]
    model = reference_models.find(repo_id)
    if model is None or model.decoding is None:
        return profile
    return replace(
        profile,
        **model.decoding.model_dump(),
        chat_template_kwargs=model.chat_template_kwargs,
    )


def _specs(suite: str, names: str | None) -> tuple[TasksetSpec, ...]:
    """A launchable suite, or its named subset in the suite's own order."""
    all_specs = suites.resolve(suite)
    if not names:
        return all_specs
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    known = {spec.name for spec in all_specs}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(f"suite {suite!r} has no taskset {unknown} (known: {', '.join(sorted(known))})")
    return tuple(spec for spec in all_specs if spec.name in wanted)


def _run_id(stage: str, profile: str) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{stage}-{profile}"


def _positive_finite_float(raw: str) -> float:
    value = float(raw)
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limite-eval",
        description="Evaluate an immutable Limite or reference Hugging Face checkpoint.",
    )
    parser.add_argument("checkpoint", help="a checkpoint directory or a Hugging Face repo id")
    parser.add_argument(
        "--suite",
        default="math-extended",
        help="pinned taskset suite; math-extended is the default claim-bearing comparison set",
    )
    parser.add_argument(
        "--sampling",
        default=None,
        choices=("avg-k", "greedy"),
        help="default: greedy for base-kshot, avg-k otherwise; greedy is temperature 0, "
        "top-p 1 and one rollout per problem",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="base-kshot only: append a thinking prefix to the few-shot prompt (off by default)",
    )
    parser.add_argument(
        "--temperature",
        type=_positive_finite_float,
        default=None,
        help="avg-k sampling temperature; default is the profile standard (0.6)",
    )
    parser.add_argument("--tasksets", help="comma-separated subset of the suite; default is all of it")
    parser.add_argument(
        "--format-policy",
        default="strict",
        choices=("strict", "lenient", "permissive"),
        help="ablation only; it selects nothing in the scoring path. It is recorded in the "
        "manifest, where it stops a run comparing against one that declared a different policy",
    )
    parser.add_argument(
        "--truncation-policy",
        default=aggregate.DEFAULT_TRUNCATION_POLICY,
        choices=("score", "fail"),
        help="what a completion that ran out of generation budget scores. `score` (the "
        "default) grades the text it produced; `fail` makes it wrong at every correctness "
        "metric. Neither drops the rollout. It is recorded in the manifest and gates "
        "comparability",
    )
    parser.add_argument(
        "--chat-template",
        help="a committed template name (%s) or the path of a .jinja file, overriding the "
        "one the profile implies. The bytes actually served are hashed into the fingerprint, "
        "so naming a different template re-baselines against a run that did not"
        % ", ".join(templates.names()),
    )
    parser.add_argument(
        "--stop",
        action="append",
        help="a stop string, repeatable, replacing the rendering's defaults on any profile. "
        "Hashed with the template under decision 9; a base rendering without any runs every "
        "rollout to the token cap, and decision 10 then scores all of them wrong",
    )
    parser.add_argument("--run-id", help="default is a UTC timestamp with the stage and profile")
    parser.add_argument(
        "--resume-from",
        help="new-format attempt directory to recover into a fresh run; only missing samples run",
    )
    parser.add_argument("--out-root", default=str(report.DEFAULT_ROOT))
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="total context capacity; default reads the checkpoint's immutable config.json",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="generation budget for every taskset; default is 8192 for base-kshot, "
        "otherwise native context minus "
        f"the {CONTEXT_RESERVE}-token prompt reserve",
    )
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--engine-venv", default=str(DEFAULT_ENGINE_VENV))
    parser.add_argument(
        "--expected-vllm-version",
        default=PINNED_VLLM_VERSION,
        help="exact vLLM version the live engine must report",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=DEFAULT_DATA_PARALLEL_SIZE,
        help="engine replicas behind one endpoint; throughput only, never scoring",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=_positive_int,
        default=1,
        help="GPU shards per engine replica; recorded because it changes serving arithmetic",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=_positive_int,
        default=None,
        help="vLLM scheduler sequence capacity; throughput only",
    )
    parser.add_argument(
        "--mtp-speculative-tokens",
        type=_positive_int,
        default=None,
        help="enable Qwen3.5 MTP with this many speculative tokens; also selects the pinned compilation mode",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA graph and torch.compile capture in vLLM. Opt-in because it changes "
        "the serving mode; recorded in the manifest and runs with different modes do not compare",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="maximum in-flight client requests; default is 320 per data-parallel replica",
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_finite_float,
        default=runner.DEFAULT_REQUEST_TIMEOUT,
        help="seconds allowed for connection, write, and pool operations; completion reads are unbounded",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base sampling seed (default: 0). Each rollout derives its own from it, "
        "so a group still draws k different samples",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="score only the first N problems of each taskset. For a bounded preflight; "
        "the run records it as a different dataset and will not compare with a full one. "
        "Grouped benchmarks require a boundary that retains every fixed variant",
    )
    return parser
