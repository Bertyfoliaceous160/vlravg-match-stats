"""The result schema: what a run writes down, and what makes two runs comparable.

Everything a reader needs to decide whether two numbers may be compared lives in
`RunManifest`. A number without its manifest is diagnostic only; the schema
states that rule as a type rather than leaving it to an operator runbook.

`Fingerprint` is the serving contract. It is captured from the live engine after
it binds and compared against what the run declared; a mismatch in a fatal field
aborts before any GPU time is spent, because a number produced by a different
architecture, artifact, or context length than the one recorded is not merely
imprecise, it is mislabelled.

`MetricSet` is what a taskset says it reports. The five-rung ladder used to be
the schema itself; it is now one declaration among possible others, because a
constraint-checked taskset reports prompt-level and instruction-level accuracy
and no rung can hold either. A taskset that reports something else is not a
broken ladder, and the summary has to be able to say so rather than coerce it.

`RunManifest.instrument_era` is the comparability key nothing else can stand in
for. It is derived from the record rather than declared, and it separates the
runs whose completions reached the grader with their reasoning delimiters intact
from the runs where E42 had already deleted them. Numbers do not cross it:
`assert_comparable_with` raises, and `limite_evals.rescore` refuses to rescore the
far side at all, because a rescore reads the stored text and the stored text is
the damage.

`ProtocolRecord` and `ProtocolSample` are what makes a column readable. A number
scored under `lenient` is meaningless without the anchor it relaxed and the
waivers it applied, and under D-108 that anchor **varies by taskset**: MATH-500
relaxes `exact`, because a published grader exists for it, while AIME relaxes
`reference`, because none does. Those are two different quantities under one
column heading, so the base is recorded per taskset and per sample rather than
inferred from whichever anchor happened to produce a number elsewhere in the
table. A protocol that bound no artefact records the reason it could not and
carries `correct = None`: absence and failure are different facts, and a reader
who cannot tell them apart reads an unobtainable grader as a model that got
every problem wrong.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

FormatPolicy = Literal["strict", "lenient", "permissive"]
Stage = Literal["pretrain", "sft", "posttrain", "reference"]
Profile = Literal["chat", "base-kshot"]
#: How a run drew its samples. `avg-k` is the suite's own policy: temperature
#: 0.6, top-p 0.95, and each taskset's declared group size, reported as avg@k.
#: `greedy` is temperature 0, top-p 1 and one rollout per problem -- the protocol
#: a published single-sample number is quoted under. The two are different
#: quantities in the same column, so `comparable_with` refuses to mix them.
SamplingPolicy = Literal["avg-k", "greedy"]
#: Where a run's completion ceiling came from. Historical manifests used the
#: taskset reference budgets; native-max is a different measurement even when a
#: coarse maximum happens to coincide.
GenerationBudgetPolicy = Literal["reference", "native-max", "override"]
#: What a completion that ran out of generation budget scores.
#:
#: `score` reads whatever the grader recovers from the text it did produce, and
#: counts the rollout in the denominator either way. `fail` is the older rule:
#: the completion is wrong at every correctness metric whatever its text says.
#:
#: They are not two presentations of one number and the difference is not small.
#: A completion is truncated because the model did not reach an end, and under
#: `score` a column can be raised by text that was never concluded -- most of all
#: at `permissive`, whose `answer_anywhere` waiver reads the abandoned reasoning.
#: Under `fail` a column can be lowered by a budget rather than by the
#: checkpoint. Both are real, they point opposite ways, and neither is
#: recoverable from the other's number, so `comparable_with` refuses to mix them
#: and the truncation rate is reported beside every column under either policy.
#:
#: Neither policy drops a rollout. Scoring only the completions that terminated
#: would remove the failures along with them -- a model runs out of context on
#: the problems it could not finish -- and that number is not available from this
#: instrument under any flag.
TruncationPolicy = Literal["score", "fail"]

#: Why a completion stopped, which `finish_reason` alone cannot say.
#:
#: `finish_reason` reads `"stop"` both for a completion that ran into one of the
#: rendering's stop strings and for one that emitted a learned end-of-sequence
#: token, and telling those two apart is exactly what separates a pretraining
#: checkpoint from a post-trained one: the first has no reliable terminator and
#: is halted from outside, the second ends by itself. vLLM's `stop_reason` beside
#: it carries the distinction -- the matched string, or a token id, or nothing at
#: all when the model simply finished -- so the split is read from the engine
#: rather than guessed from the text.
StopClass = Literal["stop_string", "eos", "stop_token", "length", "unknown"]
MetricPresentation = Literal["percent", "points"]
NativeAggregation = Literal["sample_mean", "source_group_mean", "source_group_all"]

#: Which instrument produced a number, on the one axis that cannot be recovered
#: from the number itself.
#:
#: `e42` is every legacy run served before instrument bug E42 was found: the
#: pre-Limite artifact tokenizer could not name ids 151665 and 151666, so `<think>`
#: and `</think>` decoded to the empty string and **every reasoning delimiter was
#: deleted between the engine and the grader**. The completions in those runs'
#: `samples.jsonl` are the destroyed text, not the model's output, and no token
#: ids were persisted beside them. `corrected` is every run served through the
#: pinned vocabulary, whose completions carry the delimiters the model emitted.
#:
#: The two are not two settings of one instrument. Every think-dependent
#: quantity -- `format_ok`, the A--T case letters, the strict lens on either
#: answer format, the relaxations' `unclosed_think` waiver -- answers a different
#: question in each, and five of the twenty cases were unreachable in the first.
#: So the era is a comparability key of the same standing as the suite, and
#: `RunManifest.assert_comparable_with` refuses across it rather than warning.
InstrumentEra = Literal["corrected", "e42"]

#: The identity of the scoring path a summary was produced by, stamped into every
#: manifest `aggregate.summarize` writes.
#:
#: It exists because a stored verdict and a stored **classification** are not the
#: same kind of record. A verdict is recomputed from the completion whenever
#: anything asks for it; an A--T case letter is written once by the scoring path
#: and read back by aggregation, which is a cache -- and a cache with no version
#: on it is a cache that keeps answering after the function behind it changed.
#: `aggregate.failure_case` therefore trusts a stored letter only from this
#: version and reclassifies otherwise.
#:
#: Bump it whenever a change moves what any scorer here recovers or how the
#: taxonomy partitions a completion. It is not a package version: it names the
#: scoring semantics, and two checkouts that score identically may share one.
SCORER_VERSION = "2026.10-limite-evals-v1"

#: A mismatch in any of these means the served model is not the one the run
#: declared, so the run aborts. Everything else in the fingerprint is recorded
#: and warned about but does not stop the run.
FATAL_FINGERPRINT_FIELDS = ("architecture", "artifact_sha256", "template_sha256", "max_model_len", "dtype")


class InstrumentEraMismatch(Exception):
    """Raised where two runs from different instrument eras would be compared.

    An exception rather than a reason in a list, because the reasons
    `comparable_with` returns are advice a caller may weigh and this is not:
    the numbers on the two sides of this boundary were produced from different
    *text*, and no aggregation, delta or case-count table across it means
    anything. It carries both eras so the message names the boundary.
    """


class Fingerprint(BaseModel):
    """What the engine is actually serving, read back after it binds."""

    architecture: str
    artifact_sha256: str
    template_sha256: str
    max_model_len: int
    dtype: str
    vllm_version: str | None = None
    engine_venv: str | None = None
    #: The vocabulary the completions were written in and read back through,
    #: as `repo@revision` or a directory. `None` for an artifact served with its
    #: own tokenizer, and on every summary written before instrument bug E42 was
    #: found -- which is the same set, since affected legacy runs served the
    #: artifact's incompatible tokenizer.
    #:
    #: Recorded rather than fatal. It is already inside `template_sha256`, which
    #: is fatal, so gating on it a second time would compare a declaration
    #: against itself; what it is for is a reader who has to tell a corrected
    #: number from an E42-era one without recomputing a digest.
    tokenizer: str | None = None
    #: The ids the engine was launched to stop on. Read from the vocabulary where
    #: one pins it and from the artifact's `generation_config.json` otherwise, so
    #: this is what the engine did rather than what a file declared -- and the two
    #: disagreed on affected legacy runs before the correction.
    eos_token_ids: list[int] | None = None

    @property
    def records_vocabulary(self) -> bool:
        """Whether this run wrote down which alphabet it served through.

        The two fields arrived with the E42 correction and are the same pair
        `report._vocabulary_rows` prints, so their presence is the primary
        witness that a run was served by the corrected instrument. Their absence
        is not by itself proof of the opposite -- a reference baseline is served
        its own tokenizer and pins no terminator, so it records neither and is
        perfectly corrected-era. That is why `RunManifest.instrument_era` reads a
        second witness before deciding, and why this is a plain question about
        the record rather than a verdict about the era.
        """
        return self.tokenizer is not None or bool(self.eos_token_ids)

    def mismatches(self, declared: Fingerprint) -> dict[str, tuple[object, object]]:
        """Fields where this differs from `declared`, as `{field: (declared, observed)}`."""
        return {
            name: (getattr(declared, name), getattr(self, name))
            for name in type(self).model_fields
            if getattr(self, name) is not None
            and getattr(declared, name) is not None
            and getattr(self, name) != getattr(declared, name)
        }

    @staticmethod
    def fatal(mismatches: dict[str, tuple[object, object]]) -> dict[str, tuple[object, object]]:
        return {k: v for k, v in mismatches.items() if k in FATAL_FINGERPRINT_FIELDS}


class ProtocolRecord(BaseModel):
    """What one protocol meant for one taskset in this run.

    The report renders a column from this, never from the protocol's name alone.
    `anchor` names the artefact that produced the number -- a published grader, a
    reference verifier, or a base plus a waiver list -- and `relaxes` names the
    anchor a relaxation actually relaxed, which under D-108 is `exact` on a
    taskset that has a published grader and `reference` on one that does not.

    `unavailable_reason` is the D-107 state: no artefact could be bound, so the
    column is absent and says why. It is not a score of zero and no surface may
    render it as one.
    """

    model_config = ConfigDict(frozen=True)

    protocol: str
    anchor: str | None = None
    #: The closed set of things this protocol forgives on top of its base, empty
    #: for an anchored protocol. A list in the record rather than an adjective in
    #: a docstring, so what a number forgave is recoverable from the number.
    waivers: tuple[str, ...] = ()
    relaxes: str | None = None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def describe(self) -> str:
        """The protocol as a reader has to see it: never the bare name.

        D-108's binding mitigation is that no rendering surface may show
        `lenient` or `permissive` without the base it relaxed, so the base is
        built into the one string every surface formats.
        """
        if self.relaxes is not None:
            waived = ", ".join(self.waivers)
            return f"{self.protocol} (relaxes {self.relaxes}: {waived})"
        return self.protocol


class ProtocolSample(BaseModel):
    """One completion's verdict under one protocol.

    `correct is None` means no artefact was bound, so no verdict exists. It is
    deliberately not `False`: a taskset whose published grader is unobtainable
    would otherwise be recorded as a checkpoint that answered every problem
    wrongly, which is the one reading D-107 exists to prevent.
    """

    protocol: str
    anchor: str | None = None
    correct: bool | None = None
    unavailable_reason: str | None = None
    relaxes: str | None = None


class Pins(BaseModel):
    """The versions that can move a number without any model changing.

    A change to any of these invalidates comparison with earlier runs. They are
    recorded per run rather than per repository so a summary read months later
    still says what produced it.
    """

    verifiers: str
    research_environments: str
    math_verify: str
    dataset_revision: str
    #: Per-taskset prompt source identities. Blank on legacy manifests and on
    #: suites whose prompt contract needs no separate provenance boundary.
    task_prompt_revision: str = ""
    #: Per-taskset identities for deterministic benchmark-native scoring
    #: compositions. Blank on legacy manifests and suites without native scorers.
    native_scorer_revision: str = ""
    #: Per-taskset revisions for changed answer-scoring contracts. Blank on
    #: historical manifests and tasksets whose answer scorer has not changed.
    answer_scorer_revision: str = ""


class Sampling(BaseModel):
    temperature: float
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    #: How the engine detokenised each completion before it was scored, as the
    #: run asked for it rather than as the engine defaulted to it.
    #:
    #: It is here because it decides *what text a grader read*, which no other
    #: field records: a completion detokenised with special tokens skipped and
    #: the same completion detokenised without them are two different strings,
    #: and under the E42-era tokenizer the difference included every `<think>`
    #: block. There is no flag that moves it, so two runs of this instrument
    #: cannot differ on it -- what it is for is the reader who has to establish,
    #: from the record alone, which decode produced a number.
    #:
    #: `None` on a summary written before it was recorded, which is every run up
    #: to and including the E42-era ones. That is absence, not `True`: those runs
    #: took whatever the bound engine defaulted to, and writing the default in
    #: here would restate a default nobody chose as a decision somebody made.
    skip_special_tokens: bool | None = None


class SpeculativeDecoding(BaseModel):
    """The exact speculative decoder whose tokens contributed to completions."""

    method: Literal["qwen3_5_mtp"]
    num_speculative_tokens: int = Field(gt=0)

    def label(self) -> str:
        """Compact stable identity used in comparability diagnostics."""
        return f"{self.method}:{self.num_speculative_tokens}"


class RunManifest(BaseModel):
    """Everything needed to decide whether this run may be compared with another."""

    run_id: str
    created_at: str
    checkpoint: str
    stage: Stage
    profile: Profile
    suite: str
    format_policy: FormatPolicy = "strict"
    #: The public vLLM version the run declares before serving. `0.26.0` is the
    #: legacy default because every run written before this field existed was
    #: held to that repository-wide pin. The live engine's full reported value
    #: remains in `fingerprint.vllm_version` as runtime evidence.
    expected_vllm_version: str = "0.26.0"
    #: vLLM's eager mode skips CUDA graph and torch.compile capture. It is a
    #: serving-mode decision rather than part of the model fingerprint: the
    #: engine can still serve the declared architecture and weights, while
    #: eager execution may change the generated measurement. Missing on older
    #: manifests means the historical default, which was not eager.
    enforce_eager: bool = False
    #: Number of GPUs used to shard each engine replica. Missing on historical
    #: manifests means the former one-GPU-per-replica serving contract.
    tensor_parallel_size: int = Field(default=1, gt=0)
    exemplars_sha256: str | None = None
    #: Which anchor each taskset's relaxations actually relaxed, joined
    #: deterministically (`aime24=reference,math500=exact,…`) for the same reason
    #: `Pins.dataset_revision` is: one field, several tasksets, and string
    #: equality is exactly what `comparable_with` needs. Empty on a summary
    #: written before protocols existed, which is why it is not required.
    relaxation_bases: str = ""
    #: Defaulted rather than required so a summary written before the policy
    #: existed still loads, and loads as what it was: every such run drew avg@k.
    sampling_policy: SamplingPolicy = "avg-k"
    #: Defaults to ``reference`` so manifests written before native-max keep the
    #: meaning they had when their tasksets selected the completion budget.
    generation_budget_policy: GenerationBudgetPolicy = "reference"
    #: Defaulted to `fail` for the same reason and with the opposite sign to the
    #: current default: a summary written before this field existed was produced
    #: under the gate, so loading it as `score` would restate its numbers as
    #: something they are not. New runs default to `score` at the CLI, which is
    #: where a default belongs -- not here, where it would rewrite history.
    truncation_policy: TruncationPolicy = "fail"
    #: Which k each taskset reported its headline at, joined deterministically
    #: (`aime24=1+4+32,math500=1+4+32`) for the reason `relaxation_bases` is:
    #: one field, several tasksets, and string equality is what
    #: `comparable_with` needs. The k of one taskset are joined with `+` because
    #: `,` already separates the tasksets.
    #:
    #: Stamped by `aggregate.summarize` off the metric sets it actually
    #: computed, never accepted from a caller -- the same rule `scorer_version`
    #: follows, and for the same reason: a field naming which columns exist
    #: cannot be told by the caller which columns exist.
    #:
    #: Empty where no taskset in the run declared a k set at all, which is every
    #: summary written before D-06 and every run over constraint-checked
    #: tasksets alone. Those two are deliberately one state rather than two:
    #: neither reports a k column, so neither is measuring something the other
    #: is not.
    #:
    #: That empty state is **not comparable-with-nothing**, which is where this
    #: field parts company with `relaxation_bases`: it adds columns rather than
    #: changing what a shared one means, so a run that declared none still has
    #: every other column readable beside a run that declared some.
    #: `comparable_with` says why.
    pass_at_k: str = ""
    #: Which scoring semantics produced these numbers, stamped by
    #: `aggregate.summarize` rather than by a caller. `None` on a summary written
    #: before the stamp existed, which is the same thing as "unknown scorer": no
    #: stored classification from such a summary may be trusted, and
    #: `aggregate.failure_case` reclassifies instead of reading one.
    scorer_version: str | None = None
    #: The explicit RoPE scaling this run was served under, as `yarn:2`, or
    #: `None` for the native context every other run uses.
    #:
    #: A scaled run serves a basis the checkpoint's authors never published, so
    #: the artifact digest beside it describes weights that were nonetheless
    #: asked a different question. That is why this is a manifest field rather
    #: than a note: `comparable_with` refuses to read a scaled number beside an
    #: unscaled one, including against the same checkpoint at its own context.
    rope_scaling: str | None = None
    speculative_decoding: SpeculativeDecoding | None = None
    compilation_config: Literal["default", "mode0-full-decode-only"] = "default"
    sampling: Sampling
    pins: Pins
    fingerprint: Fingerprint

    @computed_field  # type: ignore[prop-decorator]
    @property
    def instrument_era(self) -> InstrumentEra:
        """Which instrument produced this run, derived rather than declared.

        A computed field, so it is written into every summary and can never be
        set from outside: an era read back from a file somebody edited is worth
        nothing, and the whole point of the marker is that a summary cannot
        misreport which text its graders read.

        **Two witnesses, either of which is sufficient.** The fingerprint's
        tokenizer and terminator are the primary one and the one the correction
        was about. `Sampling.skip_special_tokens` is the second, and it is what
        keeps the rule from misfiling a *reference baseline*: a reference
        checkpoint is served its own tokenizer and pins no terminator, so it
        records no vocabulary at all while being every bit as corrected-era as a
        Limite run beside it. Quarantining those runs would refuse to rescore
        completions whose delimiters were never at risk. The field arrived with
        the same correction and is written on every run since, whatever was
        served, so the pair together is exact in both directions.
        """
        if self.fingerprint.records_vocabulary or self.sampling.skip_special_tokens is not None:
            return "corrected"
        return "e42"

    def assert_comparable_with(self, other: RunManifest) -> None:
        """Refuse outright where the two runs sit on opposite sides of E42.

        The hard gate beside `comparable_with`'s advisory list, and the two are
        deliberately different in kind. Every other reason that list returns is a
        difference in *how the same text was scored*, which a reader may weigh --
        a wider budget, another sampling policy, a moved relaxation base. This
        one is a difference in **what text there was**: the E42-era completions
        had every `<think>` and `</think>` deleted before anything read them, so
        a case count, a `format_ok` and a strict number from either side are
        answers to different questions. There is no weighing to do, so this
        raises rather than reporting.

        Callers comparing case counts or any think-dependent metric across two
        runs must pass through here first.
        """
        if self.instrument_era != other.instrument_era:
            raise InstrumentEraMismatch(
                f"{self.run_id} was scored by the {self.instrument_era} instrument and "
                f"{other.run_id} by the {other.instrument_era} one: the E42-era runs had every "
                "`<think>` and `</think>` deleted between the engine and the grader, so their "
                "completions are not the text the model produced. Case counts, `format_ok`, "
                "the strict lens and every relaxation that waives an unclosed think block "
                "answer different questions on the two sides of that boundary and may not be "
                "compared, aggregated or differenced. Regenerate the E42-era side; no rescore "
                "can recover delimiters that were destroyed at decode time."
            )

    def comparable_with(self, other: RunManifest) -> list[str]:
        """Reasons these two runs are not comparable; empty means they are.

        The rendering profile is deliberately **not** a blocker: comparing a
        base-kshot run against a chat run is the whole point of the repository.
        What must match is everything that would change the meaning of the
        metric rather than the thing being measured.

        The generation budget is one of those things. A completion that runs out
        of budget scores wrong, so the same checkpoint on the same problems
        scores lower at a smaller budget; two runs that differ there are
        measuring different quantities. The check is coarse -- `max_tokens`
        holds the largest budget in the run -- so it catches a whole-run change
        and can miss a single taskset's.

        A change of **relaxation base** is likewise a difference, and D-108 makes
        it one explicitly. `lenient(exact)` and `lenient(reference)` forgive the
        same six things on top of two different graders, so they are not one
        measurement wearing one name; two runs whose `lenient` relaxed different
        anchors are reporting different quantities in the same column, and the
        manifest is where a reader finds that out rather than by reading the
        scoring code at both revisions.

        **Truncation policy** is the fourth, and it is the one that can move a
        column furthest. Under `score` a completion that ran out of budget is
        graded on the text it produced; under `fail` it is wrong whatever that
        text says. On a suite where two rollouts in five run out of context --
        which the 65k math-extended runs did -- the same completions read tens of
        points apart across the two, so a run quoting one while having computed
        the other is what this field exists to refuse.

        **Sampling policy** is the third. avg@k over 32 rollouts at temperature
        0.6 and a single greedy rollout are not the same number with different
        error bars -- they are the mean of a distribution and one draw from its
        mode, and on a thirty-problem competition set the two can differ by more
        than two checkpoints do. The policy names that avg@k-versus-greedy
        distinction and its group-size semantics. Temperature and top-p also
        gate independently because avg@k permits an explicit temperature
        override: two runs drawn from different distributions are different
        measurements even when they share the same policy name.

        **Speculative decoding and compilation configuration** describe the
        generation path used by the serving engine. MTP changes which tokens
        are proposed before verification, while compilation and CUDA-graph
        modes change how decode is executed. They are recorded and gated so a
        performance optimization is never silently folded into a comparison
        with a different execution path.

        **The expected vLLM version** is the declared engine contract rather
        than a runtime observation. Its public version gates comparisons and
        frozen resume identity; the fingerprint separately records the exact
        version the bound engine reported.

        **The k set is here because absence has two causes**, and it is the one
        entry that gates on two declarations rather than on a difference. A run
        that declared `(1, 4)` and a run that declared `(1, 4, 32)` both report
        no `pass@32` on a taskset that drew four rollouts, and only the manifest
        says which of them was asked -- so where both declared and the
        declarations differ, they are asking different questions of the same
        rollouts and the columns must not be read across.

        **An unrecorded k set is the absence of a declaration, not a different
        one**, and this is deliberately the opposite of the rule above it for
        the relaxation base. The two fields are not the same kind of fact. The
        base decides what a **shared** column means: `lenient` relaxed against a
        published grader and `lenient` relaxed against prime-rl's verifier are
        two quantities under one heading, so a run that never recorded a base
        has a `lenient` nobody can read and is comparable with none. The k set
        adds columns and changes no shared one: `exact`, `reference` and every
        rung mean exactly what they meant, whether or not `pass@4` was also
        computed. Refusing on it would put every run written before D-06 out of
        reach of every run after it -- which is the pre-to-post comparison this
        repository exists to make -- for a reason that describes none of the
        numbers on either side. And it would be a reason naming no pair: where
        one side has no k column at all, there is nothing there to quote across.

        **RoPE scaling is the one entry that is not about the measurement but
        about the model.** Every other reason here separates two ways of asking
        the same checkpoint a question; this one separates two checkpoints. A
        model served under YaRN at twice its trained context has a different
        frequency basis in every attention layer, so the artifact digest the two
        runs share is describing weights that were evaluated as different
        functions. `native` on both sides is the ordinary case and blocks
        nothing.

        **The instrument era is listed first and is also a refusal.** It is the
        one reason here that is not advice: `assert_comparable_with` raises on
        it, and it appears in this list too so that a caller reading the reasons
        rather than calling the gate still sees the boundary named.
        """
        reasons = []
        if self.instrument_era != other.instrument_era:
            reasons.append(
                f"different instrument era: {self.instrument_era} vs {other.instrument_era} "
                "(E42 deleted every reasoning delimiter before the grader read it)"
            )
        if self.sampling_policy != other.sampling_policy:
            reasons.append(
                f"different sampling policy: {self.sampling_policy} vs {other.sampling_policy}"
            )
        if self.sampling.temperature != other.sampling.temperature:
            reasons.append(
                f"different temperature: {self.sampling.temperature} vs {other.sampling.temperature}"
            )
        if self.sampling.top_p != other.sampling.top_p:
            reasons.append(f"different top-p: {self.sampling.top_p} vs {other.sampling.top_p}")
        for field, label in (
            ("top_k", "top-k"),
            ("min_p", "min-p"),
            ("presence_penalty", "presence penalty"),
            ("repetition_penalty", "repetition penalty"),
        ):
            mine, theirs = getattr(self.sampling, field), getattr(other.sampling, field)
            if mine != theirs:
                reasons.append(
                    f"different {label}: "
                    f"{'none' if mine is None else mine} vs "
                    f"{'none' if theirs is None else theirs}"
                )
        if self.speculative_decoding != other.speculative_decoding:
            mine = "none" if self.speculative_decoding is None else self.speculative_decoding.label()
            theirs = "none" if other.speculative_decoding is None else other.speculative_decoding.label()
            reasons.append(f"different speculative decoding: {mine} vs {theirs}")
        if self.compilation_config != other.compilation_config:
            reasons.append(
                f"different compilation config: {self.compilation_config} vs "
                f"{other.compilation_config}"
            )
        mine_vllm = self.expected_vllm_version.partition("+")[0]
        their_vllm = other.expected_vllm_version.partition("+")[0]
        if mine_vllm != their_vllm:
            reasons.append(f"different expected vLLM version: {mine_vllm} vs {their_vllm}")
        if self.generation_budget_policy != other.generation_budget_policy:
            reasons.append(
                "different generation budget policy: "
                f"{self.generation_budget_policy} vs {other.generation_budget_policy}"
            )
        if self.truncation_policy != other.truncation_policy:
            reasons.append(
                f"different truncation policy: {self.truncation_policy} vs "
                f"{other.truncation_policy}"
            )
        if self.pass_at_k and other.pass_at_k and self.pass_at_k != other.pass_at_k:
            reasons.append(f"different k set: {self.pass_at_k} vs {other.pass_at_k}")
        if self.relaxation_bases != other.relaxation_bases:
            reasons.append(
                f"different relaxation base: {self.relaxation_bases or 'unrecorded'} vs "
                f"{other.relaxation_bases or 'unrecorded'}"
            )
        if self.suite != other.suite:
            reasons.append(f"different suite: {self.suite} vs {other.suite}")
        if self.sampling.max_tokens != other.sampling.max_tokens:
            reasons.append(
                f"different generation budget: {self.sampling.max_tokens} vs "
                f"{other.sampling.max_tokens}"
            )
        if self.rope_scaling != other.rope_scaling:
            reasons.append(
                f"different RoPE scaling: {self.rope_scaling or 'native'} vs "
                f"{other.rope_scaling or 'native'}"
            )
        if self.format_policy != other.format_policy:
            reasons.append(f"different format policy: {self.format_policy} vs {other.format_policy}")
        if self.enforce_eager != other.enforce_eager:
            reasons.append(
                f"different vLLM eager mode: {self.enforce_eager} vs {other.enforce_eager}"
            )
        if self.tensor_parallel_size != other.tensor_parallel_size:
            reasons.append(
                "different tensor parallel size: "
                f"{self.tensor_parallel_size} vs {other.tensor_parallel_size}"
            )
        for field in type(self.pins).model_fields:
            mine, theirs = getattr(self.pins, field), getattr(other.pins, field)
            if mine != theirs:
                if field in {"task_prompt_revision", "native_scorer_revision", "answer_scorer_revision"}:
                    label = field.replace("_", " ")
                    reasons.append(
                        f"different {label} pin: "
                        f"{mine or 'unrecorded'} vs {theirs or 'unrecorded'}"
                    )
                else:
                    reasons.append(f"different {field} pin: {mine} vs {theirs}")
        return reasons


class SampleResult(BaseModel):
    """One rollout of one problem, with everything its taskset scores recorded.

    The ladder's four booleans stay declared fields rather than moving into
    `metrics`: the runner writes them directly, every sample file already
    written carries them there, and they are the shape the boxed and hashed
    verifiers produce. They default to False so a taskset scored by something
    other than the ladder -- a constraint checker, which has no boxed answer to
    extract -- does not have to write four meaningless falses on every row.

    `comparison_timeout` sits beside `truncated` because the two are the same
    kind of fact: neither says the model was wrong, both mean a number computed
    from this row is a lower bound. `truncated` says the completion never
    finished; `comparison_timeout` says the verifier never finished. It is
    False for a constraint-checked row for the same reason the rungs are --
    nothing compared an answer there, so nothing could time out.
    """

    task: str
    problem_index: int
    rollout_index: int
    #: Stable identities from a benchmark whose statistical unit spans several
    #: dataset rows. They are optional so every row written before native
    #: tasksets existed, and every ordinary one-row problem, loads unchanged.
    source_group_id: str | None = None
    source_variant_id: str | None = None
    gold: str
    completion: str
    finish_reason: str | None = None
    #: vLLM's own field beside `finish_reason`, left exactly as the engine sent
    #: it: the matched stop string, a stop token id, or absent when the model
    #: emitted its learned terminator. `finish_reason` reads `"stop"` for all
    #: three, so this is the only thing that separates a checkpoint halted from
    #: outside from one that ended by itself.
    stop_reason: str | int | None = None
    #: The digest of the rendering this rollout was produced under -- the served
    #: template bytes and their stop strings. Per sample because one run serves
    #: several renderings, and a rollout whose rendering is not recoverable from
    #: its own row is a rollout that cannot be regrouped after the fact.
    template_sha256: str | None = None
    truncated: bool = False
    #: Every registered protocol's verdict on this completion, keyed by protocol
    #: id. This is what the run reports; the four ladder booleans below are kept
    #: beside it for continuity with rows already written, and `metrics` is what
    #: aggregation reads.
    protocols: dict[str, ProtocolSample] = Field(default_factory=dict)
    comparison_timeout: bool = False
    strict: bool = False
    lenient: bool = False
    permissive: bool = False
    format_ok: bool = False
    #: Whether this rollout has no answer region for a format lens to read: a
    #: think block that was opened, never closed, and ran into the generation
    #: cap. `format_ok` is False beside it for continuity with rows already
    #: written, and aggregation drops the row from that metric's denominator
    #: rather than counting a formatting failure the model never got to commit.
    #:
    #: On a constraint-checked row the same fact excludes the two accuracies
    #: instead: word counts, casing and comma rules asked of a pure reasoning
    #: fragment are not a measurement of instruction following. Which metrics it
    #: reaches is the taskset's declaration, `MetricSet.requires_answer_region`,
    #: never a name matched here.
    #:
    #: False on every row written before it existed, which is correct for all of
    #: them: under the E42-era decode no completion carried a delimiter, so no
    #: row could have had an unclosed block to detect.
    format_not_evaluable: bool = False
    #: Which scoring path wrote the verdicts and the case letter on this row.
    #: `None` on a row written before the stamp existed. `aggregate.failure_case`
    #: reads it before trusting `failure_case`, so a classification cached by an
    #: older scorer is recomputed rather than believed.
    scorer_version: str | None = None
    #: What this rollout scored at the metrics its taskset declares beyond the
    #: ladder, keyed by metric name. Floats rather than booleans because a
    #: constraint checker reports a fraction of instructions followed, not a
    #: verdict.
    metrics: dict[str, float] = Field(default_factory=dict)
    strict_answer: str | None = None
    lenient_answer: str | None = None
    completion_tokens: int | None = None
    #: Which of the twenty A--T failure-mode cases this completion's shape falls
    #: in, from `limite_evals.taxonomy`. `None` on a row written before the
    #: tracker existed and on a constraint-checked row, which has no answer to
    #: extract and therefore no shape to describe -- the same absence its empty
    #: `protocols` already records. Aggregation classifies a row that carries
    #: none, so a stored `None` costs a reader nothing but the sample file's own
    #: itemisability.
    failure_case: str | None = None


class Interval(BaseModel):
    """A point estimate with a confidence interval. Both bounds always present."""

    value: float
    low: float
    high: float
    confidence: float = 0.95


class NativeMetric(BaseModel):
    """One scalar emitted directly by a taskset's benchmark-native scorer.

    Unlike protocol metrics, a native scalar may live outside the unit interval.
    Its range and presentation therefore travel with the declaration instead of
    being inferred from its name. `provenance` names the scoring implementation
    or formula and must disclose any local composition. An unavailable metric remains declared with its
    reason and is omitted from sample values; absence is never converted to zero.

    Source-group aggregations are for benchmarks such as ABench Phy_B, where
    several dataset rows are fixed variants of one source problem. The declared
    variants are selected once, collapsed to one scalar, and only the outer
    source groups are resampled.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    minimum: float
    maximum: float
    presentation: MetricPresentation
    provenance: str
    unavailable_reason: str | None = None
    aggregation: NativeAggregation = "sample_mean"
    source_variants: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_contract(self) -> NativeMetric:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("a native metric range must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("a native metric maximum must be greater than its minimum")
        if not self.provenance.strip():
            raise ValueError("a native metric must name its scorer provenance")
        if self.aggregation == "sample_mean" and self.source_variants:
            raise ValueError("sample_mean cannot declare source variants")
        if self.aggregation != "sample_mean" and not self.source_variants:
            raise ValueError("a source-group metric must declare its complete variant set")
        if len(set(self.source_variants)) != len(self.source_variants):
            raise ValueError("a native metric cannot declare duplicate source variants")
        return self


class MetricSet(BaseModel):
    """What a taskset reports, and what a report may say about it.

    `metrics` is ordered and the first entry is the headline. That was already
    the convention when the rungs were a bare tuple -- decision 3 headlines
    `strict` and lists the rest beside it -- so it is stated here rather than
    duplicated as a second field that could contradict the order.

    `correctness` names the metrics a truncated completion cannot pass
    (decision 10). A metric that describes what the completion *looked like*
    rather than whether it was right is deliberately left out: forcing it would
    erase the fact being reported.

    `row_label` and `note` are here rather than in the report because the report
    is written for a reader with no local files open, and neither sentence is
    true of every metric set. "Rung" is the ladder's vocabulary; a constraint
    checker has no rungs, and a paragraph about the strict-to-lenient gap says
    nothing about one.

    **The comparison-timeout rate is not a metric and does not belong here.**
    Every name a metric set may hold describes the completion -- whether it was
    right, whether it was formatted, whether it ran out of context. A comparison
    timeout describes the *verifier*: the same rollout on a faster machine, or
    under a different `math-verify` pin, may not produce one. Declaring it here
    would make it a peer of `strict` when what it actually does is qualify
    `strict` -- it says the rung is a lower bound, which is a statement about
    the set rather than a member of it. Two consequences follow and both are
    load-bearing. A metric set is per taskset, so a constraint-checked taskset
    would have to declare a metric its scoring path cannot produce or leave a
    hole a reader would read as "no timeouts"; and every metric here is
    rendered as a table row, a plotted bar and a clause in `note`, so adding one
    would re-render every `math-standard` report already quoted. It lives on
    `TaskSummary` instead, as a plain rate beside the numbers it qualifies.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    metrics: tuple[str, ...]
    correctness: tuple[str, ...] = ()
    #: The metrics that need an answer region to exist before they mean
    #: anything, and are therefore **not evaluable** on a rollout whose think
    #: block ran into the token cap (`SampleResult.format_not_evaluable`). Such a
    #: rollout leaves this taskset's denominator for these metrics only; it is
    #: never dropped from the taskset, and every other column still counts it.
    #:
    #: Declared per metric set rather than matched by name, because which metrics
    #: those are is a property of what the taskset measures. For the ladder it is
    #: `format_ok` alone: the correctness rungs are entitled to say a completion
    #: that never presented an answer did not get one right. For a constraint
    #: checker it is both accuracies, since "no commas" and "under 40 words" are
    #: satisfied by accident by an abandoned reasoning fragment.
    requires_answer_region: tuple[str, ...] = ()
    #: The group sizes this set's **headline** is additionally reported at, as
    #: `pass@k` and `pass^k`. Empty for a set that reports neither, which is
    #: every metric set written before D-06 and every constraint-checked one:
    #: pass@k needs a per-rollout verdict, and a fraction of instructions
    #: followed is not one.
    #:
    #: A declaration rather than a flag, for the reason every other field here
    #: is one. The k set decides what the columns are, so two runs that declared
    #: different sets measured different things, and `RunManifest.pass_at_k`
    #: carries it into `comparable_with` rather than leaving a reader to infer
    #: it from which columns happen to be present -- an absent `pass@32` is
    #: otherwise indistinguishable from a taskset that drew too few rollouts for
    #: one (D-107).
    #:
    #: It reaches the headline alone and not all four correctness protocols.
    #: That is D-06: the rollouts stay archived, so any other protocol's k
    #: columns are recoverable later, and a table carrying five k columns for
    #: each of four protocols is a table nobody reads.
    pass_at_k: tuple[int, ...] = ()
    #: Definitions for metrics produced directly by a benchmark scorer rather
    #: than by the protocol or constraint lanes. Empty preserves the complete
    #: legacy contract. Names must also appear in `metrics`, whose order remains
    #: the report order and headline declaration.
    native_metrics: tuple[NativeMetric, ...] = ()
    row_label: str = "Metric"
    note: str = ""

    @model_validator(mode="after")
    def _native_names_are_declared_once(self) -> MetricSet:
        names = [metric.name for metric in self.native_metrics]
        if len(set(names)) != len(names):
            raise ValueError("a metric set cannot define one native metric twice")
        undeclared = sorted(set(names) - set(self.metrics))
        if undeclared:
            raise ValueError(f"native metrics are not declared in metrics: {', '.join(undeclared)}")
        return self

    @property
    def headline(self) -> str:
        """The number the report leads with: the first metric declared."""
        return self.metrics[0]

    def native(self, name: str) -> NativeMetric | None:
        """The native declaration for `name`, or none for legacy/protocol metrics."""
        return next((metric for metric in self.native_metrics if metric.name == name), None)


#: The extraction ladder, as a declaration. Every suite in the repository today
#: reports exactly this, which is why it is the default everywhere a metric set
#: can be left unstated.
LADDER = MetricSet(
    name="ladder",
    metrics=("strict", "lenient", "permissive", "format_ok", "truncated"),
    correctness=("strict", "lenient", "permissive"),
    requires_answer_region=("format_ok",),
    row_label="Rung",
    note=(
        "`strict` is the headline at every stage; `lenient`, `permissive`, `format_ok` and "
        "`truncated` are reported beside it. Intervals are a hierarchical bootstrap that "
        "resamples problems and then that problem's rollouts. A truncated completion is "
        "counted in the truncation rate and is never dropped; what it scores is the run's "
        "truncation policy, stated with the comparability facts below. `format_ok` asks "
        "whether an answer sat where the reference wanted it, so a rollout whose think block "
        "ran into the token cap has no answer region for it to read: those leave that one "
        "denominator and are reported as `format_not_evaluable` beneath the table."
    ),
)


#: What a run reports now that a protocol, rather than a rung, is the thing a
#: number is scored under.
#:
#: **`exact` is the headline wherever it exists**, because it is the only column
#: comparable with a public leaderboard: it reproduces the benchmark's own
#: published grader and carries no local scoring logic at all. Where no published
#: grader could be bound the column is *absent with its reason* (D-107) and the
#: report says so; nothing is promoted into the headline in its place, since a
#: table whose leading column silently changed meaning between rows is the one
#: failure this whole metric set exists to prevent.
#:
#: `reference` sits beside it rather than beneath it. The two anchors name
#: different artefacts -- the benchmark's published protocol and the verifier
#: prime-rl actually executes -- so neither contains the other and neither is
#: derivable from the other. `reference` is what makes an offline number
#: comparable with the in-run reinforcement-learning signal, which is a
#: comparison this repository already relies on.
#: The k a protocol-scored taskset reports its headline at, beside avg@k.
#:
#: One entry per group size any taskset in this repository declares -- 32 for
#: the competition sets, 4 for MATH-500, 1 for the single-draw ones -- because a
#: k above a taskset's group size is not a low score but an unasked question,
#: and the column is absent with that reason rather than reported. `1` is in the
#: set although it duplicates avg@k exactly: it is the same estimator evaluated
#: at the k a published single-sample number is quoted at, and printing it beside
#: `pass@32` is what makes the distance between one draw and thirty-two legible
#: without a reader holding two tables side by side.
PASS_AT_K = (1, 4, 32)


PROTOCOL_METRICS = MetricSet(
    name="protocols",
    metrics=("exact", "reference", "lenient", "permissive", "format_ok", "truncated"),
    correctness=("exact", "reference", "lenient", "permissive"),
    requires_answer_region=("format_ok",),
    pass_at_k=PASS_AT_K,
    row_label="Protocol",
    note=(
        "Each column names the artefact that produced it. `exact` reproduces the benchmark's "
        "own published grader and is the headline wherever one exists; where none does the "
        "column is absent with its reason and nothing is promoted in its place. `reference` "
        "reproduces the verifier prime-rl executes, which is what makes a number comparable "
        "with the in-run signal. `lenient` and `permissive` are named relaxations, and each "
        "is shown with the anchor it relaxed -- that base is `exact` on a taskset with a "
        "published grader and `reference` on one without, so the two are not interchangeable "
        "quantities. Intervals are a hierarchical bootstrap that resamples problems and then "
        "that problem's rollouts. A truncated completion is counted in the truncation rate "
        "and is never dropped; what it scores at every correctness protocol is the run's "
        "truncation policy, stated with the comparability facts below. `format_ok` asks "
        "whether an answer sat where the reference wanted it, so a rollout whose think block "
        "ran into the token cap has no answer region for it to read: those leave that one "
        "denominator and are reported as `format_not_evaluable` beneath the table."
    ),
)


class CaseCount(BaseModel):
    """One A--T failure-mode case on one taskset: how many, and what they scored.

    `strict_scored` and `permissive_scored` are what the taskset scored under
    those two names on the rollouts in this case, counted after the run's
    truncation policy has been applied -- they are observations, not the
    taxonomy's prediction. `strict_expected` and `permissive_expected` are the
    prediction, copied off the frozen table so a summary read later states what
    was claimed without a lookup into the code that produced it.

    Holding both is what makes the row a reconciliation rather than a tally: a
    case with `strict_expected` false and a non-zero `strict_scored` is a
    rollout the taxonomy said the strict lens could not accept and the strict
    lens accepted, which is the finding this table exists to surface.

    **`strict_expected` is `None` where the taxonomy predicts nothing**, which
    is every row whose `answer_format` is not `boxed`. Each case in the table
    describes where a `\\boxed{}` answer sits, so a historical strict lens that
    reads something else -- such as GSM8K's retired `#### 42` contract -- is outside what the column was
    written about. `None` is not `False`: there is no claim there to violate,
    and a `False` would report every rollout that lens accepted as a
    falsification of a prediction nobody made. It is the same distinction
    `ProtocolSample.correct` draws for a grader that could not be bound.
    `permissive_expected` is never `None`, because "was the answer reachable
    from anywhere in the text" does not depend on the notation the reference
    asked for.

    A count rather than an `Interval`: these are events in this run, not a rate
    a resample could estimate, for the same reason `stop_reasons` is counted.
    """

    model_config = ConfigDict(frozen=True)

    case: str
    description: str
    samples: int
    strict_expected: bool | None
    permissive_expected: bool
    strict_scored: int
    permissive_scored: int


class TaskSummary(BaseModel):
    """One taskset's numbers, one field per metric its metric set declares.

    The metrics are the model's *extra* fields rather than a nested map, and
    that is load-bearing in both directions: a summary written before metric
    sets existed carries its five rungs at the top level of each task and still
    loads, and anything already reading `task["strict"]` straight off the JSON
    keeps working. Typing the extras as `Interval` is what keeps "allow extras"
    from meaning "accept anything" -- a stray key that is not an interval is
    still a validation error.

    `metric_set` travels with the numbers, repeated per taskset, for the same
    reason the manifest travels with the run: a summary read months later has to
    say what its own numbers are without a second file or a lookup table that
    may since have moved. A metric may not be named after one of this model's
    own fields.

    `comparison_timeout_rate` is a declared field and deliberately **not** a
    metric. See `MetricSet` for why it is not one of them; here the mechanical
    consequence is what matters: the extras are typed `Interval`, and this is a
    bare fraction with no interval to put on it. Defaulted to zero so a summary
    written before it existed loads as what it was, a run with no timeouts.
    """

    __pydantic_extra__: dict[str, Interval] = Field(init=False)

    model_config = ConfigDict(extra="allow")

    task: str
    problems: int
    group_size: int
    metric_set: MetricSet = LADDER
    #: Which reference notation this taskset's strict lens reads, from its spec.
    #: Recorded because it decides whether the A--T table's strict column says
    #: anything here at all, and a summary has to be able to answer that on its
    #: own -- the alternative is a reader inferring the scope of a claim from a
    #: taskset name. `None` for a constraint-checked taskset, which has no
    #: answer to extract, and on a summary written before the field existed.
    answer_format: str | None = None
    #: What each protocol meant for this taskset, in registry order. Carries the
    #: anchors, the waiver lists, the relaxation bases and the reason for any
    #: protocol that bound no artefact -- so a column absent from `metrics` is
    #: explained here rather than merely missing.
    protocols: list[ProtocolRecord] = Field(default_factory=list)
    #: How this taskset's rollouts ended, counted by `StopClass`. Reported beside
    #: the truncation rate because the stop-string/end-of-sequence split is the
    #: evidence that a pretraining checkpoint and a post-trained one are being
    #: scored without a per-stage scoring switch.
    stop_reasons: dict[str, int] = Field(default_factory=dict)
    #: This taskset's rollouts partitioned by A--T failure-mode case, in the
    #: frozen table's own order and with the cases nothing fell in omitted.
    #: Empty for a taskset with no answer to extract, and on a summary written
    #: before the tracker existed. Beside the metrics rather than among them:
    #: every entry is a count of rollouts, and none of them is a number the
    #: report leads with or a resample could estimate.
    failure_cases: list[CaseCount] = Field(default_factory=list)
    #: Fraction of this taskset's rollouts whose answer comparison hit its time
    #: bound and therefore scored wrong. Over rollouts rather than the ladder's
    #: mean-over-problems-of-means: it counts events in this run on this machine,
    #: not a rate a resample could estimate.
    comparison_timeout_rate: float = 0.0
    #: Fraction of this taskset's rollouts that left the denominator of the
    #: metrics in `MetricSet.requires_answer_region` -- a think block opened,
    #: never closed, and ended by the token cap.
    #:
    #: A declared field beside the numbers rather than a metric among them, for
    #: the reason `comparison_timeout_rate` is: it does not describe whether a
    #: completion was right or how it was shaped, it says how much of the column
    #: above it was computed over. It is a flat rate over rollouts with no
    #: interval, because it counts events in this run rather than estimating a
    #: quantity a resample could move. Defaulted to zero so a summary written
    #: before it existed loads as what it was -- an era in which no completion
    #: carried a delimiter, so none of them could have had an unclosed block.
    format_not_evaluable_rate: float = 0.0

    @property
    def metrics(self) -> dict[str, Interval]:
        """The declared metrics in declaration order, headline first.

        A metric the taskset could not produce is **absent** rather than zero,
        so a protocol whose artefact could not be bound has no entry here at
        all and `protocols` carries the reason.
        """
        return self.__pydantic_extra__ or {}

    def protocol(self, name: str) -> ProtocolRecord | None:
        """This taskset's record for one protocol, or None if it declared none."""
        return next((record for record in self.protocols if record.protocol == name), None)

    def protocol_source(self, name: str) -> str | None:
        """The executed source behind an anchor or a relaxation's recorded base."""
        record = self.protocol(name)
        if record is not None and record.relaxes is not None:
            record = self.protocol(record.relaxes)
        return record.anchor if record is not None else None


class RunSummary(BaseModel):
    manifest: RunManifest
    tasks: list[TaskSummary]
