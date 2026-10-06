"""The rollout runner: one chat completion per rollout, scored by the ladder.

**A deliberate departure, and the reason for it.** The tasks in `tasksets.py` are
defined in the `verifiers` v1 shape, but they are driven from here rather than
through `verifiers`' own environment and harness stack. That stack is built for
agentic rollouts: its `null` harness -- the single-turn one -- launches a `uv run`
subprocess per rollout to make one chat completion. AIME24 at thirty problems and
a group size of thirty-two is 960 subprocesses to send 960 chat requests, plus an
interception server and a runtime in front of each. For a single-turn math eval
that machinery buys nothing and costs a great deal of wall-clock and failure
surface.

What the `verifiers` pin still buys, and why it is not wasted: the task and
reward definitions stay loadable by `vf eval` for cross-checking, the reward and
metric structure is the framework's own rather than an invention, and the
tasksets we may add later that genuinely need a sandbox -- code execution --
already have somewhere to go.

The scoring path is identical either way. `LimiteTask` scores through
`protocols`, and this runner asks the same registry for the same verdicts the
framework would.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Set

import httpx

from limite_evals import taxonomy
from limite_evals.profiles import Profile
from limite_evals.serve import ReasoningParserRefused
from limite_evals.suites import TasksetSpec
from limite_evals.templates import Rendering
from limite_evals_core.ladder import format_not_evaluable, score
from limite_evals_core.protocols import ProtocolResult, TasksetProtocols
from limite_evals_core.protocols import score as score_protocols
from limite_evals_core.schema import SCORER_VERSION, ProtocolSample, SampleResult

#: Concurrent in-flight requests per vLLM data-parallel replica. The CLI scales
#: this by its replica count; direct single-replica callers use it as-is.
DEFAULT_CONCURRENCY_PER_REPLICA = 320
DEFAULT_CONCURRENCY = DEFAULT_CONCURRENCY_PER_REPLICA
DEFAULT_REQUEST_TIMEOUT = 1800.0


def _client_timeout(timeout: float) -> httpx.Timeout:
    """Bound transport setup while allowing an accepted generation to finish."""

    return httpx.Timeout(timeout, read=None)

#: Transport faults that kill a request before the engine has answered it, and
#: which a resend can therefore legitimately paper over.
#:
#: **Why this exists.** Three long runs in one round died at 39m, 59m and 1h18m
#: with a single `httpx.ReadError` out of `generate`, after two of three tasksets
#: had already completed. The engine was healthy either side of it -- thousands
#: of tokens a second seconds before, no CUDA fault, no OOM, and the process only
#: went down because the client raised. The client's source ports jump to a fresh
#: range immediately before each failure, alongside `RuntimeWarning: coroutine
#: 'connect_tcp.<locals>.try_connect' was never awaited`, which puts the fault in
#: the connection pool: a pooled connection is recycled underneath an in-flight
#: request. It is time- and churn-dependent, which is why halving concurrency
#: did not help, and it costs a whole evaluation each time it fires.
#:
#: **Why a resend is sound here.** Every entry is a failure of the transport,
#: raised because no response was received -- not a refusal, not a bad request,
#: and not a partial read of a streamed body, since these completions are not
#: streamed. `HTTPStatusError` is deliberately absent: an engine that answers
#: with a 4xx or 5xx has been reached, and retrying it would loop on a real
#: defect. `ReadTimeout` is absent for the same reason in the other direction --
#: the connection is alive and the engine is simply still working, so a resend
#: would double the wall clock and leave the original generating.
RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)

#: Total attempts per rollout, the first one included. Bounded, and small: this
#: is a fix for a pool race that fires a handful of times in six thousand
#: requests, not a way to sit out an engine that has actually died. Four attempts
#: over the backoff below give up after about seven seconds.
TRANSPORT_ATTEMPTS = 4

#: Seconds before the first resend; doubled each attempt. Short, because the
#: connection the resend needs is a new one and nothing has to recover for it to
#: succeed -- the delay is there to let a pool in the middle of recycling finish.
TRANSPORT_BACKOFF = 1.0


class TransportRetries:
    """How many rollouts were resent, and after which fault.

    Kept as a count rather than folded into the samples because a resend is a
    property of the run, not of the rollout: the request that eventually
    succeeded is byte-identical to the one that failed, so there is no field on
    the row it belongs in. The count is written into the run's artefacts so that
    a run which needed resends is auditable as one -- a reader can see that the
    transport was flaky and how flaky, without that changing a single number.

    **What a resend is, scientifically.** The request body is reconstructed by
    nobody: the same `body` dict, with the same derived `seed`, temperature,
    `top_p`, `max_tokens` and stop configuration, goes back out. vLLM seeds each
    request's sampler from that `seed`, so the sampling decisions are reproducible
    given identical logits -- but continuous batching means the resend is very
    unlikely to be batched with the same neighbours, and the logits are therefore
    not guaranteed bitwise identical. So a resend is *not* claimed to be
    server-side deterministic. It is, at worst, a fresh draw from the same
    declared distribution over the same prompt, which is the property that
    matters: avg@k over a group is an estimate of that distribution, and
    redrawing one member of the group leaves the estimator unbiased. Nothing
    about the sample is resampled by the client -- no re-rendering, no
    re-seeding, no change of parameters.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def record(self, error: BaseException) -> None:
        name = type(error).__name__
        self._counts[name] = self._counts.get(name, 0) + 1

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "by_error": dict(sorted(self._counts.items())),
            "attempts_allowed": TRANSPORT_ATTEMPTS,
        }


#: How the engine detokenises a completion before anything scores it, sent on
#: every request rather than left to the server's default.
#:
#: It is a constant and not a flag. What text a grader reads is not an axis this
#: instrument offers, and a run that moved it would not be comparable with one
#: that did not; it is sent explicitly and recorded in the manifest so that the
#: value is a declared input rather than whatever the bound engine happens to
#: default to -- vLLM's default is `True` today, and a default that changed under
#: a version bump would change every reported number with nothing saying so.
#:
#: **Why `True` rather than `False`, given E42.** The value that keeps the
#: reasoning delimiters is not the one that keeps *every* special token. Under
#: the served vocabulary `<think>` and `</think>` are added tokens marked
#: non-special -- the upstream convention for reasoning delimiters, and now a
#: refusal in `vocabulary.assert_visible` rather than an assumption -- so they
#: survive this and reach the grader. What `False` would additionally put into
#: the graded text is the ChatML structure: `<|im_start|>` and `<|im_end|>`
#: emitted mid-completion, which downstream checks would treat as completion
#: content. That is a grader-visible change to every taskset, in
#: exchange for a property already guaranteed, so the flag stays where the engine
#: has it and the guarantee is enforced one level down, at the vocabulary.
#:
#: The completion's own terminator never appears in the text under either value:
#: vLLM excludes a stop-terminated token from what it detokenises.
SKIP_SPECIAL_TOKENS = True


def _assert_raw_text(message: dict) -> None:
    """Refuse a completion that reached here already taken apart.

    The backstop to `serve._assert_no_reasoning_parser`, and not a duplicate of
    it: that one holds the launch line this repository assembles, while this one
    holds the *response*, whoever started the engine it came from. An operator
    pointing `--base-url` at an engine somebody else configured, or a vLLM whose
    default changed under a version bump, passes the first and is caught here.

    `reasoning_content` is populated only when a parser ran, and when it did the
    `content` beside it has had its `<think>` block removed. Reading that content
    would put a completion with no reasoning delimiters through `strip_think`,
    the A--T tracker and the protocols -- all of which are written for text that
    still has them -- and every one of those numbers would still be reported. So
    the completion is refused rather than scored, and rather than repaired: see
    `ReasoningParserRefused`.
    """
    if not message.get("reasoning_content"):
        return
    raise ReasoningParserRefused(
        "the engine returned a completion with a non-empty message.reasoning_content, which means "
        "a reasoning parser is running in front of it. Completions must arrive as raw text in "
        "message.content, reasoning delimiters included; a parser moves the think block out of "
        "the content it returns, so strip_think, the A--T tracker and every think-aware "
        "measurement would read a completion that never had one -- silently, with numbers still "
        "reported (E42's failure by a third route). Reasoning parsers are forbidden with this "
        "instrument: serve the engine without one rather than reassembling what it split."
    )


def _record(result: ProtocolResult) -> ProtocolSample:
    """One protocol's verdict as the row records it.

    `correct` is carried through unchanged, `None` included: that is a protocol
    whose artefact could not be bound, and flattening it to `False` here would
    put a zero in a column that has no number, on every rollout, permanently.
    """
    return ProtocolSample(
        protocol=result.protocol,
        anchor=result.anchor,
        correct=result.correct,
        unavailable_reason=result.unavailable_reason,
        relaxes=result.relaxes,
    )


async def generate(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    prompt: str,
    profile: Profile,
    rendering: Rendering,
    max_tokens: int,
    seed: int | None = None,
    retries: TransportRetries | None = None,
) -> tuple[str, str | None, str | int | None, int | None]:
    """One chat completion: the text, the finish reason, the stop reason, the tokens.

    `base_url` is the OpenAI-compatible root with `/v1` already on it, which is
    what `serve.Server.base_url` hands over.

    The finish reason is carried out rather than discarded because it is the only
    trustworthy source of truncation (decision 10); inferring it from the text
    would misreport exactly the runs where the number matters most.

    The stop reason is vLLM's own field beside it, and it says *which* stop
    string matched -- or which token id did, when a stop token ended the
    completion instead. `finish_reason` cannot make that distinction: it reads
    `"stop"` both for a completion that ran into one of the profile's stop
    strings and for one that emitted an end-of-sequence token. Those two are
    exactly what separates a pretraining checkpoint, which has no learned
    terminator, from a fine-tuned one that does, so the split is collected at
    the source rather than reconstructed later from the text.
    """
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "max_tokens": max_tokens,
        # The two vLLM extensions that decide what text a grader is handed and
        # where a completion ends. Both are sent rather than defaulted, for the
        # same reason: a server default is not a recorded input. See
        # `SKIP_SPECIAL_TOKENS`.
        "skip_special_tokens": SKIP_SPECIAL_TOKENS,
    }
    for field in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
        value = getattr(profile, field)
        if value is not None:
            body[field] = value
    if rendering.chat_template_kwargs:
        body["chat_template_kwargs"] = dict(rendering.chat_template_kwargs)
    # From the resolved rendering, because the stop strings are hashed into its
    # digest: the strings sent here and the ones the fingerprint records are one
    # value, not two that have to be kept in step. A stop string that never fires
    # lets a base model run to the cap after it has answered, and decision 10
    # then scores that finished answer wrong.
    if rendering.stop:
        body["stop"] = list(rendering.stop)
    # The terminator, from the same resolved rendering and for the same reason:
    # it is inside the rendering digest, so the ids the run recorded and the ids
    # the engine is asked to stop on are one value rather than two kept in step.
    #
    # It is sent per request because the launch line cannot carry it. vLLM 0.26
    # takes the terminator from the served tokenizer's own `eos_token`, and reads
    # the artifact's `generation_config.json` for any *further* stop ids;
    # `--override-generation-config` reaches neither, since only sampling
    # parameters are read back out of an override (`ModelConfig.
    # get_diff_sampling_param`). So on a Limite run the correct id already stops
    # the completion -- it is the pinned vocabulary's -- while the artifact's own
    # stale declaration is added beside it, and nothing in the request said which
    # was intended. This says it. Where the rendering has no terminator to name,
    # nothing is sent and the engine keeps the behaviour it already had.
    if rendering.terminator:
        body["stop_token_ids"] = list(rendering.terminator)
    if seed is not None:
        body["seed"] = seed

    # The body is built once, above, and every attempt below sends that same
    # object: a resend is the same request, not a new one. See
    # `RETRYABLE_TRANSPORT_ERRORS` for which faults are resent and why, and
    # `TransportRetries` for what a resend means for the sample it produces.
    for attempt in range(TRANSPORT_ATTEMPTS):
        try:
            response = await client.post(f"{base_url}/chat/completions", json=body)
            break
        except RETRYABLE_TRANSPORT_ERRORS as error:
            # The last attempt's fault is not a resend and is not counted as one:
            # it is the exception that ends the run, and the run says so
            # everywhere already. `total` is resends performed, so that a run
            # which finished can be read as "this many rollouts needed a second
            # connection" rather than as a fault tally with no outcome attached.
            if attempt == TRANSPORT_ATTEMPTS - 1:
                raise
            if retries is not None:
                retries.record(error)
            await asyncio.sleep(TRANSPORT_BACKOFF * 2**attempt)

    if response.status_code >= 400:
        # The engine's own message names the cause -- a request over the context
        # window, an unknown model id. `raise_for_status` alone reports only the
        # status, which sends the operator to a log that may say nothing.
        raise httpx.HTTPStatusError(
            f"{response.status_code} from the engine: {response.text[:500]}",
            request=response.request,
            response=response,
        )
    payload = response.json()
    choice = payload["choices"][0]
    # Before the content is read, because the point is that reading it would
    # succeed: a parsed completion carries text, and that text scores.
    _assert_raw_text(choice["message"])
    usage = payload.get("usage") or {}
    return (
        choice["message"].get("content") or "",
        choice.get("finish_reason"),
        # Left exactly as the engine reported it -- a string for a stop string, an
        # integer for a stop token id, absent on an engine that does not send it.
        choice.get("stop_reason"),
        usage.get("completion_tokens"),
    )


async def run_taskset(
    spec: TasksetSpec,
    problems: list[tuple[str, str]],
    *,
    base_url: str,
    model: str,
    profile: Profile,
    rendering: Rendering,
    bindings: TasksetProtocols | None,
    max_tokens: int,
    seed: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    retries: TransportRetries | None = None,
    completed_keys: Set[tuple[int, int]] = frozenset(),
    on_sample: Callable[[SampleResult], None] | None = None,
) -> list[SampleResult]:
    """Every problem at its group size, scored under every protocol.

    `problems` is `(rendered prompt, gold answer)`; rendering happened upstream so
    this function never has to know which taskset prepends its instruction and
    which appends it.

    `bindings` is what this taskset's anchors bind to, resolved once by the
    caller rather than per rollout: binding `exact` executes a pinned published
    grader, and doing that per completion would re-read and re-hash the artefact
    thousands of times.

    `max_tokens` is the completion budget and is required: a taskset declares no
    budget of its own, so there is no value to fall back to and an omitted one is
    a `TypeError` here rather than a silently substituted number. `seed` is
    the run's base seed; each rollout gets its own derived from it, because one
    seed shared across a group would draw the same sample k times and make
    avg@k meaningless.

    `completed_keys` omits requests from the original full problem list, keeping
    its indices and seed derivation intact. Only new results are returned, in
    original request order. `on_sample` runs synchronously as each scored result
    finishes, before waiting for siblings; persistence failures fail the run.
    """
    budget = max_tokens
    if any(
        not (0 <= problem < len(problems) and 0 <= rollout < spec.group_size)
        for problem, rollout in completed_keys
    ):
        raise ValueError(f"{spec.name}: completed sample key is outside the requested coverage")
    semaphore = asyncio.Semaphore(concurrency)
    results: list[SampleResult] = []

    # Enough keepalive slots for every in-flight request, rather than httpx's
    # default twenty. Otherwise every connection above that default has no slot
    # to return to and is torn down and rebuilt continuously for the length of
    # the run -- churn that is the occasion for the recycling race described at
    # `RETRYABLE_TRANSPORT_ERRORS`.
    # This is a mitigation and not the fix: one of the three failures ran at a
    # concurrency of sixteen, under the default slot count, and died anyway. The
    # resend above is what makes the run survive; this makes the fault rarer.
    #
    # Both bounds are the semaphore's own, which is the honest value: no more
    # than `concurrency` requests are ever in flight, and a connection is back in
    # the pool before its rollout releases the semaphore, so `concurrency`
    # connections is exactly enough and every one of them has a slot to return to.
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(timeout=_client_timeout(timeout), limits=limits) as client:

        async def one(problem_index: int, rollout_index: int, prompt: str, gold: str) -> SampleResult:
            async with semaphore:
                text, finish_reason, stop_reason, tokens = await generate(
                    client,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    profile=profile,
                    rendering=rendering,
                    max_tokens=budget,
                    seed=None if seed is None else seed + problem_index * spec.group_size + rollout_index,
                    retries=retries,
                )
            # What every rollout records whatever scored it. Written once rather
            # than twice so the two branches below cannot drift on the fields that
            # have nothing to do with scoring -- which is every field here: how a
            # completion was produced, and how it ended, are the same facts under
            # a protocol and under a constraint checker.
            common = {
                "task": spec.name,
                "problem_index": problem_index,
                "rollout_index": rollout_index,
                "gold": gold,
                "completion": text,
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "template_sha256": rendering.sha256(),
                "truncated": finish_reason == "length",
                "completion_tokens": tokens,
                # Whether this rollout reached an answer region at all, recorded
                # once for both branches because it is a fact about the text and
                # not about what scored it. Which metrics it excludes is the
                # taskset's declaration; here it is only observed.
                "format_not_evaluable": format_not_evaluable(text, truncated=finish_reason == "length"),
                # Which scoring path wrote this row, so a later reader -- and
                # `aggregate.failure_case` in particular -- can tell a stored
                # classification it may trust from one it has to redo.
                "scorer_version": SCORER_VERSION,
            }
            answer = gold
            if bindings is None:
                raise ValueError(f"{spec.name}: answer-protocol scoring needs protocol bindings")
            scored = score_protocols(text, answer, bindings)
            # The ladder is still computed and still recorded. It is not what the
            # run reports -- the protocols are -- but `format_ok` and the two
            # extracted answers have no protocol to belong to, and every sample
            # row already written carries these fields.
            ladder = score(
                text,
                answer,
                answer_format=spec.answer_format,
                truncated=common["truncated"],
            )
            return SampleResult(
                **common,
                protocols={name: _record(result) for name, result in scored.items()},
                # What aggregation reads. Only the protocols that produced a
                # verdict appear: an unbound artefact leaves the metric absent,
                # so it can never be averaged in as a zero, and the reason for
                # its absence travels in `protocols` beside it. On a dual-lane
                # taskset the constraint metrics join them in the same mapping,
                # which is what makes a correctness column and a constraint
                # column two readings of one completion rather than of two.
            metrics={
                name: float(result.correct)
                for name, result in scored.items()
                if result.correct is not None
            },
                strict=ladder.strict,
                lenient=ladder.lenient,
                permissive=ladder.permissive,
                format_ok=ladder.format_ok,
                # Only this branch can carry one: the constraint checker never
                # calls `math-verify`, so its rows keep the False default rather
                # than record a fault that had no opportunity to occur.
                comparison_timeout=ladder.comparison_timeout,
                strict_answer=ladder.strict_answer,
                lenient_answer=ladder.lenient_answer,
                # Which shape the completion took, recorded beside the verdicts
                # rather than left for aggregation to re-derive: the row is what
                # a later reader itemises a disagreement from, and a case that
                # only ever existed in a summary cannot be traced to a rollout.
                # It changes nothing above it -- the taxonomy observes the
                # graders and never feeds them.
                failure_case=taxonomy.classify(text, answer, truncated=common["truncated"]),
            )

        async def publish(*args) -> SampleResult:
            result = await one(*args)
            if on_sample is not None:
                on_sample(result)
            return result

        pending = [
            asyncio.create_task(publish(problem_index, rollout_index, prompt, gold))
            for problem_index, (prompt, gold) in enumerate(problems)
            for rollout_index in range(spec.group_size)
            if (problem_index, rollout_index) not in completed_keys
        ]
        try:
            results = list(await asyncio.gather(*pending))
        except BaseException:
            # No callback may outlive the caller's journal or this HTTP client.
            for request in pending:
                request.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

    return results
