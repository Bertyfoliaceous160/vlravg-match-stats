"""Taskset definitions in the `verifiers` v1 shape, scored under named protocols.

One generic task covers every member of the standard suite, because they differ
only in where their rows come from and which artefacts their anchors bind to --
both of which are already data in `suites.TasksetSpec`. Near-identical task
classes would be as many places for a scoring rule to drift.

**Scoring goes through `protocols`, not through the ladder directly.** The
ladder's `strict` rung was documented as the benchmark's reference and is in
fact the verifier prime-rl executes, and those two differ (F-04). A protocol
makes the difference sayable: `exact` reproduces the benchmark's own published
grader, `reference` reproduces the function prime-rl calls, and `lenient` and
`permissive` are named relaxations carrying the anchor they relaxed. `ladder.py`
is untouched and still supplies the extraction that `reference` is composed
from -- what changed is which of them a number is reported under, not how an
answer is pulled out of a completion.

**The `verifiers` reward stays `reference`, and deliberately.** The reporting
headline is `exact`, but this class exists so the taskset is loadable by
`vf eval` for cross-checking against prime-rl -- and the reward prime-rl sums
*is* the reference verifier. Binding the framework's reward to anything else
would make the cross-check compare two different quantities. `exact` is not
exposed as a `@vf.metric` at all, because a metric must return a float and
`exact` may be legitimately absent (D-107); a float cannot express "no artefact
was bound", and 0.0 would report an unobtainable grader as a checkpoint that got
everything wrong.

Truncation comes from `trace.is_truncated`, which reads the API's finish reason,
not from inspecting the text. Decision 10 makes it a reported number, and a
heuristic guess at it would be exactly the kind of silent measurement error the
number exists to expose.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

from limite_evals.profiles import Profile
from limite_evals.suites import TasksetSpec
from limite_evals_core import protocols
from limite_evals_core.ladder import AnswerFormat
from limite_evals_core.ladder import score as ladder_score
from limite_evals_core.protocols import TasksetProtocols, Unavailable

#: Where the published graders live: inside `limite_evals_core`, shipped with the
#: package rather than under `tests/`.
#:
#: They used to live under `tests/graders/vendor`, which is not in the wheel
#: (`[tool.hatch.build.targets.wheel]` packages the source directories alone), so
#: a library module reaching for them there would have produced an `exact` column
#: from a source checkout and none from an installed one -- the worst possible
#: property for an instrument whose whole purpose is that a number means one
#: thing. The relocation landed, and this constant is the default `exact_grader`
#: resolves to when a caller names no root of its own. It has to keep matching
#: where the bytes actually are: if it ever stops, this directory does not exist
#: and `exact` reads as unavailable with the reason below, which is a safe
#: failure but a silent one.
PACKAGED_GRADER_ROOT = Path(protocols.__file__).parent / "published" / "vendor"

#: Why a taskset whose grader is declared and vendored still has no `exact`: the
#: directory it would be loaded from is not there.
#:
#: The default above is present in any install, so this is now reached only
#: through the explicit `vendor_root` argument -- which is not a dead path.
#: `tests/test_exact_parity.py` drives the explicit form for every entry in its
#: tamper table, pointing the loader at a temporary directory, and any caller
#: that pins its own copy of the vendored bytes takes the same route. What this
#: reason exists for is that such a caller naming a directory that is not there
#: must get a stated absence under D-107 rather than a `FileNotFoundError` out of
#: the middle of a scoring loop, hours into a run, with the completions already
#: paid for.
GRADERS_NOT_PACKAGED = (
    "the vendored published graders could not be read from the directory this run was told "
    "to load them from, because that directory does not exist. The graders ship inside "
    "`limite_evals_core.published.vendor` and that is the default, so this reports a caller "
    "that named a `vendor_root` of its own and named one that is not there -- not a missing "
    "install. Under D-107 an artefact that cannot be loaded reads as an absent column with a "
    "reason rather than as a number or as a crash mid-run. `reference` is unaffected and is "
    "reported normally"
)

#: Why a taskset that is genuinely part of a suite still has no `exact` anchor.
#:
#: Distinct from `Unavailable(AIME_NO_PUBLISHED_GRADER)`, and the distinction is
#: the point: that one is a checked claim that no published grader exists, while
#: this one says only that this revision has not bound one. HMMT is the case that
#: forced the difference and is also the case that shows it was worth keeping --
#: MathArena publishes a grader for it, so recording "no published grader exists"
#: there would have been false, and the grader has since been identified, pinned
#: and bound. No taskset in `SUITES` reaches this reason today; it is what a
#: taskset added to a suite without a `TASKSET_EXACT` entry would get, in place
#: of the `ValueError` `protocols.exact_grader` raises for a name it has never
#: heard of.
NO_EXACT_DECLARATION = (
    "no `exact` anchor has been declared for this taskset in `protocols.TASKSET_EXACT` at "
    "this revision. This is a statement about the instrument, not about the benchmark: it "
    "does not claim that no published grader exists, only that none has been identified, "
    "pinned and bound here yet"
)


class LimiteData(vf.TaskData):
    answer: str
    """The gold answer, as the reference taskset extracts it from its dataset."""

    problem: str = ""
    """The row's own problem text, before the taskset's instruction is applied.

    `prompt` is the rendered form and is the only one the model sees, but a
    constraint checker that asks whether the model repeated *the prompt* means
    this one: under a k-shot profile the rendered prompt carries exemplars the
    constraint never referred to. Defaulted so a caller that has no row -- the
    tests, and any caller holding only a `(problem, answer)` pair -- keeps
    working unchanged.
    """


class LimiteTaskConfig(vf.TaskConfig):
    answer_format: AnswerFormat = "boxed"
    """Which verifier the `reference` anchor reproduces."""

    taskset: str = ""
    """Which taskset's anchors to bind, since `exact` is declared per taskset."""


def taskset_protocols(spec: TasksetSpec, *, vendor_root: Path | None = None) -> TasksetProtocols:
    """The artefacts this taskset's anchors bind to.

    Takes a `TasksetSpec` rather than a name, and that is load-bearing. An
    undeclared taskset is turned into `Unavailable` here, which
    `protocols.exact_grader` deliberately refuses to do for a bare string --
    absence and a typo mean opposite things, and a mistyped name must never be
    able to manufacture a checked statement that no grader exists. A spec is
    proof the taskset is real, because specs are built in `suites` and resolved
    by name against a suite before anything reaches this function.

    `vendor_root` is an argument so a parity test can bind the real published
    graders from where they are vendored, without this module reaching into a
    test directory on a production run.
    """
    return _bindings(spec.name, spec.answer_format, vendor_root)


@lru_cache(maxsize=None)
def _bindings(
    taskset: str, answer_format: AnswerFormat, vendor_root: Path | None
) -> TasksetProtocols:
    """Cached because binding `exact` re-reads, re-hashes and executes the
    vendored sources, and a taskset scores thousands of completions against one
    binding. The digest check therefore runs once per taskset per run, which is
    where it belongs: it guards the artefact, not each use of it."""
    return TasksetProtocols(
        taskset=taskset,
        exact=_exact(taskset, vendor_root),
        reference=protocols.reference_grader_for_taskset(taskset, answer_format),
    )


def _exact(taskset: str, vendor_root: Path | None) -> protocols.Grader | Unavailable:
    """The published grader, or the reason there is no number in that column."""
    if taskset not in protocols.TASKSET_EXACT:
        return Unavailable(NO_EXACT_DECLARATION)
    declared = protocols.TASKSET_EXACT[taskset]
    if isinstance(declared, Unavailable):
        # The benchmark published no grader. That reason is the authoritative
        # one and outranks anything this module could say about packaging.
        return declared
    root = vendor_root if vendor_root is not None else PACKAGED_GRADER_ROOT
    if not root.is_dir():
        return Unavailable(GRADERS_NOT_PACKAGED)
    return protocols.exact_grader(taskset, root)


class LimiteTask(vf.Task[LimiteData, vf.State, LimiteTaskConfig]):
    """One problem, scored under every registered protocol.

    The protocols are computed once per trace and cached: each handler below is
    invoked separately by the framework, and both `math-verify` and a published
    grader are expensive enough that scoring the same completion once per column
    would dominate the run.
    """

    def __init__(self, data: LimiteData, config: LimiteTaskConfig | None = None) -> None:
        super().__init__(data, config)
        # Per instance, never a class attribute: one shared dict would accumulate
        # every trace scored in the whole run and never release them.
        self._scored: dict[str, dict[str, protocols.ProtocolResult]] = {}
        self._formats: dict[str, bool] = {}
        self._bindings = _bindings(self.config.taskset, self.config.answer_format, None)

    @property
    def gold_answer(self) -> str:
        """The answer the ladder and the protocols are scored against.

        A hook rather than a direct read keeps the protocol task explicit about
        which stored value it grades.
        """
        return self.data.answer

    def _protocols(self, trace: vf.Trace) -> dict[str, protocols.ProtocolResult]:
        key = str(trace.id)
        if key not in self._scored:
            self._scored[key] = protocols.score(
                trace.last_reply or "", self.gold_answer, self._bindings
            )
        return self._scored[key]

    def _format_ok(self, trace: vf.Trace) -> bool:
        """Whether the completion *looked* right, which no protocol reports.

        It says what the completion was shaped like rather than whether it was
        correct, so it is not a scoring protocol and has no anchor; it stays the
        ladder's own observation, computed here and reported beside them.
        """
        key = str(trace.id)
        if key not in self._formats:
            self._formats[key] = ladder_score(
                trace.last_reply or "",
                self.gold_answer,
                answer_format=self.config.answer_format,
            ).format_ok
        return self._formats[key]

    @vf.reward(weight=1.0)
    async def reference(self, trace: vf.Trace) -> float:
        """The verifier prime-rl executes, which is the reward it sums.

        `exact` is the report's headline but cannot be this, because it may be
        absent -- and because a cross-check against prime-rl has to compare the
        quantity prime-rl actually optimises.
        """
        return float(self._protocols(trace)["reference"].correct)

    @vf.metric
    async def lenient(self, trace: vf.Trace) -> float:
        return float(self._protocols(trace)["lenient"].correct)

    @vf.metric
    async def permissive(self, trace: vf.Trace) -> float:
        return float(self._protocols(trace)["permissive"].correct)

    @vf.metric
    async def format_ok(self, trace: vf.Trace) -> float:
        return float(self._format_ok(trace))

    @vf.metric
    async def truncated(self, trace: vf.Trace) -> float:
        return float(trace.is_truncated)


def load_rows(spec: TasksetSpec) -> list[tuple[str, str]]:
    """`(problem, gold answer)` pairs for a taskset, at its pinned revision.

    Field names and the gold-answer extraction follow each reference taskset
    exactly: AIME 2025 carries its problems under `question` and arrives in two
    subsets that must be concatenated, and GSM8K's gold answer is the text after
    its `####` marker rather than the whole solution.

    The pair is lossy on purpose -- it is what a caller needs to render and score
    a taskset whose instruction is a fixed string. A taskset whose instruction is
    built per row needs the fields this pair drops, and so goes through
    `load_raw_rows` instead.
    """
    return [(_spec_problem(spec, row), _gold(row, spec)) for row in load_raw_rows(spec)]


#: One dataset row, in whichever shape its loader hands it over.
#:
RawRow = dict[str, Any]


def load_raw_rows(spec: TasksetSpec) -> list[RawRow]:
    """The dataset's own rows, at the pinned revision, with nothing extracted."""
    from datasets import concatenate_datasets, load_dataset

    kwargs: dict[str, Any] = {"split": spec.split}
    if spec.revision:
        kwargs["revision"] = spec.revision

    if spec.dataset_subsets:
        rows = concatenate_datasets(
            [load_dataset(spec.dataset, subset, **kwargs) for subset in spec.dataset_subsets]
        )
    elif spec.dataset_config:
        rows = load_dataset(spec.dataset, spec.dataset_config, **kwargs)
    else:
        rows = load_dataset(spec.dataset, **kwargs)

    return [dict(row) for row in rows]


def render_row(
    spec: TasksetSpec, row: RawRow, profile: Profile | None = None
) -> tuple[str, str]:
    """One raw row as the `(rendered prompt, gold answer)` pair the runner takes.

    The single place a row becomes a prompt. `build_tasks` and `cli._evaluate`
    are two callers of one expression rather than two copies of it: a taskset
    whose instruction is per row must render the same way down both, or the
    prompt a test checks is not the prompt the run sends.

    The `base-kshot` exemplars are baked into the served template and never alter
    this prompt. `profile` remains part of the interface shared with the framework
    path, but both profiles therefore render the row with the same taskset-owned
    instruction here.

    """
    prompt = profile.kshot_prompt(spec.name, row) if profile is not None else None
    if prompt is None:
        prompt = spec.render_problem(_spec_problem(spec, row), row)
        if profile is not None:
            prompt = profile.with_static_exemplars(spec.name, prompt)
    return prompt, _gold(row, spec)


def build_tasks(
    spec: TasksetSpec, profile: Profile | None = None
) -> list[LimiteTask]:
    """The taskset's problems, rendered with its own instruction.

    Built from the raw rows rather than from `load_rows` pairs, because a per-row
    instruction reads fields the pair has already dropped and the checker-facing
    `problem` has to survive rendering.

    Every taskset uses the answer-protocol scoring lane.
    """
    config = LimiteTaskConfig(answer_format=spec.answer_format, taskset=spec.name)
    tasks: list[LimiteTask] = []
    for index, row in enumerate(load_raw_rows(spec)):
        prompt, answer = render_row(spec, row, profile)
        tasks.append(
            LimiteTask(
                LimiteData(
                    idx=index,
                    prompt=prompt,
                    problem=_spec_problem(spec, row),
                    answer=answer,
                ),
                config,
            )
        )
    return tasks


def _spec_problem(spec: TasksetSpec, row: RawRow) -> str:
    """`_problem` for a spec, in the one place its two column fields are named.

    Both fields are optional and both are `str | None`, so passing them
    positionally is a mix-up that would silently score the wrong text rather
    than fail. Mapping them once here is what stops that from being possible at
    four separate call sites.

    """
    return _problem(row, problem_field=spec.problem_field)


def _problem(
    row: dict[str, Any],
    *,
    problem_field: str | None = None,
) -> str:
    """The row's own problem text, with its multiple-choice options if it has any.

    `problem_field` names the column outright for a taskset whose source uses
    neither `problem` nor `question`. Naming it is not the same as adding more
    candidates to the search below: a maths row that happened to carry one of
    those columns would then silently change which text is scored, which is the ambiguity
    `test_aime25_problems_are_read_from_the_question_field` already refuses.

    """
    if problem_field is not None:
        problem = str(row[problem_field])
    else:
        for field in ("problem", "question"):
            if field in row:
                problem = str(row[field])
                break
        else:
            raise KeyError(f"no problem field in row with keys {sorted(row)}")
    return problem


def _gold(row: RawRow, spec: TasksetSpec) -> str:
    """What this row's completion is scored against."""
    return _answer(row, spec.answer_list_field)


def _answer(row: dict[str, Any], answer_list_field: str | None = None) -> str:
    if answer_list_field is not None:
        golds = row[answer_list_field]
        if len(golds) != 1:
            # The single element is an assumption verified over all 674 OlympiadBench
            # rows, not a convention. Taking `[0]` of a longer list would score
            # against one gold and silently discard the rest, which is the class of
            # error the strict rung exists to make impossible.
            raise ValueError(
                f"{answer_list_field} holds {len(golds)} answers, not one: {golds!r}"
            )
        return str(golds[0]).strip()

    answer = str(row["answer"])
    # GSM8K ships the full worked solution in `answer`, with the gold value after
    # its `####` marker. Every other taskset stores the bare answer.
    return answer.split("####")[-1].strip() if "####" in answer else answer.strip()
