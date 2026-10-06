"""Scoring protocols: what a number reproduces, said in the number's own record.

The ladder has four hand-written rungs and their names claim more than the code
delivers. `strict` is documented as the benchmark's reference and is in fact the
verifier prime-rl executes -- and those two differ, which is why GSM8K's rung
hands a whole completion to `math-verify` where MATH-500's refuses to guess. The
looser rungs are worse: "forgives an unclosed think block, an unterminated box,
a trailing bare number" is a sentence in a docstring, so what a `lenient` number
forgave is recoverable only by reading `ladder.lenient_answer` at the revision
that produced it.

This module replaces the adjective with an object. A protocol is an id, an
**anchor** naming the artefact it reproduces, and -- when it is a relaxation --
the closed set of **waivers** it applies on top of its base. All three travel
with the number, so a report says which grader produced a column and exactly
what that column forgave.

| id | anchor | reproduces |
|---|---|---|
| `exact` | `published-grader` | the taskset's declared pinned grader, executed |
| `reference` | `prime-rl-verifier` | the function prime-rl actually calls |
| `lenient` | `relaxation-of` | its taskset's available base, plus five waivers |
| `permissive` | `relaxation-of` | its taskset's available base, plus all six |

OlympiadBench deliberately uses Limite's MathArena scoring contract, recorded
separately from historical OpenBMB results; see `OLYMPIADBENCH_EXACT`.

**`exact` carries no scoring logic of its own, and that is the whole point.** It
has no default and no fallback: a taskset either binds a concrete grader that
this repository executes, or declares `Unavailable` with a reason and the column
reads as absent. There is deliberately no third path, because the moment `exact`
can be satisfied by anything local it stops being the thing decision D-103 named
and becomes another rung. D-107 -- an unobtainable published grader must never
read as a number -- therefore falls out of the type rather than out of a policy
someone has to remember.

**The relaxations relax `exact` where a published grader exists, and `reference`
where none does.** D-103 defined them as relaxations of `exact`; D-107 makes an
unobtainable published grader absent rather than zero; composed strictly, those
two blank three columns instead of one. AIME24 and AIME25 -- half of
`math-standard` -- have no published grader to relax, so a strict composition
would have deleted the strict-to-lenient gap on exactly the tasksets where
post-training moves it most. D-108 resolved that with a fallback base, chosen by
the user over the alternative of always relaxing `reference`.

**`lenient(exact)` and `lenient(reference)` are not the same quantity.** They
forgive the same six things on top of two different graders, so a `lenient`
number for MATH-500 and a `lenient` number for AIME24 are not comparable
figures -- they are one waiver list applied to two artefacts. Naming the column
identically does not make them one measurement, and the objection that a single
column then carries two meanings across rows of one table is a real cost, put to
the user and accepted by them.

What makes it survivable is that the base is never implicit. Every relaxed
`ProtocolResult` names the anchor it relaxed in `relaxes` and carries that
anchor's own result in `base`, so the thing a number was relaxed from travels
with the number rather than being inferred from whether `exact` happened to be
available in some other column. A rendering surface cannot show `lenient`
without having its base in hand; a `ProtocolResult` for a relaxation that names
no base cannot be constructed at all. That is the mitigation D-108 attaches to
the fallback, and it is a constructor invariant rather than a convention.

**The two anchors are independent and do not nest.** `exact` and `reference`
name different artefacts, so neither contains the other, and no code here
pretends otherwise. `tests/test_protocols.py` carries a concrete completion that
`reference` scores correct and `permissive` scores wrong.

**What `exact` binds to is declared per taskset in `TASKSET_EXACT`,** and the
declaration is a pinned file rather than a description of one: a repository, a
revision, a path, and the SHA-256 of the bytes at that revision. `exact_grader`
executes those bytes; it does not re-express what they do. Where no grader can
be bound the entry is an `Unavailable` carrying the reason, and
`tests/test_exact_parity.py` refuses to pass silently over it. Those reasons are
not one fact in three wordings: a benchmark that published no grader, a taskset
this revision has not reached, and a taskset whose correctness is not decided by
checking an answer at all are three different states.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import signal
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from limite_evals_core.equivalence import equivalent
from limite_evals_core.ladder import (
    AnswerFormat,
    extract_boxed,
    lenient_answer,
    permissive_candidates,
    strict_answer,
    strict_answer_hashed,
    strip_think,
    think_unclosed,
)
from limite_evals_core.published import VENDOR_ROOT

#: The closed set of things a relaxation may forgive. Each name corresponds to
#: exactly one fallback `ladder.lenient_answer` or `ladder.permissive_candidates`
#: already implements; nothing here invents a new way to recover an answer. A
#: closed set is what makes "a bit more permissive" a list a manifest can record.
Waiver = Literal[
    "unclosed_think",
    "unterminated_box",
    "answer_phrase",
    "last_line_inline_math",
    "last_line_bare_number",
    "answer_anywhere",
]

WAIVERS: frozenset[Waiver] = frozenset(
    (
        "unclosed_think",
        "unterminated_box",
        "answer_phrase",
        "last_line_inline_math",
        "last_line_bare_number",
        "answer_anywhere",
    )
)

#: The four waivers `ladder.lenient_answer` realises **jointly**. That function is
#: one ordered ladder -- boxed, unterminated box, answer phrase, inline maths on
#: the last line, bare number on the last line -- and it returns the first branch
#: that fires. Disabling one branch without the others would mean rewriting it,
#: and `ladder.py` is read-only by decision. So a protocol declares all four or
#: none, and `Protocol` refuses a subset rather than quietly over-forgiving.
LADDER_FALLBACK_WAIVERS: frozenset[Waiver] = frozenset(
    ("unterminated_box", "answer_phrase", "last_line_inline_math", "last_line_bare_number")
)

#: What a protocol claims to reproduce. `published-grader` and `prime-rl-verifier`
#: name an external artefact that a parity test executes; `relaxation-of` names a
#: protocol in this registry plus the waivers applied on top of it.
AnchorKind = Literal["published-grader", "prime-rl-verifier", "relaxation-of"]

#: `(completion, gold) -> correct`. Published graders return a float score and
#: take their arguments in their own order; adapting one to this signature is the
#: binding's job, so that the adapter is visible in the taskset declaration rather
#: than buried in a scoring path.
GradeFn = Callable[[str, str], bool]


@dataclass(frozen=True)
class Anchor:
    """What a protocol is anchored to, and -- for a relaxation -- to which base.

    The concrete artefact name is deliberately *not* here: each taskset has its
    own published grader and its own reference verifier, so the name belongs to
    the per-taskset binding. `base` is registry-level and therefore is.
    """

    kind: AnchorKind
    base: str | None = None


@dataclass(frozen=True)
class Protocol:
    """A named scoring protocol: an id, an anchor, and its waivers."""

    id: str
    anchor: Anchor
    waivers: frozenset[Waiver] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = self.waivers - WAIVERS
        if unknown:
            raise ValueError(f"unknown waivers on protocol {self.id!r}: {sorted(unknown)}")
        if self.anchor.kind == "relaxation-of":
            if self.anchor.base is None:
                raise ValueError(f"protocol {self.id!r} relaxes nothing: no base named")
            if not self.waivers:
                raise ValueError(f"protocol {self.id!r} relaxes {self.anchor.base!r} by nothing")
            partial = self.waivers & LADDER_FALLBACK_WAIVERS
            if partial and partial != LADDER_FALLBACK_WAIVERS:
                missing = sorted(LADDER_FALLBACK_WAIVERS - partial)
                raise ValueError(
                    f"protocol {self.id!r} waives part of the lenient ladder; "
                    f"it is one ordered function and cannot be split. Missing: {missing}"
                )
        else:
            if self.anchor.base is not None:
                raise ValueError(f"anchored protocol {self.id!r} must not name a base")
            if self.waivers:
                raise ValueError(
                    f"anchored protocol {self.id!r} must not waive anything: it reproduces "
                    "an external artefact, and a waiver is a local variation"
                )

    def describe(self) -> str:
        """The waiver list as it belongs in a manifest, sorted for stability."""
        return ", ".join(sorted(self.waivers))


#: `lenient` forgives the four fallbacks plus an unclosed think block, which is
#: the shape of a completion that hit the token limit. `permissive` adds the
#: answer appearing anywhere at all, reasoning included. Both match what
#: `ladder.score` already computes for its rungs of the same name.
_LENIENT_WAIVERS: frozenset[Waiver] = LADDER_FALLBACK_WAIVERS | frozenset(("unclosed_think",))

PROTOCOLS: dict[str, Protocol] = {
    protocol.id: protocol
    for protocol in (
        Protocol("exact", Anchor("published-grader")),
        Protocol("reference", Anchor("prime-rl-verifier")),
        Protocol("lenient", Anchor("relaxation-of", base="exact"), _LENIENT_WAIVERS),
        Protocol("permissive", Anchor("relaxation-of", base="exact"), WAIVERS),
    )
}

for _protocol in PROTOCOLS.values():
    if _protocol.anchor.base is not None and _protocol.anchor.base not in PROTOCOLS:
        raise ValueError(f"protocol {_protocol.id!r} relaxes an unregistered base")


def protocol(protocol_id: str) -> Protocol:
    """The protocol with this id. Raises on an unknown one rather than defaulting,
    so a typo in a config cannot silently change which artefact a metric names."""
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol_id!r}")
    return PROTOCOLS[protocol_id]


@dataclass(frozen=True)
class Grader:
    """A concrete artefact an anchor binds to, and the call that executes it.

    `name` is what the manifest records and what a reader goes and reads -- a
    module path, a file in a pinned submodule, a vendored script. It is not a
    label: the parity test for this protocol executes exactly what `grade` calls.
    """

    name: str
    grade: GradeFn


@dataclass(frozen=True)
class Unavailable:
    """No artefact could be bound, with the reason it could not.

    The only legitimate value for `exact` besides a `Grader`. AIME has no
    official grader, only de-facto ones, and inventing a plausible substitute
    would put a number in the column that no published artefact backs.
    """

    reason: str


@dataclass(frozen=True)
class TasksetProtocols:
    """Which artefacts one taskset's anchors bind to.

    `reference` admits no `Unavailable`: it is the function prime-rl executes on
    this taskset, and a taskset the in-run evaluation cannot score is not a
    taskset this repository can claim comparability with. If one ever appears,
    that is a stop-and-report, not a third state to add here.
    """

    taskset: str
    exact: Grader | Unavailable
    reference: Grader


#: The base a relaxation falls back to when its declared base is unavailable.
#: `reference` and not something cleverer, because it is the only anchor that
#: cannot be absent: `TasksetProtocols.reference` admits no `Unavailable`. This is
#: D-108, and it is the one place the fallback is written down.
RELAXATION_FALLBACK_BASE = "reference"


@dataclass(frozen=True)
class ProtocolResult:
    """One completion scored under one protocol, carrying what produced it.

    `correct is None` is absence, not failure: no artefact was bound, so no
    number exists. A reader who cannot tell that apart from `False` reads an
    unobtainable grader as a model that got everything wrong.

    For a relaxation, `relaxes` names the anchor that was actually relaxed and
    `base` is that anchor's own result. Both or neither -- the constructor
    refuses a relaxed result whose base is missing, and refuses a base that
    disagrees with the name. Under D-108 the base varies by taskset, so a
    relaxed number without it is not underspecified prose, it is a number whose
    meaning cannot be recovered.
    """

    protocol: str
    anchor: str | None
    correct: bool | None
    unavailable_reason: str | None = None
    relaxes: str | None = None
    base: ProtocolResult | None = None

    def __post_init__(self) -> None:
        if (self.relaxes is None) != (self.base is None):
            raise ValueError(
                f"result for {self.protocol!r} names a relaxation base without carrying it, "
                "or the reverse; under D-108 the base varies per taskset and must travel "
                "with the number"
            )
        if self.base is not None and self.base.protocol != self.relaxes:
            raise ValueError(
                f"result for {self.protocol!r} says it relaxes {self.relaxes!r} but carries "
                f"the result of {self.base.protocol!r}"
            )

    @property
    def available(self) -> bool:
        return self.correct is not None


def reference_grader(answer_format: AnswerFormat) -> Grader:
    """The `reference` binding for a taskset scored under `answer_format`.

    Composed from the ladder rather than reimplemented, and named after the
    artefact the ladder reproduces. Active maths tasksets, including GSM8K, use
    `boxed`; `hashed` remains available only to interpret historical rows whose
    recorded answer format names the retired GSM8K verifier.

    `tests/test_protocols.py` asserts this agrees with `ladder.score(...).strict`
    on every vector, so the composition cannot drift from the rung it replaces.
    """
    if answer_format == "hashed":
        return Grader(
            name="verifiers environments/gsm8k_v1/gsm8k_v1/verify.py",
            grade=_grade_hashed,
        )
    return Grader(
        name="verifiers.v1.utils.score.verify_boxed_math_answer",
        grade=_grade_boxed,
    )


def reference_grader_for_taskset(taskset: str, answer_format: AnswerFormat) -> Grader:
    """Bind a task-specific reference when one is part of its pinned protocol."""
    return reference_grader(answer_format)


def _grade_boxed(completion: str, gold: str) -> bool:
    """`ladder.score`'s strict branch, and nothing else."""
    answer = strict_answer(completion)
    return bool(answer) and equivalent(_unwrap_gold(gold), answer)


def _grade_hashed(completion: str, gold: str) -> bool:
    """`ladder._score_hashed`'s strict branch: the answer region is the fallback.

    No gold unwrapping here, because the reference does none -- the asymmetry
    with the boxed branch is inherited deliberately rather than smoothed over.
    What is *not* inherited is the reference's blindness to reasoning
    delimiters: the marker is looked for in the answer region, the fallback is
    that region rather than the whole text, and an unclosed think block has no
    region and fails. See `ladder.strict_answer_hashed` for why reproducing the
    reference bit-for-bit here would reproduce a defect rather than a protocol.
    """
    if think_unclosed(completion):
        return False
    answer = strict_answer_hashed(completion)
    return equivalent(gold, answer if answer is not None else strip_think(completion))


def _unwrap_gold(gold: str) -> str:
    """`\\boxed{42}` shipped as a gold answer is the answer 42.

    Mirrors `ladder.score`, which unwraps before comparing, so a taskset that
    ships a wrapped gold is not scored zero on every rollout.
    """
    return (extract_boxed(gold) or gold).strip()


def waived_candidates(completion: str, waivers: frozenset[Waiver]) -> list[str]:
    """Every answer the given waivers recover from `completion`, base excluded.

    An unclosed think block gates everything: a completion that stopped
    mid-reasoning yields no waived candidate at all unless `unclosed_think` is
    waived. That is the ladder's own rule for `lenient`, extended to
    `answer_anywhere` for consistency -- both relaxations in the registry waive
    it, so this changes nothing today and refuses to guess for a protocol that
    does not.
    """
    if not waivers:
        return []
    if think_unclosed(completion) and "unclosed_think" not in waivers:
        return []

    candidates: list[str] = []
    if waivers & LADDER_FALLBACK_WAIVERS:
        fallback = lenient_answer(completion)
        if fallback is not None:
            candidates.append(fallback)
    if "answer_anywhere" in waivers:
        candidates.extend(permissive_candidates(completion))
    return candidates


def score(completion: str, gold: str, bindings: TasksetProtocols) -> dict[str, ProtocolResult]:
    """Score one completion under every registered protocol.

    Returned in registry order, keyed by protocol id, so a caller writing a row
    per protocol writes the same columns in the same order every run.
    """
    results: dict[str, ProtocolResult] = {}
    for spec in PROTOCOLS.values():
        if spec.anchor.kind == "relaxation-of":
            results[spec.id] = _score_relaxation(spec, completion, gold, results)
        else:
            results[spec.id] = _score_anchored(spec, completion, gold, bindings)
    return results


#: A grader that has not returned in this many seconds is not going to. Healthy
#: graders return in milliseconds; the bound exists because an anchored grader
#: runs arbitrary sympy and regex over a whole model completion, and one
#: pathological completion can loop it forever with nothing reaching the run --
#: observed on aime26 at seed 1234, where the same rollout wedged two jobs in a
#: row. `math-verify` bounds itself the same way inside `equivalence`; this is
#: the identical reading at the anchor boundary: a grader that hits its bound
#: scores wrong, stated in a warning, never a hang.
GRADER_TIMEOUT = 30


class _GraderGaveUp(Exception):
    """Raised by the alarm when an anchored grader exceeds `GRADER_TIMEOUT`."""


@contextmanager
def _bounded_grading() -> Iterator[None]:
    """A `signal.alarm` bound around one grader call.

    Signals only exist on the main thread, which is where the runner scores; on
    any other thread this deliberately degrades to no bound rather than raising,
    matching what `math-verify`'s own alarm-based timeout would do there.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _give_up(signum: int, frame: Any) -> None:
        raise _GraderGaveUp()

    previous = signal.signal(signal.SIGALRM, _give_up)
    signal.alarm(GRADER_TIMEOUT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _score_anchored(
    spec: Protocol, completion: str, gold: str, bindings: TasksetProtocols
) -> ProtocolResult:
    binding = bindings.exact if spec.anchor.kind == "published-grader" else bindings.reference
    if isinstance(binding, Unavailable):
        return ProtocolResult(spec.id, anchor=None, correct=None, unavailable_reason=binding.reason)
    try:
        with _bounded_grading():
            correct = bool(binding.grade(completion, gold))
    except _GraderGaveUp:
        logging.getLogger(__name__).warning(
            "%s grader %s exceeded %ss on one completion; scored wrong",
            spec.id,
            binding.name,
            GRADER_TIMEOUT,
        )
        correct = False
    return ProtocolResult(spec.id, anchor=binding.name, correct=correct)


def _score_relaxation(
    spec: Protocol, completion: str, gold: str, scored: dict[str, ProtocolResult]
) -> ProtocolResult:
    """The base's verdict, or any waived candidate that is equivalent to the gold.

    The candidates are judged with `equivalence.equivalent` rather than by the
    base grader, because a waived candidate is an answer string and a published
    grader takes a completion. Worth stating plainly: that means a relaxation
    also forgives any disagreement between the published grader's own
    equivalence and `math-verify`'s. It is a waiver the closed set does not name,
    and it is inherent to relaxing an opaque grader at all.

    Which base gets relaxed is `_relaxation_base`'s answer, not this function's
    assumption, and it comes back in the result.
    """
    base = _relaxation_base(spec, scored)
    anchor = f"{base.protocol} + {spec.describe()}"
    correct = bool(base.correct) or any(
        equivalent(_unwrap_gold(gold), candidate)
        for candidate in waived_candidates(completion, spec.waivers)
    )
    return ProtocolResult(spec.id, anchor=anchor, correct=correct, relaxes=base.protocol, base=base)


def _relaxation_base(spec: Protocol, scored: dict[str, ProtocolResult]) -> ProtocolResult:
    """The declared base if it produced a number, otherwise D-108's fallback.

    The fallback is not defensive: for AIME24 and AIME25 it is the ordinary path,
    because neither has a published grader and neither ever will have one on this
    revision. If the fallback itself is absent, something has bound `reference`
    to nothing, which `TasksetProtocols` forbids by type -- so this raises rather
    than inventing a third base, and the raise is the stop-and-report W-1's
    docstring promised.
    """
    declared = scored.get(spec.anchor.base) if spec.anchor.base is not None else None
    if declared is not None and declared.available:
        return declared

    fallback = scored[RELAXATION_FALLBACK_BASE]
    if not fallback.available:
        raise ValueError(
            f"protocol {spec.id!r} has no base to relax: its declared base "
            f"{spec.anchor.base!r} is unavailable and so is the {RELAXATION_FALLBACK_BASE!r} "
            "fallback, which cannot happen through TasksetProtocols"
        )
    return fallback


# --------------------------------------------------------------------------- #
# The `exact` anchor: which published grader each taskset binds to.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PublishedSource:
    """One file of a benchmark's published grader, pinned tightly enough to check.

    A name is not provenance. `repository`, `revision` and `path` locate the exact
    bytes upstream, `sha256` is the digest of those bytes, and `vendored` is the
    copy of them kept in this repository so the parity test runs offline. The test
    asserts the copy still hashes to `sha256`, and -- when the network allows --
    that `url` still serves the same bytes. An artefact that fails either is not
    the artefact this protocol claims to reproduce.

    `names` selects top-level definitions when the published file cannot be
    executed whole. That is not editing: the excerpt is a verbatim line slice of
    the vendored bytes, and `third_party_imports` pins the imports the slice
    leaves behind, so a revision that grows a dependency fails loudly instead of
    quietly executing a narrower file.

    `required_imports` is the other half of that pin and it exists because
    dropping an import is only safe when the grader does not need it. MATH's
    evaluation script imports `transformers` and `remove_boxed` never touches it;
    prm800k's grader imports `sympy` and `pylatexenc` and its whole answer-
    equivalence path runs through them, inside bare `except:` blocks that would
    turn a missing name into a silently stricter grader. Roots named here are
    kept in the excerpt and therefore imported for real, so an environment that
    cannot supply one fails at load with an `ImportError` rather than producing
    numbers under a protocol it is no longer running.

    `module` handles the case where one published file reaches another through
    the package it was published in -- prm800k's `grader.py` calls
    `math_normalize.normalize_answer(...)`, and `from grading import
    math_normalize` cannot resolve outside its repository. The sibling source
    declares the name it is known by, and what it defined is exposed under that
    name in the shared namespace. Same convention the sibling imports already
    follow, one level up: the import is dropped and pinned, and the vendored,
    digest-checked bytes it names supply what it would have bound.
    """

    repository: str
    revision: str
    path: str
    sha256: str
    vendored: str
    names: tuple[str, ...] = ()
    third_party_imports: tuple[str, ...] = ()
    required_imports: tuple[str, ...] = ()
    module: str | None = None

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{self.path}"

    def __str__(self) -> str:
        return f"{self.repository}@{self.revision[:7]}:{self.path}"


@dataclass(frozen=True)
class PublishedConfig:
    """A published *configuration* a grader is run under, pinned like a source.

    Not a `PublishedSource`, because it is never executed: there is no excerpt to
    take, no import to drop and none to require, so the two halves of that pin
    would have nothing to hold. What it shares is the half that decides a number
    -- repository, revision, path, and the digest of the bytes -- because a
    published grader run under a different published configuration is a different
    grader. MathArena's `extract_and_grade` reads five flags off one of these
    files, and which of the five the file sets is the whole difference between
    an extraction that demands `\\boxed` and one that falls back to a bare
    integer.

    **It is not parsed here, and the reason is a dependency rather than a
    preference.** `limite-evals-core` deliberately has no YAML reader; adding a
    parser so that library code could read one line of a five-line file would be
    the wrong trade. So the bytes are vendored and digest-pinned like every
    other artefact in this module, and `tests/test_exact_parity.py` asserts
    against those bytes that the only flag `extract_and_grade` reads which
    either config sets is the one `_bind_matharena` passes. That makes the
    binding's claim about the config a checked one without making the claim's
    checker part of the library.
    """

    repository: str
    revision: str
    path: str
    sha256: str
    vendored: str

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{self.path}"

    def __str__(self) -> str:
        return f"{self.repository}@{self.revision[:7]}:{self.path}"


@dataclass(frozen=True)
class PublishedGrader:
    """A taskset's `exact` declaration: the artefact, and how it is called.

    `bind` receives the namespace the sources executed into and returns the
    `GradeFn`. It adapts argument order and the shape of a gold answer -- nothing
    else. Anything that changed a verdict would be the local variation `exact`
    exists to exclude, and `tests/test_exact_parity.py` carries vectors on which
    the published verdict differs from this repository's own ladder precisely so
    that a binding which quietly stopped executing the artefact would fail.
    """

    artefact: str
    sources: tuple[PublishedSource, ...]
    bind: Callable[[dict[str, Any]], GradeFn]


def _verbatim_excerpt(
    text: str, names: tuple[str, ...], required: tuple[str, ...] = ()
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Executable verbatim slices of a published file, and what was left out.

    Published graders are shipped inside research code: GSM8K's lives in the file
    that also defines a `torch` dataset, and MATH's `remove_boxed` lives in an
    evaluation script that imports `transformers`. Neither file imports as a
    module here, and vendoring an edited copy would forfeit the claim. So the
    rule is mechanical and stated rather than judged: keep every top-level import
    of a standard-library module, keep every import whose roots the source
    declared `required`, keep the requested top-level definitions, drop the rest,
    and return both the imports that were dropped and the foreign ones that were
    kept so a caller can pin each set. Every kept line is byte-for-byte the
    published line.

    Returning the kept foreign roots as well as the dropped ones is what makes
    the pin symmetric: a revision that removes `import sympy` would otherwise
    satisfy the dropped-imports check by dropping nothing, and prm800k's grader
    without sympy still returns a verdict -- a stricter one, silently.

    **A kept definition keeps its decorators, and until this revision it did
    not.** `ast.FunctionDef.lineno` is the line of the `def`, not of the first
    `@`, so slicing from it silently produced an undecorated copy of a decorated
    published function -- which is precisely the local variation this function
    exists to make impossible, arrived at by omission rather than by editing. It
    `_first_line` preserves decorators; otherwise a digest-correct source could
    still execute as a locally modified grader.
    """
    module = ast.parse(text)
    lines = text.splitlines(keepends=True)
    kept: list[ast.stmt] = []
    dropped: list[str] = []
    satisfied: list[str] = []

    for node in module.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            foreign = sorted(_import_roots(node) - sys.stdlib_module_names)
            if not foreign:
                kept.append(node)
            elif set(foreign) <= set(required):
                kept.append(node)
                satisfied.extend(foreign)
            else:
                dropped.extend(foreign)
        elif not names or _defined_names(node) & set(names):
            kept.append(node)

    excerpt = "".join(
        "".join(lines[_first_line(node) - 1 : node.end_lineno]) for node in kept if node.end_lineno
    )
    return excerpt, tuple(dict.fromkeys(dropped)), tuple(dict.fromkeys(satisfied))


def _first_line(node: ast.stmt) -> int:
    """The first published line of a statement, decorators included.

    A decorator sits above the `def` it applies to and the AST does not count it
    as part of the definition, so the span has to be widened rather than read
    off. Written as a `min` over the decorators and the node itself because a
    decorator expression may span lines and only the first of them carries the
    `@`.
    """
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _import_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    # A relative import has no resolvable root here, and is never standard library.
    return {node.module.split(".")[0]} if node.module and not node.level else {""}


def _defined_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    return set()


def execute_published(
    sources: tuple[PublishedSource, ...], vendor_root: Path | None = None
) -> dict[str, Any]:
    """Run the vendored published files and return the namespace they built.

    Digest first, execute second. If the vendored bytes have drifted from the
    revision they claim, this raises before anything runs -- a grader that no
    longer is what it says it is must not produce a number under the name of the
    one that was.

    `vendor_root` defaults to where the bytes actually live, so a caller does not
    have to know. It stays overridable for the one test that points at a tampered
    copy in a temporary directory.
    """
    namespace: dict[str, Any] = {}
    for source in sources:
        path = Path(vendor_root if vendor_root is not None else VENDOR_ROOT) / source.vendored
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source.sha256:
            raise ValueError(
                f"vendored {source.vendored} is not the artefact it claims: {source} hashes "
                f"to {digest}, pinned {source.sha256}"
            )

        with warnings.catch_warnings():
            # `math_equivalence.py` contains a literal `\%`, which is an invalid
            # escape sequence under a modern Python. Fixing it would change the
            # digest; the escape is inside a `.replace()` argument and behaves
            # identically, so the warning is suppressed and the bytes are not.
            # It fires from the parse as well as the compile, hence the span.
            warnings.simplefilter("ignore", SyntaxWarning)
            excerpt, dropped, satisfied = _verbatim_excerpt(
                raw.decode(), source.names, source.required_imports
            )
            if dropped != source.third_party_imports:
                raise ValueError(
                    f"{source} drops {dropped} rather than the pinned "
                    f"{source.third_party_imports}: the excerpt is not the one that was reviewed"
                )
            if satisfied != source.required_imports:
                raise ValueError(
                    f"{source} keeps the foreign imports {satisfied} rather than the declared "
                    f"{source.required_imports}: an import this grader's verdict depends on is "
                    "no longer in the file, and dropping it would only make the grader stricter"
                )
            defined_before = set(namespace)
            exec(compile(excerpt, str(path), "exec"), namespace)  # noqa: S102

        if source.module is not None:
            namespace[source.module] = SimpleNamespace(
                **{
                    name: value
                    for name, value in namespace.items()
                    if name not in defined_before and not name.startswith("__")
                }
            )
    return namespace


#: How a completion becomes an answer string, in MATH's own published code. Both
#: graders below share these two files, because extraction is not the thing they
#: disagree about: `last_boxed_only_string` takes the last `\boxed`/`\fbox` span
#: anywhere in the string and `remove_boxed` unwraps it, returning `None` for
#: anything not prefixed `\boxed{`. `remove_boxed` is lifted out of the
#: evaluation script rather than rewritten because that script is where the
#: published harness composes the pair, at `modeling/eval_math_gpt.py` lines
#: 204-240.
_MATH_EXTRACTION_SOURCES = (
    PublishedSource(
        repository="hendrycks/math",
        revision="985bdc1696e88e8643f081a0ff4719da39f2ae2a",
        path="modeling/dataset/util.py",
        sha256="9183a9ea7bce3c126ac33f993d21bf85cffbdb7bab10088216d286699c9e084a",
        vendored="hendrycks_math__modeling__dataset__util.py",
    ),
    PublishedSource(
        repository="hendrycks/math",
        revision="985bdc1696e88e8643f081a0ff4719da39f2ae2a",
        path="modeling/eval_math_gpt.py",
        sha256="e0a70061d1be9d37681dd07f632d19d177beffc1ee17a10148008d1c95877ead",
        vendored="hendrycks_math__modeling__eval_math_gpt.py",
        names=("remove_boxed",),
        third_party_imports=("transformers", "numpy", "tqdm", "torch", "dataset", "math_equivalence"),
    ),
)

#: The taskset is `HuggingFaceH4/MATH-500`, and it is not the MATH benchmark. It
#: is the 500 problems OpenAI held out in *Let's Verify Step by Step*: they moved
#: 4,500 of MATH's 5,000 test problems into training and evaluated on the
#: remainder, and that remainder is `prm800k/math_splits/test.jsonl`. The
#: Hugging Face dataset card says so and links to exactly that file. So the
#: benchmark whose protocol `exact` must reproduce is the subset, its definers
#: are OpenAI, and the grader they published for it is
#: `prm800k/grading/grader.grade_answer` -- described in their README as "the
#: python grading logic we used for determining if a model-outputted answer
#: correctly matched the ground truth answer", with the recommendation to call
#: `grade_answer(model_answer, gt_answer)` on two strings.
#:
#: MATH's own `is_equiv` was the previous binding and is retained below as
#: `MATH500_HENDRYCKS`, because the choice between them is a real one and the
#: size of it is measurable. `math_normalize.py`'s own docstring says it is
#: "largely copied from the Hendrycks' MATH release (math_equivalence)", and
#: `grade_answer` runs that copy first and only then adds a sympy path -- so the
#: prm800k grader accepts a strict superset of what `is_equiv` accepts, which is
#: what their README means by rejecting correct answers "less frequently than the
#: normalization logic from MATH". `is_equiv` is the grader of the *parent*
#: dataset's authors: a real published artefact, but one anchor removed from this
#: taskset. `tests/test_exact_parity.py` carries the vectors the two split on.
#:
#: What is composed rather than taken whole, and it must be said plainly:
#: `grade_answer` grades an answer against an answer and publishes no extraction
#: step, because OpenAI's samples arrived with an `answer` field their unreleased
#: sampling harness had already filled in. The extraction is therefore MATH's
#: published pair, above. Both halves are published artefacts executed verbatim;
#: neither is re-expressed here; and the seam between them is this comment.
MATH500_EXACT = PublishedGrader(
    artefact=(
        "openai/prm800k@7ecc794703b2877f63226f2477a49b34f9b25163 "
        "prm800k/grading/grader.grade_answer over "
        "hendrycks/math@985bdc169 "
        "modeling/eval_math_gpt.remove_boxed(modeling/dataset/util.last_boxed_only_string(...))"
    ),
    sources=(
        *_MATH_EXTRACTION_SOURCES,
        PublishedSource(
            repository="openai/prm800k",
            revision="7ecc794703b2877f63226f2477a49b34f9b25163",
            path="prm800k/grading/math_normalize.py",
            sha256="13998bf8c35bdc8a76868c84e1704a78567e2552557fda69027cc186568d3a4d",
            vendored="openai_prm800k__prm800k__grading__math_normalize.py",
            module="math_normalize",
        ),
        PublishedSource(
            repository="openai/prm800k",
            revision="7ecc794703b2877f63226f2477a49b34f9b25163",
            path="prm800k/grading/grader.py",
            sha256="9e8bbb6f504ee0d8068e1eca031d78174f1c925cfec391996fdf45c21bb016a9",
            vendored="openai_prm800k__prm800k__grading__grader.py",
            third_party_imports=("grading",),
            required_imports=("sympy", "pylatexenc"),
        ),
    ),
    bind=lambda namespace: _bind_math500(namespace),
)

#: MATH's own grader over the same extraction: what MATH-500's `exact` was bound
#: to before this node, kept because a superseded anchor is evidence and not
#: clutter. It is deliberately absent from `TASKSET_EXACT` -- nothing scores under
#: it -- and present here so `tests/test_exact_parity.py` can execute it beside
#: the bound one and record, as a table of concrete completions, how much the
#: provenance question was actually worth.
MATH500_HENDRYCKS = PublishedGrader(
    artefact=(
        "hendrycks/math@985bdc1696e88e8643f081a0ff4719da39f2ae2a "
        "modeling/math_equivalence.is_equiv over "
        "modeling/eval_math_gpt.remove_boxed(modeling/dataset/util.last_boxed_only_string(...))"
    ),
    sources=(
        *_MATH_EXTRACTION_SOURCES,
        PublishedSource(
            repository="hendrycks/math",
            revision="985bdc1696e88e8643f081a0ff4719da39f2ae2a",
            path="modeling/math_equivalence.py",
            sha256="c4101b1f51a2bb65665194aecd9f761d668c0c248a0350f196c2950e87488527",
            vendored="hendrycks_math__modeling__math_equivalence.py",
        ),
    ),
    bind=lambda namespace: _bind_math500_hendrycks(namespace),
)

#: MathArena's own grader, and the benchmark it is the benchmark's own grader
#: *for*. `eth-sri/matharena` is what the dataset cards of `MathArena/aime_2026`
#: and `MathArena/hmmt_feb_2025` both name as their `Repository:`, so the same
#: argument that binds MATH-500 to prm800k rather than to MATH applies here: the
#: taskset is MathArena's benchmark, its definers published a grader for it, and
#: that grader is what makes a number comparable with matharena.ai's leaderboard.
#:
#: **What this does not claim.** The AIME is administered by the Mathematical
#: Association of America and HMMT by its tournament; neither publishes a grader,
#: and neither does here. `AIME_NO_PUBLISHED_GRADER` below stays true of the
#: competitions. What changed for AIME 2026 is the *dataset the suite pins*: it
#: is MathArena's, and MathArena publishes both halves.
#:
#: The four files are one grader spread across a package. `parse_manual` is pure
#: data; `parser` is the extraction and the equivalence; `utils` supplies the one
#: name `extract_and_grade` calls unconditionally; `grader` is the entry point the
#: runner uses. Each reaches the previous through `from matharena... import`,
#: which cannot resolve outside its repository -- so those imports are dropped and
#: pinned, and the vendored, digest-checked bytes executed before them supply what
#: each would have bound. `sympy`, `regex` and `loguru` decide verdicts and are
#: declared `required_imports`, which keeps them in the excerpt and imported for
#: real.
_MATHARENA_REVISION = "a11194deff8c67a232974a383795e8a2776b4c6f"

_MATHARENA_SOURCES = (
    PublishedSource(
        repository="eth-sri/matharena",
        revision=_MATHARENA_REVISION,
        path="src/matharena/parse_manual.py",
        sha256="8858daf09f0b8fda2326853874db13fe32d03b506464edcd8f96215db59a3936",
        vendored="eth_sri_matharena__src__matharena__parse_manual.py",
    ),
    PublishedSource(
        repository="eth-sri/matharena",
        revision=_MATHARENA_REVISION,
        path="src/matharena/parser.py",
        sha256="b3eda3f0a16c171c3a7b1c68c55866463e30007558dd7ea0f36afe29a75f98fe",
        vendored="eth_sri_matharena__src__matharena__parser.py",
        third_party_imports=("matharena",),
        required_imports=("sympy", "regex", "loguru"),
    ),
    PublishedSource(
        repository="eth-sri/matharena",
        revision=_MATHARENA_REVISION,
        path="src/matharena/utils.py",
        sha256="fa7c16f177f41b30140d597e78d0336bac88b9c95234836280444165cff79eaf",
        vendored="eth_sri_matharena__src__matharena__utils.py",
        names=("is_conversation_broken",),
        third_party_imports=("sympy", "loguru"),
    ),
    PublishedSource(
        repository="eth-sri/matharena",
        revision=_MATHARENA_REVISION,
        path="src/matharena/grader.py",
        sha256="ea4af7c475416db0eb3c92a8bb322c53537a7952bccad5c029dd384b39435a64",
        vendored="eth_sri_matharena__src__matharena__grader.py",
        third_party_imports=("matharena",),
        required_imports=("loguru",),
    ),
)

#: The grading configuration MathArena ships for each of the two competitions
#: whose datasets this suite pins. Neither is executed and neither is read by
#: this module -- see `PublishedConfig` for why -- but both are vendored and
#: pinned, because "the binding passes the published config" is a claim about
#: two files and a claim about a file that nobody holds to a digest is a claim
#: about nothing. Each is named on its dataset card's `Repository:` and each is
#: the file MathArena's own runner loads for that competition; the `dataset_path`
#: line inside each one is `MathArena/hmmt_feb_2025` and `MathArena/aime_2026`
#: respectively, which is how the config is matched to the taskset rather than
#: by the file's name.
AIME26_COMPETITION_CONFIG = PublishedConfig(
    repository="eth-sri/matharena",
    revision=_MATHARENA_REVISION,
    path="configs/competitions/aime/aime_2026.yaml",
    sha256="473f860cfb3d0da28fe3886639181b09e9264dc1a15331b8c4593199896cb736",
    vendored="eth_sri_matharena__configs__competitions__aime__aime_2026.yaml",
)

AIME25_COMPETITION_CONFIG = PublishedConfig(
    repository="eth-sri/matharena",
    revision=_MATHARENA_REVISION,
    path="configs/competitions/aime/aime_2025.yaml",
    sha256="1994936c68f87d5d6710e70b804403ae21bcd14f98c512c8972c11c75e067a1f",
    vendored="eth_sri_matharena__configs__competitions__aime__aime_2025.yaml",
)

HMMT25_COMPETITION_CONFIG = PublishedConfig(
    repository="eth-sri/matharena",
    revision=_MATHARENA_REVISION,
    path="configs/competitions/hmmt/hmmt_feb_2025.yaml",
    sha256="abfe6145f6b564675eba5f9510054059b51ca8a3c71900ff0d191f01c72eafec",
    vendored="eth_sri_matharena__configs__competitions__hmmt__hmmt_feb_2025.yaml",
)

HMMT26_COMPETITION_CONFIG = PublishedConfig(
    repository="eth-sri/matharena",
    revision=_MATHARENA_REVISION,
    path="configs/competitions/hmmt/hmmt_feb_2026.yaml",
    sha256="9ba66e1ec44653c24a3a6b3e0bb2bb06c17c5fce2f0e5e4ce8e997d65a8ac6d7",
    vendored="eth_sri_matharena__configs__competitions__hmmt__hmmt_feb_2026.yaml",
)

#: The shortlist's config is the one whose filename does *not* name its dataset:
#: it is `configs/competitions/apex/shortlist_2025.yaml` and the taskset is
#: `MathArena/apex-shortlist`. Matching by the `dataset_path` line inside the file
#: rather than by its name is what the four configs above already do, and this is
#: the entry that would have been matched wrongly by any other rule -- MathArena
#: also publishes a separate `MathArena/apex_2025` under
#: `configs/competitions/apex/apex_2025.yaml`, which is a different benchmark.
APEX_SHORTLIST_COMPETITION_CONFIG = PublishedConfig(
    repository="eth-sri/matharena",
    revision=_MATHARENA_REVISION,
    path="configs/competitions/apex/shortlist_2025.yaml",
    sha256="f6006853c10fabad2d53dd0f4c98d85c6ff8d075da5172caf8ad70b14e77fb3e",
    vendored="eth_sri_matharena__configs__competitions__apex__shortlist_2025.yaml",
)

#: AIME 2026 as MathArena defines and scores it. The taskset pins
#: `MathArena/aime_2026`; `configs/competitions/aime/aime_2026.yaml` at the same
#: revision is that benchmark's own grading configuration and carries exactly one
#: field `extract_and_grade` reads, `strict_parsing: false`. Everything else --
#: `final_answer`, `typed_delimited_answers`, `exact_match_parsing`, `lean` --
#: falls to the published default, which is what `.get(...)` in the vendored
#: function supplies. Passing the config through rather than re-expressing its
#: consequences is the point: `gold_answer_is_list = is_final_answer and "," in
#: gold_answer` is a rule of the published grader, and transcribing it here would
#: be the local variation `exact` exists to exclude.
AIME26_EXACT = PublishedGrader(
    artefact=(
        f"eth-sri/matharena@{_MATHARENA_REVISION} src/matharena/grader.extract_and_grade "
        "under configs/competitions/aime/aime_2026.yaml (strict_parsing: false)"
    ),
    sources=_MATHARENA_SOURCES,
    bind=lambda namespace: _bind_matharena(namespace),
)

#: AIME 2025, through the same grader and the same entry point as AIME 2026.
#:
#: **This entry was `Unavailable` until an audit found the config, and what it
#: was wrong about is worth stating precisely.** The reason it carried said no
#: published grader exists for the AIME. That is true of the Mathematical
#: Association of America, which releases an answer key and nothing executable --
#: and it was already false of this taskset at the moment it was written, because
#: `AIME26_EXACT` above binds MathArena's grader on exactly the criterion
#: `MathArena/aime_2025` also meets. `configs/competitions/aime/aime_2025.yaml`
#: sits beside `aime_2026.yaml` at the same pinned revision, sets the same single
#: flag `extract_and_grade` reads, and covers the same thirty problems. The
#: sentence was not stale; the entry was inconsistent with its neighbour, and a
#: negative claim that contradicts a positive one two lines above it is the kind
#: this table exists to make impossible.
#:
#: **The taskset keeps the dataset it already had, and that is the seam.** The
#: problems are served from `opencompass/AIME2025`, not from MathArena's own
#: `MathArena/aime_2025`, because a grader is a function of a completion and a
#: gold and switching the dataset would re-baseline a taskset that has already
#: reported numbers. Binding across the two is therefore a claim about the golds,
#: and it was checked rather than assumed: 29 of the 30 are byte-identical, and
#: the thirtieth is `336^\circ` where MathArena writes `336` -- opencompass has
#: appended the degree unit to an answer the AIME defines as an integer from 0 to
#: 999. Under this grader's non-strict parsing the two forms are interchangeable
#: in both directions, so the divergence cannot move a verdict.
#: `test_the_aime25_golds_agree_with_matharenas_own` holds both halves.
AIME25_EXACT = PublishedGrader(
    artefact=(
        f"eth-sri/matharena@{_MATHARENA_REVISION} src/matharena/grader.extract_and_grade "
        "under configs/competitions/aime/aime_2025.yaml (strict_parsing: false)"
    ),
    sources=_MATHARENA_SOURCES,
    bind=lambda namespace: _bind_matharena(namespace),
)

#: HMMT February 2025 as MathArena defines and scores it, through the same four
#: files and the same entry point as AIME 2026 above.
#:
#: The taskset pins `MathArena/hmmt_feb_2025`, and the config MathArena ships for
#: that competition is `configs/competitions/hmmt/hmmt_feb_2025.yaml` -- matched
#: to the taskset by its own `dataset_path: MathArena/hmmt_feb_2025` line and not
#: by its filename. Read at the pinned revision it carries the same single field
#: `extract_and_grade` reads, `strict_parsing: false`, and the same four defaults
#: are left to the published `.get(...)`. That is why one binding serves both:
#: not because the two competitions were assumed alike, but because the two
#: published configs were fetched, vendored, digest-pinned and compared, and
#: `tests/test_exact_parity.py` re-checks the comparison against the vendored
#: bytes rather than trusting this sentence.
#:
#: **This entry was `Unavailable` until the LaTeX parser worked here, and the
#: measurement that closed it is worth keeping.** MathArena's parser falls back
#: to `sympy.parsing.latex.parse_latex` whenever `sympy.sympify` cannot read an
#: answer, and that parser refuses to run unless `antlr4-python3-runtime` is
#: 4.11. This environment resolved 4.13.2 through `latex2sympy2-extended`, under
#: which two of the thirty gold answers at the pinned dataset revision --
#: `\frac{9 \sqrt{23}}{23}` and `\sqrt{23}-2 \sqrt{3}`, both spaced before
#: `\sqrt` -- parsed to `None`, and a `None` gold scores every completion for
#: those problems wrong: a 6.7% floor error no model behaviour could avoid, which
#: is why binding was refused rather than degraded. `pyproject.toml` now pins
#: `antlr4-python3-runtime==4.11.0`; measured again at that pin, 0 of the 30 gold
#: answers parse to `None`, and the two named above parse to 1.8766297265136729
#: and 1.331729908174965. `test_the_matharena_latex_fallback_works_in_this_
#: environment` is what fails first if a dependency bump ever reverts that.
HMMT25_EXACT = PublishedGrader(
    artefact=(
        f"eth-sri/matharena@{_MATHARENA_REVISION} src/matharena/grader.extract_and_grade "
        "under configs/competitions/hmmt/hmmt_feb_2025.yaml (strict_parsing: false)"
    ),
    sources=_MATHARENA_SOURCES,
    bind=lambda namespace: _bind_matharena(namespace),
)

#: HMMT February 2026 and the MathArena Apex Shortlist, through the same four
#: files and the same entry point as every MathArena binding above.
#:
#: **Neither needed a new pin, and that is the whole argument for binding them.**
#: `_MATHARENA_REVISION` is unchanged, and at that already-vendored revision the
#: repository ships `configs/competitions/hmmt/hmmt_feb_2026.yaml` and
#: `configs/competitions/apex/shortlist_2025.yaml`. Read there, each carries the
#: same single field `extract_and_grade` reads, `strict_parsing: false`, and
#: leaves the same four -- `final_answer`, `typed_delimited_answers`,
#: `exact_match_parsing`, `lean` -- to the published `.get(...)` default. So one
#: binding serving five tasksets is not an assumption that five competitions are
#: alike; it is five published configs fetched, vendored, digest-pinned and
#: compared, with `tests/test_exact_parity.py` re-checking the comparison against
#: the vendored bytes rather than against this sentence.
#:
#: **What `exact` does and does not claim here.** It claims that the benchmark
#: authors' own grader decided each verdict. It does not claim the resulting
#: number is MathArena's leaderboard number: the completions being graded come
#: from Limite's prompt, which is `suites.AIME_INSTRUCTION` and not the
#: `instruction` line these configs also carry. That divergence is declared on the
#: tasksets themselves; it is repeated here because a reader arriving at a
#: `PublishedGrader` is the reader most likely to mistake a bound grader for a
#: reproduced leaderboard.
HMMT26_EXACT = PublishedGrader(
    artefact=(
        f"eth-sri/matharena@{_MATHARENA_REVISION} src/matharena/grader.extract_and_grade "
        "under configs/competitions/hmmt/hmmt_feb_2026.yaml (strict_parsing: false)"
    ),
    sources=_MATHARENA_SOURCES,
    bind=lambda namespace: _bind_matharena(namespace),
)

APEX_SHORTLIST_EXACT = PublishedGrader(
    artefact=(
        f"eth-sri/matharena@{_MATHARENA_REVISION} src/matharena/grader.extract_and_grade "
        "under configs/competitions/apex/shortlist_2025.yaml (strict_parsing: false)"
    ),
    sources=_MATHARENA_SOURCES,
    bind=lambda namespace: _bind_matharena(namespace),
)

#: Limite's intentional scoring re-baseline, using AIME's engine and defaults.
#: The original OlympiadBench prompt remains in suites.py; this is not the
#: OpenBMB scoring protocol. Keep the shared MathArena bytes and binder intact.
OLYMPIADBENCH_SCORER_REVISION = (
    f"limite-evals:olympiadbench-matharena@{_MATHARENA_REVISION}:strict_parsing=false:v1"
)
OLYMPIADBENCH_EXACT = PublishedGrader(
    artefact=(
        f"eth-sri/matharena@{_MATHARENA_REVISION} src/matharena/grader.extract_and_grade "
        "(strict_parsing: false); MathArena applied to OlympiadBench by Limite "
        "(not the original OpenBMB scoring protocol)"
    ),
    sources=_MATHARENA_SOURCES,
    bind=lambda namespace: _bind_matharena(namespace),
)

#: Why AIME24 and AIME25 have no `exact`, in the words the column has to be read
#: with. This is the D-107 route, and the reason D-108 needed answering at all.
#:
#: It stays true of AIME 2026 as a *competition* and is deliberately not reused
#: for that taskset: `aime26` pins `MathArena/aime_2026`, a dataset whose card
#: names a repository that publishes a grader, where `aime24` and `aime25` pin
#: datasets that ship problems and answers and nothing executable. The reason
#: below names those two datasets for that reason -- it is a claim about what the
#: suite pins, not a claim about every AIME year.
AIME_NO_PUBLISHED_GRADER = (
    "no published grader covers the thirty problems this taskset serves. The competition is "
    "administered by the Mathematical Association of America, which releases the problems and "
    "their integer answers and no grading implementation, and the dataset this suite pins "
    "(HuggingFaceH4/aime_2024) ships problems and answers and nothing executable. MathArena "
    "does publish a grader for AIME 2024 -- the same eth-sri/matharena extract_and_grade that "
    "scores aime25 and aime26 here -- but it publishes it against two fifteen-problem "
    "competitions, MathArena/aime_2024_I and MathArena/aime_2024_II, under two configs, and "
    "there is no published config or dataset for the thirty as one. Binding it would mean "
    "choosing which of the two configs to run the thirty under, which is a local decision "
    "about the published protocol rather than the published protocol. Every other AIME grader "
    "in circulation -- math-verify, prime-rl's verify_boxed_math_answer, "
    "lm-evaluation-harness -- is a de-facto community implementation"
)

#: GSM8K's published grader is deliberately not run against the current task
#: contract. It accepts `#### N`; the prompt, strict scorer, exemplars and
#: headline now require `\boxed{}`. Reporting the old grader would therefore
#: measure compliance with a retired format rather than the declared task.
GSM8K_PUBLISHED_GRADER_INCOMPATIBLE_WITH_BOXED = (
    "the published openai/grade-school-math grader requires a final `#### N` marker, "
    "but this taskset's current prompt, exemplars, strict scorer and headline require a "
    "boxed final answer. Running that grader would measure compliance with the retired "
    "hashed format rather than correctness under the boxed contract"
)

#: BeyondAIME publishes its problems and gold answers, but no executable grader
#: or complete response-extraction/equivalence specification that `exact` could
#: reproduce. The Limite boxed + math-verify contract is therefore `reference`.
BEYONDAIME_NO_PUBLISHED_GRADER = (
    "the upstream BeyondAIME release publishes problems and gold answers but no executable "
    "grader and no complete extraction or equivalence specification. The boxed answer-region "
    "extraction and math-verify equivalence used by this instrument are the Limite `reference` "
    "protocol, not a benchmark-owned artefact that `exact` could reproduce"
)

#: Every taskset in `math-extended` and the published grader bound to its
#: `exact` column. An unavailable entry carries the reason reports display.
TASKSET_EXACT: dict[str, PublishedGrader | Unavailable] = {
    "math500": MATH500_EXACT,
    "gsm8k": Unavailable(GSM8K_PUBLISHED_GRADER_INCOMPATIBLE_WITH_BOXED),
    "aime24": Unavailable(AIME_NO_PUBLISHED_GRADER),
    "aime25": AIME25_EXACT,
    "aime26": AIME26_EXACT,
    "hmmt25": HMMT25_EXACT,
    "hmmt26": HMMT26_EXACT,
    "apex-shortlist": APEX_SHORTLIST_EXACT,
    "olympiadbench": OLYMPIADBENCH_EXACT,
    "beyondaime": Unavailable(BEYONDAIME_NO_PUBLISHED_GRADER),
}


def exact_grader(taskset: str, vendor_root: Path | None = None) -> Grader | Unavailable:
    """The `exact` binding for a taskset: the executed artefact, or its absence.

    `vendor_root` defaults to `limite_evals_core.published.VENDOR_ROOT`, where
    the vendored artefacts live. The explicit form supports integrity tests
    against a temporary copy.

    An unknown taskset raises rather than returning `Unavailable`, because the
    two mean opposite things: `Unavailable` is a checked statement that no
    published grader exists, and a typo must never be able to produce one.
    """
    declared = TASKSET_EXACT.get(taskset)
    if declared is None:
        raise ValueError(
            f"no `exact` declaration for taskset {taskset!r} "
            f"(declared: {', '.join(sorted(TASKSET_EXACT))})"
        )
    if isinstance(declared, Unavailable):
        return declared
    return Grader(
        name=declared.artefact,
        grade=declared.bind(execute_published(declared.sources, vendor_root)),
    )


def _math_extraction(namespace: dict[str, Any]) -> tuple[GradeFn, Callable[[str], str | None]]:
    """MATH's published extraction pair, as `(from a completion, from a gold)`.

    The gold needs one adaptation and it is worth being precise about which. The
    published harness compares two *extracted* answers: it applies
    `remove_boxed(last_boxed_only_string(...))` to the model output, and to the
    reference solution as well when the solution is a full worked one. The
    tasksets here already ship the bare answer, so the same published pair is
    applied to the gold only when the gold is boxed, and a bare gold is passed
    through. Nothing else is adapted, and in particular no unboxed model output
    is rescued: `remove_boxed(None)` returns `None`, and both graders below score
    a `None` answer wrong, which is the published behaviour of each.
    """
    last_boxed_only_string = namespace["last_boxed_only_string"]
    remove_boxed = namespace["remove_boxed"]

    def from_completion(completion: str) -> str | None:
        return remove_boxed(last_boxed_only_string(completion))

    def from_gold(gold: str) -> str | None:
        boxed = last_boxed_only_string(gold)
        return remove_boxed(boxed) if boxed is not None else gold

    return from_completion, from_gold


def _bind_math500(namespace: dict[str, Any]) -> GradeFn:
    """prm800k's `grade_answer` over MATH's extraction: the subset's own grader.

    `grade_answer(given_answer, ground_truth)` takes the model's answer first and
    the reference second -- the order prm800k's README recommends -- and that
    order is the only thing adapted here. What it then does is its own: MATH's
    normalisation copied into `math_normalize`, and where that disagrees, a
    second normalisation plus a sympy equality check, held back from rescuing an
    unreduced fraction or a non-integer answer to an integer gold.
    """
    from_completion, from_gold = _math_extraction(namespace)
    grade_answer = namespace["grade_answer"]

    def grade(completion: str, gold: str) -> bool:
        return bool(grade_answer(from_completion(completion), from_gold(gold)))

    return grade


def _bind_math500_hendrycks(namespace: dict[str, Any]) -> GradeFn:
    """MATH's `is_equiv` over the same extraction, in the order its script uses.

    Nothing scores under this. It exists so the parity test can put the two
    published graders side by side on the same completions.
    """
    from_completion, from_gold = _math_extraction(namespace)
    is_equiv = namespace["is_equiv"]

    def grade(completion: str, gold: str) -> bool:
        return bool(is_equiv(from_completion(completion), from_gold(gold)))

    return grade


def _bind_matharena(namespace: dict[str, Any]) -> GradeFn:
    """MathArena's `extract_and_grade`, called on the shapes its runner passes.

    Three adaptations, all shape and none verdict:

    * The completion becomes a one-message conversation. `extract_and_grade`
      reads `messages[-1]["content"]` and hands the rest to
      `is_conversation_broken`, which asks only that the last message is an
      assistant response -- so one assistant message is the smallest input that
      is not "broken" and reaches exactly the same line.
    * `output_tokens` is 0. It feeds `check_output_length`, which returns False
      below 1000 and whose only effect is which warning is attached. This
      binding returns the verdict and drops the warning, so the value cannot
      reach a number; 0 is the value that makes that visible rather than a
      plausible-looking one that hides it.
    * `competition_config` passes only `strict_parsing: false`. The pinned
      AIME25/26 and HMMT25 YAML configurations carry that same flag and no other
      setting this function reads, as checked by the competition-config parity
      test. Every other flag uses the published `.get(...)` default.
      OlympiadBench deliberately reuses this configuration under Limite's
      scoring re-baseline; it is not a published OlympiadBench configuration.

    `strict_parsing: false` is not a detail. It is what turns the extraction
    into a two-step one: the boxed span if there is one -- `\\boxed` or `\\fbox`,
    the published regex takes both -- and otherwise the last bare integer
    anywhere in the completion. That is the divergence from `reference` this
    repository predicted before the grader was found, and
    `tests/test_exact_parity.py` measures it.
    """
    extract_and_grade = namespace["extract_and_grade"]

    def grade(completion: str, gold: str) -> bool:
        _, correct, _ = extract_and_grade(
            [{"role": "assistant", "content": completion}],
            0,
            gold,
            {"strict_parsing": False},
        )
        return bool(correct)

    return grade
