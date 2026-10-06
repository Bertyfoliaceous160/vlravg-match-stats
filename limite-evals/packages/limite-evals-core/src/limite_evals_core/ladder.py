"""The extraction ladder: how an answer is recovered from a completion.

Three rungs, and **every run computes all three**. The format policy chosen for a
run selects which rung the reward carries; it never selects which rungs are
computed, because the gap between them is the measurement we actually want. A
model that solves a problem but cannot box its answer scores 0 strict and 1
lenient, and that difference is the format-compliance signal that separates a
pretrained checkpoint from a post-trained one.

| rung | forgives | use |
|---|---|---|
| `strict` | nothing | the headline metric, at every stage |
| `lenient` | missing or malformed `\\boxed{}`, an unclosed think block | diagnosis of format loss |
| `permissive` | the answer anywhere at all, reasoning included | an upper bound, never a headline |

Every rung goes through the same equivalence check, and that check has a time
bound it can hit. When it does, the answer scores wrong -- so each rung a
completion reached is a lower bound rather than a verdict, and `LadderResult`
carries `comparison_timeout` to say which completions that applies to.

`strict` reproduces `verifiers.v1.utils.score.verify_boxed_math_answer`, which is
what `math500_v1`, `aime24_v1` and `aime25_v1` actually call. **Not** the
`verify.py` sitting beside each of those tasksets: that file is dead code for all
three, and it differs from the live function -- it neither strips the extracted
prediction nor unwraps a boxed gold. Pinning against it would have guaranteed
agreement with code that never runs.

Being bit-identical to the live function is the only thing that makes an offline
number comparable with what the reinforcement learning run scores in flight.
`tests/test_strict_parity.py` imports that function and asserts agreement
directly, so it cannot drift and never skips; changing anything in
`extract_boxed` or `strict_answer` breaks the comparability guarantee and must
fail that test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from limite_evals_core.equivalence import compare

#: Answer formats represented in stored samples. Current maths tasksets use
#: `boxed`.
AnswerFormat = Literal["boxed", "hashed"]

BOXED = "\\boxed{"

#: Upper bound on how many candidate answers the permissive rung will test. The
#: last N are kept, because an answer that appears at all almost always appears
#: near the end. A completion offering more than this is pathological, and the
#: rung is a diagnostic upper bound rather than a reported score, so truncating
#: its candidate list cannot silently deflate a headline number.
PERMISSIVE_CANDIDATE_CAP = 64

_ANSWER_PHRASE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)",
    re.IGNORECASE,
)
_INLINE_MATH = re.compile(r"\$([^$]+)\$|\\\((.+?)\\\)")
_NUMERIC = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True)
class LadderResult:
    """One completion scored at every rung, with the string each rung recovered.

    `comparison_timeout` says whether any of the equivalence checks behind these
    rungs ran out of time. It qualifies them rather than joining them: a timeout
    scores wrong, so a rung on a completion that saw one is a lower bound on what
    that completion would have scored had the comparator finished. It is last and
    defaulted because it is the only field here that describes the verifier
    rather than the completion.

    `format_not_evaluable` qualifies `format_ok` the same way, and for a reason
    that only became reachable once the reasoning delimiters survived the decode:
    a completion whose think block ran into the token cap has **no answer region
    at all**, so "was the answer where the reference wanted it" is a question
    about text that was never written. `format_ok` is False on such a row for
    schema continuity and aggregation drops it from the denominator instead of
    counting it as a formatting failure. See `format_not_evaluable`.
    """

    strict: bool
    lenient: bool
    permissive: bool
    format_ok: bool
    strict_answer: str | None
    lenient_answer: str | None
    comparison_timeout: bool = False
    format_not_evaluable: bool = False

    def at(self, policy: str) -> bool:
        """The rung named by a format policy. Raises on an unknown policy rather
        than defaulting, so a typo in a config cannot silently change a metric."""
        if policy not in ("strict", "lenient", "permissive"):
            raise ValueError(f"unknown format policy: {policy!r}")
        return bool(getattr(self, policy))


def think_unclosed(text: str) -> bool:
    """Whether a think block was opened and never closed -- i.e. the completion
    stopped mid-reasoning. Strict treats this as a failure; lenient does not."""
    return "<think>" in text and "</think>" not in text


def strip_think(text: str) -> str:
    """Everything after the last `</think>`, or the whole string if there is none."""
    return text.split("</think>")[-1]


def format_not_evaluable(completion: str, *, truncated: bool) -> bool:
    """Whether this completion has no answer region for a format lens to read.

    True for exactly one shape: a think block that was opened, never closed, and
    ran into the generation cap. There the model was still reasoning when the
    budget ended, so nothing it wrote was ever offered as an answer -- and
    `format_ok` on such a row would report a formatting decision the model never
    got to make. Aggregation drops it from the `format_ok` denominator rather
    than counting it as a failure.

    An unclosed think block in a completion that **stopped by itself** is a
    different fact and stays a format failure: the model had the budget to close
    the block and present an answer, and did not. The distinction is `truncated`,
    which is the run's own record of how the rollout ended, exactly as it is for
    the taxonomy's `C` against `E`.

    This is deliberately narrower than "the answer region is empty". A completion
    that closed its think block and then wrote nothing useful *did* reach the
    region and formatted nothing there, which `format_ok` is entitled to call a
    failure.
    """
    return truncated and think_unclosed(completion)


def extract_boxed(text: str) -> str:
    """Content of the last ``\\boxed{...}`` in `text`, or `""` if there is none.

    Brace-depth matched, so `\\boxed{\\frac{1}{2}}` yields `\\frac{1}{2}` rather
    than stopping at the first closing brace. An unbalanced expression yields
    `""`: strict refuses to guess where the answer ended.
    """
    start = text.rfind(BOXED)
    return _boxed_at(text, start) if start != -1 else ""


def _boxed_at(text: str, start: int) -> str:
    """Content of the `\\boxed{` occurrence beginning at `start`, `""` if unbalanced."""
    i, depth = start + len(BOXED), 1
    while i < len(text) and depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[start + len(BOXED) : i - 1] if depth == 0 else ""


def strict_answer(completion: str) -> str | None:
    """The answer under strict rules, or None when the format itself failed.

    Stripped, because the reference strips: `\\boxed{ 42 }` is the answer 42 and
    the surrounding space is typesetting. What survives the strip is what goes to
    the equivalence check, so a completion that spaced its answer out is not
    scored against a different string than one that did not.
    """
    if think_unclosed(completion):
        return None
    return extract_boxed(strip_think(completion)).strip() or None


def lenient_answer(completion: str) -> str | None:
    """The answer under a fallback ladder, or None when nothing looks like one.

    Tried in order, most to least trustworthy: a well-formed boxed answer; an
    unterminated `\\boxed{` running to the end of a truncated completion; an
    explicit "the answer is X" phrase; inline math on the last non-empty line;
    a bare number on the last non-empty line.

    Unlike strict, an unclosed think block is not fatal here -- it usually means
    the completion hit the token limit, and the tail may still carry an answer.
    """
    tail = strip_think(completion)

    boxed = extract_boxed(tail)
    if boxed:
        return boxed

    # A truncated completion can open `\boxed{` and never close it. Strict scores
    # that 0; lenient takes what is there.
    start = tail.rfind(BOXED)
    if start != -1:
        remainder = tail[start + len(BOXED) :].strip().rstrip("}").strip()
        if remainder:
            return remainder

    line = _last_nonempty_line(tail)
    if line is None:
        return None

    phrase = _ANSWER_PHRASE.search(line)
    if phrase:
        candidate = phrase.group(1).strip().rstrip(".").strip()
        if candidate:
            return candidate

    inline = _inline_math_spans(line)
    if inline:
        return inline[-1]

    numeric = _NUMERIC.findall(line)
    if numeric:
        return numeric[-1].replace(",", "")

    return None


def permissive_candidates(completion: str) -> list[str]:
    """Every plausible answer anywhere in the completion, reasoning included.

    Capped at `PERMISSIVE_CANDIDATE_CAP`, keeping the last candidates.
    """
    candidates: list[str] = []
    for match in re.finditer(re.escape(BOXED), completion):
        boxed = _boxed_at(completion, match.start())
        if boxed:
            candidates.append(boxed)
    candidates.extend(_inline_math_spans(completion))
    candidates.extend(value.replace(",", "") for value in _NUMERIC.findall(completion))

    seen: set[str] = set()
    unique = []
    for candidate in reversed(candidates):
        stripped = candidate.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(stripped)
        if len(unique) >= PERMISSIVE_CANDIDATE_CAP:
            break
    return list(reversed(unique))


def strict_answer_hashed(completion: str) -> str | None:
    """The answer under historical GSM8K: the last `#### ...` in the answer region.

    Historical GSM8K rows used a different reference verifier from the boxed tasksets
    (`verifiers` submodule, `environments/gsm8k_v1/gsm8k_v1/verify.py`), and its
    system prompt asks for `#### 42` rather than a boxed answer. Since decision 3
    binds strict to *the taskset's own* reference, the strict rung has to follow
    that split rather than force one answer format onto both.

    **The think handling is `strict_answer`'s, not the reference's, and that is a
    declared departure.** The reference was written for a model that emits no
    reasoning delimiters, so it searches the whole string; against a model that
    does, that reads the reasoning as the answer. Two consequences were measured
    on real completions and neither is a rounding error. A `####` written inside
    the reasoning and then revised is scored as the answer. And the regex is
    line-greedy, so `#### 18 here</think>` -- a marker the model wrote *while*
    thinking -- extracts `18 here</think>`, a string carrying a literal
    delimiter into the equivalence check. That is contamination of the strict
    lens by text the model never presented as an answer, and reproducing it
    bit-for-bit would only make an offline number match an in-run number that is
    wrong the same way.

    So both of `strict_answer`'s rules apply here: an unclosed think block is a
    format failure, and what is searched is the region after the last
    `</think>`. On a completion carrying no delimiters -- every pre-E42-fix
    rollout, and every non-reasoning model -- `strip_think` is the identity and
    this is the reference's own behaviour unchanged.
    """
    if think_unclosed(completion):
        return None
    matches = re.findall(r"####\s*(.+)", strip_think(completion))
    return matches[-1].strip() if matches else None


def score(
    completion: str,
    gold: str,
    *,
    answer_format: AnswerFormat = "boxed",
    truncated: bool = False,
) -> LadderResult:
    """Score one completion against one gold answer at every rung.

    `answer_format` selects which reference the strict rung reproduces: `boxed`
    for current maths tasksets, `hashed` for historical GSM8K rows.

    `truncated` reaches exactly one field, `format_not_evaluable`, and no rung.
    It is the run's own record of how the rollout ended, and what it decides is
    whether `format_ok` had a question to answer at all -- never what any lens
    recovered. What a truncated completion *scores* stays where it has always
    been decided, once, at the aggregation boundary; see `TruncationPolicy`.
    Defaulted to False so a caller with no such record scores what it always did.
    """
    if answer_format == "hashed":
        return _score_hashed(completion, gold, truncated=truncated)

    strict = strict_answer(completion)
    lenient = lenient_answer(completion)
    # The reference unwraps a boxed gold before comparing, so a taskset that
    # ships `\boxed{42}` as its answer is compared against 42 rather than
    # against the wrapper. Ours ship bare strings today; this keeps a taskset
    # that does not from silently scoring zero on every rollout.
    gold = (extract_boxed(gold) or gold).strip()

    matches = _Matcher(gold)
    strict_correct = bool(strict) and matches(strict)
    lenient_correct = strict_correct or (bool(lenient) and matches(lenient))
    permissive_correct = lenient_correct or any(
        matches(candidate) for candidate in permissive_candidates(completion)
    )

    return LadderResult(
        strict=strict_correct,
        lenient=lenient_correct,
        permissive=permissive_correct,
        format_ok=strict is not None,
        strict_answer=strict,
        lenient_answer=lenient,
        comparison_timeout=matches.timed_out,
        format_not_evaluable=format_not_evaluable(completion, truncated=truncated),
    )


def _score_hashed(completion: str, gold: str, *, truncated: bool = False) -> LadderResult:
    """GSM8K's rungs.

    The reference falls back to the *whole completion* when no `####` marker is
    present, so strict here is more forgiving than the boxed one. That asymmetry
    is inherited deliberately: reproducing the reference is what decision 3 asks
    for, and softening it to match the boxed rung would put a scoring difference
    between our GSM8K number and the in-run one.

    **What the fallback is handed is the answer region, not the whole text.** The
    reference's own fallback predates reasoning models, and against a completion
    that carries delimiters it hands the grader a string containing a literal
    `</think>` -- so the strict lens would be comparing the gold against the
    model's reasoning plus its markup. The lens keeps its shape (no marker, fall
    back to the surrounding text) and the text it falls back to is what the
    strict lens is entitled to read: `strip_think(completion)`. On a completion
    with no delimiters the two are the same string, which is every rollout the
    reference was written against.

    An unclosed think block therefore has **nothing to fall back to**: there is no
    answer region, so the strict lens fails rather than being handed the whole
    abandoned reasoning. Falling back there would grade the model's scratch work
    under the name of its answer, and it is also the one shape where the boxed and
    hashed lenses had drifted apart -- boxed has always scored an unclosed block
    zero.

    `permissive` is unchanged and still scans the **whole** completion, reasoning
    included. That is its published semantics -- the `answer_anywhere` waiver is
    defined as "the answer appeared anywhere at all" -- and narrowing it here
    would silently redefine a column already quoted.
    """
    matches = _Matcher(gold)
    strict = strict_answer_hashed(completion)
    if think_unclosed(completion):
        strict_correct = False
    else:
        strict_correct = matches(strict if strict is not None else strip_think(completion))
    lenient = strict if strict is not None else lenient_answer(completion)
    lenient_correct = strict_correct or (bool(lenient) and matches(lenient))
    permissive_correct = lenient_correct or any(
        matches(candidate) for candidate in permissive_candidates(completion)
    )
    return LadderResult(
        strict=strict_correct,
        lenient=lenient_correct,
        permissive=permissive_correct,
        format_ok=strict is not None,
        strict_answer=strict,
        lenient_answer=lenient,
        comparison_timeout=matches.timed_out,
        format_not_evaluable=format_not_evaluable(completion, truncated=truncated),
    )


class _Matcher:
    """One gold, every candidate compared against it, any timeout remembered.

    A callable rather than a loop over a collected candidate list, because the
    rungs short-circuit on purpose: `lenient` is not compared when `strict`
    already matched, and `permissive` stops at its first hit. Comparing eagerly
    to make the timeouts easy to collect would spend `math-verify` on up to 64
    candidates per rollout that today are never looked at. So the rungs keep
    their exact evaluation order, and this remembers what happened along it.
    """

    def __init__(self, gold: str) -> None:
        self._gold = gold
        self.timed_out = False

    def __call__(self, candidate: str) -> bool:
        outcome = compare(self._gold, candidate)
        self.timed_out = self.timed_out or outcome.timed_out
        return outcome.equal


def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return None


def _inline_math_spans(text: str) -> list[str]:
    spans = []
    for dollar, paren in _INLINE_MATH.findall(text):
        content = (dollar or paren).strip()
        if content:
            spans.append(content)
    return spans
