"""The A--T failure-mode taxonomy: which shape a completion took, case by case.

A run already reports *how many* rollouts a lens accepted. It does not report
*why* the rest were refused, and the two questions have different answers: a
completion that never closed its think block, one that ran its `\\boxed{` into
the token cap, and one that put the right answer inside its reasoning and then
gave up are three different faults with three different fixes, and all three
read as one number today.

This module assigns every scored rollout exactly one of twenty cases. It is an
**observer**: nothing here grades anything, no case changes a score, and the
counts are reported beside the existing columns rather than inside them. What it
adds is a partition of the rollouts a lens refused.

## The table

The cases and their lens expectations are the user's, frozen. `strict` and
`permissive` are the two `limite_evals_core.ladder` rungs the table was written
against: the answer where the reference demands it, and the answer anywhere at
all.

What a run *counts* under those two names is whatever that taskset reports under
them, which `aggregate.case_counts` reads through `aggregate.scored` exactly as
every column in the report does. On a protocol run that means the ladder rung
for `strict`, which no protocol is named after, and the `permissive` protocol
for `permissive`, which relaxes a published grader rather than the rung. The two
are not the same artefact and this repository does not let one stand in for the
other silently -- but they answer the same question the table asks, "was the
answer reachable from anywhere in this text", and on the 6,015 rollouts of
`res339-ev0-baseline-002` they disagreed on two.

| case | shape | strict | permissive |
|---|---|---|---|
| `A` | well-formed box, think closed | yes | yes |
| `M` | multiple boxes, the last one right | yes | yes |
| `Q` | thousands commas in the box, `1,000` | yes | yes |
| `R` | gold already boxed | yes | yes |
| `B` | well-formed box, think never closed | no | yes |
| `C` | box left unterminated by the token cap | no | yes |
| `D` | unterminated box and unclosed think | no | yes |
| `E` | box with unbalanced braces | no | yes |
| `F` | empty box, `\\boxed{}` | no | yes |
| `G` | `\\fbox` instead of `\\boxed` | no | yes |
| `H` | no box, explicit answer phrase | no | yes |
| `I` | no box, inline maths on the last line | no | yes |
| `J` | no box, bare number on the last line | no | yes |
| `L` | think left open, number inside the reasoning | no | yes |
| `K` | answer only inside a closed think block | no | yes |
| `N` | multiple boxes, the last one wrong | no | yes |
| `O` | answer phrase not on the last line | no | yes |
| `P` | scratch number on the last line | no | yes |
| `T` | right box inside a closed think, then gives up | no | yes |
| `S` | nothing numeric anywhere | no | no |

"Yes" is what the case is expected to score **when the answer it carries is
right**. A case is a statement about the shape a completion took, so a
well-formed box holding the wrong number is still `A`: the box was well formed.
Correctness enters the classification at exactly one place, `M` against `N`,
because that is the only pair whose names distinguish on it.

That is what makes the reconciliation in `aggregate.case_counts` an assertion
rather than a tautology. Two implications are being claimed, and either can
fail on real data:

- a rollout whose case is not one of `A`, `M`, `Q`, `R` cannot score at the
  strict lens;
- a rollout whose case is `S` cannot score at the permissive lens.

## The strict column is boxed-only

**Every case in the table is a statement about where a `\\boxed{}` answer sits**,
so the strict column predicts a strict lens that goes looking for one. Historical
GSM8K rows used a reference that asked for `#### 42` and fell back to the whole
completion when there was no marker. On such a stored row the shape a completion
took and what strict recovered are simply unrelated -- and measurably so. Against
`res339-ev0-baseline-002`, 132 of the 541 GSM8K rollouts strict accepted came
from cases the table calls "no", every one of them because a correct `####`
marker (117) or the whole-completion fallback (15) reached an answer no box had
to hold.

So the strict column applies where `answer_format` is `boxed` and **states
nothing anywhere else**. `strict_expectation` returns `None` on a historical
hashed row rather than `False`, and a `None` is not a prediction that failed --
it is the absence of one, which is the same distinction
`ProtocolSample.correct` draws for a grader that could not be bound. A hashed
row still gets its full partition, still reports what strict observed on each
case, and still carries the permissive prediction, which reconciled exactly on
the same run.

The permissive column needs no such scoping. It asks whether the answer was
reachable from anywhere in the text, and that question does not depend on the
notation the reference expects the answer in.

## Priority order

Classification is **total and deterministic**: every completion reaches exactly
one case, and the first rule below that matches wins. Several cases overlap on
real text -- a completion can hold two boxes, thousands commas *and* a boxed
gold -- so the order is part of the definition rather than an implementation
detail.

1. `S`, when the extractor finds no candidate answer anywhere. Read
   operationally, through `ladder.permissive_candidates`, so a boxed
   `\\text{...}` answer is a candidate and does not fall in here; "numeric" in
   the table's wording means "something a lens could have compared".
2. **Think block left open.** Nothing after it can be an answer region, because
   `ladder.strip_think` returns the whole completion when there is no
   `</think>`, so the box cases split on the box and everything else is `L`:
   `D` (last `\\boxed{` unterminated), `B` (well formed), `F` (`\\boxed{}`),
   `G` (an `\\fbox` instead), then `L`.
3. **The strict shape** -- think closed and the last box in the answer region
   well formed and non-empty, which is exactly `ladder.strict_answer` returning
   a string. Within it, most consequential first: `M`/`N` (more than one box,
   split on whether the last one is right, because the two disagree at the
   strict rung), then `R` (the gold arrived boxed), then `Q` (thousands commas),
   then `A`.
4. **A box that failed**, in the answer region: `C` (unterminated *and* the
   rollout hit its generation cap), `E` (unterminated without one -- unbalanced
   braces), `F` (`\\boxed{}`), then `G`.
5. **The lenient ladder's own order** over the last non-empty line, so the case
   names what the fallback saw: `H` (answer phrase), `I` (inline maths), `J`
   (a bare number).
6. **What the lenient ladder missed**: `O` (an answer phrase, but not on the
   last line), `P` (a number in the answer region that nothing marks as the
   answer), `T` (a box inside the closed think block), `K` (an answer inside
   the closed think block that was never boxed).

Three readings in that order are choices the table does not make on its own,
and all three are between cases that share both lens expectations, so none of
them can move a reconciliation:

- **`C` against `E`.** The two have one textual signature -- a `\\boxed{` whose
  braces never close -- and are distinguished by `truncated`, since "left
  unterminated by the token cap" is a fact about how the rollout ended and
  "unbalanced braces" is what the same text means when it ended by itself.
- **`J` against `P`.** `J` is the last non-empty line *being* a number; `P` is a
  number in the answer region that nothing presents as the answer. `P` is
  therefore the residual of the two rather than strictly a last-line rule.
- **`T` against `K`.** `T` is read as "the abandoned answer was boxed" and `K`
  as "it was not". The table's `T` says "right box", but correctness enters the
  classification only at `M`/`N`, so the box is what is checked here and
  whether it was right is left to the lens.

## What is not classified

A taskset scored by a constraint checker has no answer to extract and gets no
case at all -- `None`, the same absence its empty `protocols` and its defaulted
ladder booleans already record. A case of `S` there would read as a measured
fact about a completion nothing tried to extract an answer from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from limite_evals_core.equivalence import equivalent

# Brace matching is imported rather than rewritten. A second implementation of
# where a `\boxed{` ends would be a second extractor, and the whole point of
# this module is to describe what the real one sees; `ladder.py` is read-only by
# decision, so the private name is the only way to reach it without retyping it.
from limite_evals_core.ladder import (
    BOXED,
    _boxed_at,  # noqa: PLC2701
    extract_boxed,
    permissive_candidates,
    strip_think,
    think_unclosed,
)

Case = Literal[
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
]

FBOX = "\\fbox{"

#: A number written with thousands separators, which is what case `Q` is about.
#: At least one comma, and a group of exactly three digits after each.
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)")

#: The lenient ladder's own answer-phrase pattern. Imported by value rather than
#: by reference because `ladder._ANSWER_PHRASE` is anchored to a single line
#: there and `O` needs to ask the same question of a whole region.
_ANSWER_PHRASE = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)", re.IGNORECASE)
_INLINE_MATH = re.compile(r"\$([^$]+)\$|\\\((.+?)\\\)")
_NUMERIC = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
#: A last line that *is* a number, which is what separates `J` from `P`. Wrapped
#: maths and trailing punctuation are allowed around it, because `$42$.` on a
#: line of its own is the same statement as `42`.
_BARE_NUMBER_LINE = re.compile(
    r"^[$\\()\[\]\s]*[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?[$\\()\[\]\s.,;:!]*$"
)


@dataclass(frozen=True)
class CaseSpec:
    """One case: what it describes, and what each rung is expected to do with it."""

    case: Case
    description: str
    strict: bool
    permissive: bool


#: The frozen table, in the user's own order: the cases a strict rung can accept
#: first, then the ones only a permissive rung can, then the one neither can.
#: Reports render it in this order because the grouping is the point of it.
TAXONOMY: tuple[CaseSpec, ...] = (
    CaseSpec("A", "well-formed box, think closed", True, True),
    CaseSpec("M", "multiple boxes, the last one right", True, True),
    CaseSpec("Q", "thousands commas in the box, `1,000`", True, True),
    CaseSpec("R", "gold already boxed", True, True),
    CaseSpec("B", "well-formed box, think never closed", False, True),
    CaseSpec("C", "box left unterminated by the token cap", False, True),
    CaseSpec("D", "unterminated box and unclosed think", False, True),
    CaseSpec("E", "box with unbalanced braces", False, True),
    CaseSpec("F", "empty box, `\\boxed{}`", False, True),
    CaseSpec("G", "`\\fbox` instead of `\\boxed`", False, True),
    CaseSpec("H", "no box, explicit answer phrase", False, True),
    CaseSpec("I", "no box, inline maths on the last line", False, True),
    CaseSpec("J", "no box, bare number on the last line", False, True),
    CaseSpec("L", "think left open, number inside the reasoning", False, True),
    CaseSpec("K", "answer only inside a closed think block", False, True),
    CaseSpec("N", "multiple boxes, the last one wrong", False, True),
    CaseSpec("O", "answer phrase not on the last line", False, True),
    CaseSpec("P", "scratch number on the last line", False, True),
    CaseSpec("T", "right box inside a closed think, then gives up", False, True),
    CaseSpec("S", "nothing numeric anywhere", False, False),
)

CASES: tuple[Case, ...] = tuple(spec.case for spec in TAXONOMY)
SPECS: dict[Case, CaseSpec] = {spec.case: spec for spec in TAXONOMY}

#: The cases the strict rung is expected to be able to accept, and the ones the
#: permissive rung is. Derived from the table rather than restated beside it, so
#: a change to one row cannot leave two sets disagreeing about it.
STRICT_YES: frozenset[Case] = frozenset(spec.case for spec in TAXONOMY if spec.strict)
PERMISSIVE_YES: frozenset[Case] = frozenset(spec.case for spec in TAXONOMY if spec.permissive)


#: The answer formats the strict column was written against. Every case in the
#: table describes where a `\boxed{}` answer sits, so the column predicts a
#: strict lens that goes looking for one; see the module docstring.
STRICT_DOMAIN: frozenset[str] = frozenset({"boxed"})


def expects(case: Case, rung: str) -> bool:
    """The table's own row for `case` at `rung`, before any taskset is in view.

    This is the frozen table read literally. What a *taskset* is entitled to
    predict is `expectation`, which is this scoped to the answer format the
    column was written for. Raises on an unknown rung rather than defaulting,
    for the reason `LadderResult.at` does.
    """
    if rung not in ("strict", "permissive"):
        raise ValueError(f"the taxonomy states expectations for strict and permissive, not {rung!r}")
    return getattr(SPECS[case], rung)


def expectation(case: Case, rung: str, answer_format: str | None) -> bool | None:
    """What the table predicts for `case` at `rung` on a taskset scored under
    `answer_format`, or `None` where it predicts nothing at all.

    `None` is the whole point of this function and is not `False` wearing a
    different name. A taskset whose strict lens does not read a box is one the
    strict column was never about, so it has no prediction to fail -- and a
    `False` there would report every rollout strict accepted as a violation of
    a claim nobody made. It is the distinction `ProtocolSample.correct` already
    draws between a grader that said no and a grader that could not be bound.

    Only `strict` is scoped. `permissive` asks whether the answer was reachable
    from anywhere in the text, which does not depend on the notation the
    taskset's reference wants it written in.
    """
    if rung == "strict" and answer_format not in STRICT_DOMAIN:
        return None
    return expects(case, rung)


def classify(completion: str, gold: str, *, truncated: bool = False) -> Case:
    """The one case this completion falls in. See the module docstring for the order.

    `gold` is read for exactly two things: whether it arrived boxed (`R`) and
    whether a completion offering several boxes ended on the right one (`M`
    against `N`). `truncated` separates `C` from `E`, and is the run's own
    record of how the rollout ended rather than anything guessed from the text.
    """
    if not permissive_candidates(completion):
        return "S"

    if think_unclosed(completion):
        return _unclosed_think_case(completion)

    tail = strip_think(completion)
    kind, content = _last_box(tail)

    if kind == "filled":
        return _strict_shape_case(tail, content, gold)
    if kind == "unterminated":
        return "C" if truncated else "E"
    if kind == "empty":
        return "F"
    if FBOX in completion:
        return "G"
    return _no_box_case(completion, tail)


def _unclosed_think_case(completion: str) -> Case:
    """A completion that stopped mid-reasoning. There is no answer region.

    `strip_think` returns the whole string when there is no `</think>`, so every
    last-line rule would be reading the reasoning; the table has one case for
    that and it is `L`. Only the box cases survive here, because a box is a
    claim about the answer wherever it sits.
    """
    kind, _ = _last_box(completion)
    if kind == "unterminated":
        return "D"
    if kind == "filled":
        return "B"
    if kind == "empty":
        return "F"
    return "G" if FBOX in completion else "L"


def _strict_shape_case(tail: str, content: str, gold: str) -> Case:
    """Think closed and the last box well formed: the shapes strict can accept.

    `M` and `N` are tested first because they are the only pair here that
    disagree at the strict rung, so a case that absorbed an `N` into `Q` or `A`
    would describe a rollout strict refused as one it accepted.
    """
    if len(_boxes(tail)) > 1:
        return "M" if equivalent(_unwrap_gold(gold), content.strip()) else "N"
    if extract_boxed(gold).strip():
        return "R"
    if _THOUSANDS.search(content):
        return "Q"
    return "A"


def _no_box_case(completion: str, tail: str) -> Case:
    """No usable box anywhere: what the fallbacks saw, then what they missed."""
    line = _last_nonempty_line(tail)
    if line is not None:
        if _ANSWER_PHRASE.search(line):
            return "H"
        if _inline_math(line):
            return "I"
        if _BARE_NUMBER_LINE.match(line):
            return "J"
    if _ANSWER_PHRASE.search(tail):
        return "O"
    if _NUMERIC.search(tail) or _inline_math(tail):
        return "P"
    # Nothing in the answer region, but `S` was ruled out, so the candidate the
    # extractor found is inside the closed think block.
    return "T" if _boxes(completion) else "K"


def _last_box(region: str) -> tuple[str, str]:
    """The last `\\boxed{` in `region` as `(kind, content)`.

    `kind` is one of `none`, `unterminated` (the braces never close, which is
    what a token cap leaves behind), `empty` (`\\boxed{}` or whitespace) and
    `filled`. `extract_boxed` collapses the middle two into `""`, and the table
    needs them apart.
    """
    start = region.rfind(BOXED)
    if start == -1:
        return "none", ""
    content = _boxed_at(region, start)
    if content.strip():
        return "filled", content
    # `_boxed_at` returns `""` for an unbalanced expression and for a genuinely
    # empty one alike. A balanced pair leaves the scan at depth zero, which is
    # the same thing as the content round-tripping through a re-scan of the
    # closed text, so the brace count is asked directly.
    return ("empty" if _balanced(region, start) else "unterminated"), content


def _balanced(region: str, start: int) -> bool:
    depth = 1
    for index in range(start + len(BOXED), len(region)):
        depth += (region[index] == "{") - (region[index] == "}")
        if depth == 0:
            return True
    return False


def _boxes(region: str) -> list[str]:
    """Every well-formed, non-empty boxed span in `region`, in order."""
    return [
        content
        for match in re.finditer(re.escape(BOXED), region)
        if (content := _boxed_at(region, match.start()).strip())
    ]


def _unwrap_gold(gold: str) -> str:
    """`\\boxed{42}` shipped as a gold answer is the answer 42, as every rung reads it."""
    return (extract_boxed(gold) or gold).strip()


def _inline_math(text: str) -> bool:
    return any((dollar or paren).strip() for dollar, paren in _INLINE_MATH.findall(text))


def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return None
