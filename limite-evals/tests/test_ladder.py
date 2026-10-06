"""The lenient and permissive rungs, and the monotonicity that makes them readable.

The rungs are nested by construction: anything strict accepts, lenient accepts,
and anything lenient accepts, permissive accepts. That is what lets a report
quote all three side by side and have the differences mean something -- strict to
lenient is answer-formatting loss, lenient to permissive is the model knowing the
answer but not presenting it as one.
"""

from __future__ import annotations

import pytest

from limite_evals_core.ladder import (
    PERMISSIVE_CANDIDATE_CAP,
    lenient_answer,
    permissive_candidates,
    score,
    strict_answer_hashed,
)

LENIENT_VECTORS = [
    # A truncated completion that opened a box and never closed it.
    ("So the answer is \\boxed{42", "42"),
    # An explicit phrase, no box at all.
    ("Working through it, the answer is 42", "42"),
    ("Final answer: 42", "42"),
    # Inline math on the last line.
    ("Some reasoning.\nTherefore $x = 42$", "x = 42"),
    # A bare number on the last line, the weakest rung.
    ("Some reasoning.\n42", "42"),
    # Thousands separators are normalised away.
    ("Some reasoning.\n1,024", "1024"),
    # Nothing that looks like an answer at all.
    ("I am not sure how to proceed.", None),
]


@pytest.mark.parametrize(("completion", "expected"), LENIENT_VECTORS)
def test_lenient_ladder(completion: str, expected: str | None) -> None:
    assert lenient_answer(completion) == expected


def test_lenient_forgives_an_unclosed_think_block() -> None:
    """Strict scores an unclosed think block zero; lenient still reads the tail.

    This is the common shape of a completion that hit the token limit, and the
    gap between the two rungs on such a completion is exactly the signal the
    truncation rate is there to explain.
    """
    completion = "<think> after some work I get \\boxed{42}"
    assert score(completion, gold="42").strict is False
    assert score(completion, gold="42").lenient is True


def test_rungs_are_nested() -> None:
    """strict implies lenient implies permissive, on every vector we score."""
    completions = [c for c, _ in LENIENT_VECTORS] + [
        "\\boxed{42}",
        "<think>the answer must be 42</think> I cannot tell.",
        "no answer anywhere",
    ]
    for completion in completions:
        result = score(completion, gold="42")
        assert not result.strict or result.lenient
        assert not result.lenient or result.permissive


def test_permissive_reaches_inside_reasoning() -> None:
    """The upper bound: the answer was found, but never presented as one."""
    completion = "<think>I compute 42 here</think> I could not finish."
    result = score(completion, gold="42")
    assert (result.strict, result.lenient) == (False, False)
    assert result.permissive is True


def test_permissive_candidates_are_capped_and_ordered() -> None:
    """The cap keeps the last candidates, where an answer almost always is."""
    completion = " ".join(str(i) for i in range(200))
    candidates = permissive_candidates(completion)
    assert len(candidates) == PERMISSIVE_CANDIDATE_CAP
    assert candidates[-1] == "199"


def test_policy_selection_rejects_an_unknown_policy() -> None:
    """A typo in a config must fail loudly rather than pick a default rung."""
    result = score("\\boxed{42}", gold="42")
    assert result.at("strict") is True
    with pytest.raises(ValueError, match="unknown format policy"):
        result.at("strictt")


# --------------------------------------------------------------------------- #
# The hashed lens reads the answer region, not the reasoning.
# --------------------------------------------------------------------------- #

#: `(completion, extracted)`. Every row is a completion the reference's own
#: whole-string rule reads differently, which is the point of the departure.
HASHED_REGION_VECTORS = [
    # The contamination that named the bug. The marker was written *while*
    # thinking, and the pattern is line-greedy, so the whole-string rule extracts
    # `18 here</think>` -- a literal delimiter handed to the comparator.
    ("<think>so far #### 18 here</think>\nI conclude.", None),
    # The same completion once it also answers: the region's marker wins, and
    # the one inside the reasoning is not a candidate at all.
    ("<think>so far #### 18 here</think>\n#### 42", "42"),
    # Last marker in the region still wins, which is the reference's own rule.
    ("<think>nothing</think>\n#### 1\n#### 42", "42"),
    # No delimiters at all: `strip_think` is the identity and this is the
    # reference unchanged, which is every rollout it was written against.
    ("Working through it.\n#### 42", "42"),
    # An unclosed block is a format failure here exactly as it is for boxed --
    # there is no answer region, so there is nothing the lens may read.
    ("<think>I get #### 42", None),
]


@pytest.mark.parametrize(("completion", "expected"), HASHED_REGION_VECTORS)
def test_the_hashed_lens_reads_the_answer_region(completion: str, expected: str | None) -> None:
    assert strict_answer_hashed(completion) == expected


def test_the_hashed_lens_does_not_carry_a_delimiter_into_the_comparator() -> None:
    """The documented contamination, stated as the string it produced.

    Under the reference's whole-string rule this extracted `18 here</think>` and
    compared *that* against the gold. Nothing the lens recovers may contain a
    reasoning delimiter, because such a string was never offered as an answer.
    """
    answer = strict_answer_hashed("<think>so far #### 18 here</think>\n#### 42")
    assert answer is not None
    assert "</think>" not in answer


def test_the_hashed_fallback_falls_back_to_the_answer_region() -> None:
    """No marker anywhere: the reference falls back to text, and to which text.

    The lens keeps the reference's shape -- no marker, read the surrounding text
    -- and narrows what "surrounding" means to the region the model answered in.
    So a gold that appears only inside the abandoned reasoning is not accepted,
    and one written plainly in the answer region still is.
    """
    in_the_region = score("<think>maybe 7</think>\nIt is 42.", "42", answer_format="hashed")
    in_the_reasoning = score("<think>it is 42</think>\nI could not finish.", "42", answer_format="hashed")

    assert in_the_region.strict is True
    assert in_the_reasoning.strict is False
    # `permissive` keeps its published meaning and reads the reasoning, which is
    # the one lens that is *supposed* to.
    assert in_the_reasoning.permissive is True


def test_the_hashed_lens_refuses_an_unclosed_think_block() -> None:
    """The one shape where the two answer formats had drifted apart.

    Boxed has always scored an unclosed block zero. The hashed one fell back to
    the whole completion, so a completion that stopped mid-reasoning was graded
    on its scratch work under the name of its answer.
    """
    result = score("<think>the answer is 42 I think", "42", answer_format="hashed")
    assert (result.strict, result.format_ok) == (False, False)
    assert result.permissive is True


# --------------------------------------------------------------------------- #
# `format_ok` has three answers, and truncation decides which.
# --------------------------------------------------------------------------- #

#: `(completion, truncated, format_ok, not_evaluable)`, for both answer formats.
#: The matrix is the whole of the rule: only an unclosed block *and* a token cap
#: produce the third answer.
FORMAT_MATRIX = [
    # Think closed, an answer in the region: formatted, and evaluable.
    ("<think>work</think> the answer is {answer}", False, True, False),
    # Think closed, nothing in the region: a real formatting failure. The model
    # reached the region and put no answer in it.
    ("<think>the answer is {answer}</think> I could not finish.", False, False, False),
    # Think never closed, and the completion stopped by itself. Also a real
    # failure: there was budget left and the model did not present an answer.
    ("<think>I am still working, {answer} maybe", False, False, False),
    # Think never closed and the token cap ended it: no answer region ever
    # existed, so `format_ok` has nothing to be about.
    ("<think>I am still working, {answer} maybe", True, False, True),
    # Truncation alone does not do it. This completion reached its region and
    # formatted an answer there before running out.
    ("<think>work</think> the answer is {answer} and then some", True, True, False),
]


@pytest.mark.parametrize(("template", "truncated", "ok", "not_evaluable"), FORMAT_MATRIX)
@pytest.mark.parametrize(
    ("answer_format", "answer"), [("boxed", "\\boxed{42}"), ("hashed", "#### 42")]
)
def test_format_ok_is_not_evaluable_only_when_the_cap_ended_the_reasoning(
    template: str,
    truncated: bool,
    ok: bool,
    not_evaluable: bool,
    answer_format: str,
    answer: str,
) -> None:
    result = score(
        template.format(answer=answer),
        "42",
        answer_format=answer_format,
        truncated=truncated,
    )
    assert (result.format_ok, result.format_not_evaluable) == (ok, not_evaluable)


def test_truncation_reaches_format_ok_and_no_rung() -> None:
    """`truncated` may qualify the format question and never answer a lens.

    What a truncated completion *scores* is decided once, at the aggregation
    boundary, under the run's policy. If this flag ever moved a rung here, that
    decision would have a second home and the two could disagree.
    """
    completion = "<think>work</think> the answer is \\boxed{42}"
    finished = score(completion, "42")
    cut_off = score(completion, "42", truncated=True)
    assert (finished.strict, finished.lenient, finished.permissive) == (
        cut_off.strict,
        cut_off.lenient,
        cut_off.permissive,
    )
