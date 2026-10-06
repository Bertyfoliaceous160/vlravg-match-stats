"""Strict scoring must stay bit-identical to `vf.verify_boxed_math_answer`.

This is the test the whole comparability argument rests on. If it fails, an
offline number can no longer be compared with what the reinforcement learning run
scores in flight, and the fix is to restore `ladder.strict_answer`, never to
relax the test.

**The reference is the function, not the script.** Each maths environment ships a
`verify.py` beside its `taskset.py`, and it is tempting to read that file as the
specification. For `math500_v1`, `aime24_v1` and `aime25_v1` it is dead code:
`taskset.py` calls `vf.verify_boxed_math_answer` and never reads the file. Only
`arxivmath_v1` and `i3_math_v1` execute their own copy, as an isolated uv script.
The two implementations are not identical -- the live one strips the extracted
prediction and unwraps a boxed gold, the script does neither -- so pinning
against the script would guarantee agreement with code that never runs.

The reference is therefore imported from the pinned `verifiers` itself, which
means this test needs no checkout and never skips.
"""

from __future__ import annotations

import pytest
from verifiers.v1.utils.score import verify_boxed_math_answer

from limite_evals_core.ladder import extract_boxed, score, strict_answer

# (completion, expected extracted answer or None). Each line encodes one rule of
# the reference implementation.
EXTRACTION_VECTORS = [
    # A plain boxed answer.
    ("The answer is \\boxed{42}.", "42"),
    # Brace depth is matched, so nested braces survive.
    ("So \\boxed{\\frac{1}{2}} follows.", "\\frac{1}{2}"),
    ("\\boxed{\\text{a}{b}c}", "\\text{a}{b}c"),
    # The LAST boxed expression wins.
    ("First \\boxed{1}, corrected to \\boxed{2}.", "2"),
    # No boxed expression at all is a format failure, not a wrong answer.
    ("The answer is 42.", None),
    # An unbalanced boxed expression is refused rather than guessed at.
    ("\\boxed{42", None),
    # A closed think block is stripped; only the tail is searched.
    ("<think>\\boxed{99}</think> Therefore \\boxed{7}.", "7"),
    # A boxed answer inside reasoning does not count once the block is closed.
    ("<think>\\boxed{99}</think> no answer here", None),
    # An unclosed think block is a hard zero, even with a boxed answer present.
    ("<think> I get \\boxed{42}", None),
    # No think tags at all: the whole string is searched.
    ("no tags, \\boxed{5}", "5"),
    # Whitespace inside the braces is typesetting, and the reference strips it.
    ("\\boxed{ 42 }", "42"),
    ("\\boxed{\n42\n}", "42"),
    # Stripping to nothing is a format failure, not the answer "".
    ("\\boxed{   }", None),
]


@pytest.mark.parametrize(("completion", "expected"), EXTRACTION_VECTORS)
def test_strict_extraction_vectors(completion: str, expected: str | None) -> None:
    assert strict_answer(completion) == expected


def test_extract_boxed_matches_reference_edge_cases() -> None:
    assert extract_boxed("") == ""
    assert extract_boxed("\\boxed{}") == ""
    assert extract_boxed("\\boxed{{}}") == "{}"


def test_format_ok_tracks_extraction_not_correctness() -> None:
    """A boxed but wrong answer is a formatting success and a scoring failure.

    Keeping these two apart is what lets a report say whether a low number comes
    from the model not knowing the answer or from it not saying it properly.
    """
    result = score("\\boxed{41}", gold="42")
    assert result.format_ok is True
    assert result.strict is False


@pytest.mark.parametrize(
    ("completion", "gold"),
    [
        ("\\boxed{42}", "42"),
        ("\\boxed{41}", "42"),
        ("\\boxed{\\frac{1}{2}}", "0.5"),
        ("The answer is 42.", "42"),
        ("\\boxed{42", "42"),
        ("<think> I get \\boxed{42}", "42"),
        ("<think>\\boxed{99}</think> Therefore \\boxed{42}.", "42"),
        ("First \\boxed{1}, corrected to \\boxed{42}.", "42"),
        # The two cases the dead script would have got wrong.
        ("\\boxed{ 42 }", "42"),
        ("\\boxed{42}", "\\boxed{42}"),
    ],
)
def test_agrees_with_the_reference_verifier(completion: str, gold: str) -> None:
    expected = verify_boxed_math_answer(completion, gold) == 1.0
    assert score(completion, gold).strict is expected
