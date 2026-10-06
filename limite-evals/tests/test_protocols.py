"""The protocol registry: what each number reproduces, and what it forgave.

Four properties are worth a test here, and one non-property is worth a test more
than any of them.

The properties: an unknown protocol id raises rather than defaulting, because a
typo in a config that silently picks a rung is how a metric changes without
anyone deciding to change it; the two relaxations contain their base, which is
what lets a report quote three columns side by side and read the differences;
`reference` still agrees with the rung it replaces, so the registry is a renaming
and not a re-baseline; and an unbindable `exact` yields absence rather than a
number.

The non-property: **`exact` and `reference` do not nest.** They name different
artefacts, and `test_the_anchors_do_not_nest` carries a completion that
`reference` scores correct and `permissive` -- which relaxes `exact` -- scores
wrong. That is not a bug to fix by forcing containment; it is the reason the two
anchors exist separately at all.

`exact` is bound here to a stand-in grader, clearly labelled as one. Identifying,
obtaining and binding each benchmark's real published grader is a separate piece
of work, and it is guarded by a parity test that executes that grader rather than
one that reimplements it.
"""

from __future__ import annotations

import pytest

from limite_evals_core.equivalence import equivalent
from limite_evals_core.ladder import score as ladder_score
from limite_evals_core.ladder import strict_answer
from limite_evals_core.protocols import (
    LADDER_FALLBACK_WAIVERS,
    PROTOCOLS,
    WAIVERS,
    Anchor,
    Grader,
    Protocol,
    ProtocolResult,
    TasksetProtocols,
    Unavailable,
    protocol,
    reference_grader,
    score,
)


def _stand_in_exact() -> Grader:
    """A stand-in for a benchmark's published grader, until the real one is bound.

    Boxed extraction plus the shared equivalence check -- deliberately *not* the
    same object as `reference`, so a test that passes only because the two
    anchors happen to be the same function cannot pass here.
    """

    def grade(completion: str, gold: str) -> bool:
        answer = strict_answer(completion)
        return answer is not None and equivalent(gold, answer)

    return Grader(name="stand-in published grader", grade=grade)


def _bindings(answer_format: str = "boxed", exact: Grader | Unavailable | None = None) -> TasksetProtocols:
    return TasksetProtocols(
        taskset="test",
        exact=exact if exact is not None else _stand_in_exact(),
        reference=reference_grader(answer_format),
    )


# Completions covering every waiver in the closed set, plus the shapes that
# should reach none of them.
COMPLETIONS = [
    "\\boxed{42}",
    "\\boxed{41}",
    "\\boxed{ 42 }",
    "So the answer is \\boxed{42",
    "Working through it, the answer is 42",
    "Final answer: 42",
    "Some reasoning.\nTherefore $x = 42$",
    "Some reasoning.\n42",
    "<think> after some work I get \\boxed{42}",
    "<think>I compute 42 here</think> I could not finish.",
    "<think>the answer must be 42</think> I cannot tell.",
    "#### 42",
    "I am not sure how to proceed.",
]


def test_the_registry_is_the_four_declared_protocols() -> None:
    assert list(PROTOCOLS) == ["exact", "reference", "lenient", "permissive"]
    assert PROTOCOLS["exact"].anchor == Anchor("published-grader")
    assert PROTOCOLS["reference"].anchor == Anchor("prime-rl-verifier")
    assert PROTOCOLS["lenient"].anchor == Anchor("relaxation-of", base="exact")
    assert PROTOCOLS["permissive"].anchor == Anchor("relaxation-of", base="exact")


def test_the_waiver_set_is_closed_and_the_relaxations_are_ordered() -> None:
    """Six named waivers, and `permissive` forgives strictly more than `lenient`."""
    assert PROTOCOLS["permissive"].waivers == WAIVERS
    assert PROTOCOLS["lenient"].waivers < PROTOCOLS["permissive"].waivers
    assert PROTOCOLS["permissive"].waivers - PROTOCOLS["lenient"].waivers == {"answer_anywhere"}
    assert PROTOCOLS["lenient"].describe() == (
        "answer_phrase, last_line_bare_number, last_line_inline_math, "
        "unclosed_think, unterminated_box"
    )


def test_an_unknown_protocol_id_raises() -> None:
    """A typo in a config must fail loudly rather than pick a protocol."""
    assert protocol("exact").id == "exact"
    with pytest.raises(ValueError, match="unknown protocol"):
        protocol("exactt")


def test_an_unknown_waiver_raises() -> None:
    with pytest.raises(ValueError, match="unknown waivers"):
        Protocol("made-up", Anchor("relaxation-of", base="exact"), frozenset({"almost_right"}))


def test_a_partially_waived_lenient_ladder_raises() -> None:
    """`ladder.lenient_answer` is one ordered function and cannot be split.

    A protocol that declared only `answer_phrase` would forgive the other three
    anyway, and its manifest entry would then understate what its number allowed.
    """
    with pytest.raises(ValueError, match="waives part of the lenient ladder"):
        Protocol("half", Anchor("relaxation-of", base="exact"), frozenset({"answer_phrase"}))


def test_an_anchored_protocol_cannot_waive_anything() -> None:
    """A waiver is a local variation, and an anchored protocol admits none."""
    with pytest.raises(ValueError, match="must not waive anything"):
        Protocol("exact-ish", Anchor("published-grader"), LADDER_FALLBACK_WAIVERS)


@pytest.mark.parametrize("answer_format", ["boxed", "hashed"])
@pytest.mark.parametrize("completion", COMPLETIONS)
def test_reference_agrees_with_the_rung_it_replaces(completion: str, answer_format: str) -> None:
    """The registry renames the strict rung; it must not re-score it.

    `reference` is composed from the ladder rather than reimplemented, and this
    pins that composition to `ladder.score(...).strict` on every vector, both
    formats. If it ever drifts, the offline number stops matching what the
    reinforcement learning run scores in flight.
    """
    binding = reference_grader(answer_format)
    expected = ladder_score(completion, gold="42", answer_format=answer_format).strict
    assert binding.grade(completion, "42") is expected


@pytest.mark.parametrize("completion", COMPLETIONS)
def test_the_relaxations_contain_their_base(completion: str) -> None:
    """exact implies lenient implies permissive, on every vector we score.

    This holds by construction -- each relaxation returns its base's verdict
    before consulting a single waived candidate, and `permissive`'s waivers are a
    superset of `lenient`'s -- but it is the property a three-column report is
    read through, so it is asserted rather than asserted about.
    """
    results = score(completion, gold="42", bindings=_bindings())
    assert not results["exact"].correct or results["lenient"].correct
    assert not results["lenient"].correct or results["permissive"].correct


def test_the_anchors_do_not_nest() -> None:
    """A historical reference can accept text that `permissive` rejects.

    The retired GSM8K reference handed the *whole completion* to `math-verify`
    when no `#### 42` marker was present, so it accepted a bare expression.
    Nothing relaxing `exact` reaches that: the waivers recover answer *strings*,
    and `1+x` on its own line yields the candidate `1`. Retaining this generic
    historical binding lets stored rows be interpreted without exposing a
    current GSM8K selector.

    So there is no containment between the two anchors in either direction, and
    the registry must not be read as a ladder. `exact` versus `reference` is a
    disagreement between two artefacts, not a strictness ordering.
    """
    results = score("1+x", gold="x+1", bindings=_bindings("hashed"))
    assert results["reference"].correct is True
    assert results["permissive"].correct is False
    assert results["exact"].correct is False


def test_an_unavailable_exact_is_absent_and_not_wrong() -> None:
    """An unobtainable published grader must never read as a model getting it wrong.

    AIME has no official grader, only de-facto ones. The `exact` column reads as
    absent, with the reason attached. `reference` is unaffected, because it names
    a different artefact.

    What the relaxations do in this case is D-108's subject, and
    `test_the_relaxations_fall_back_to_reference_when_exact_is_absent` pins it.
    """
    reason = "AIME publishes answers, not a grader"
    results = score("\\boxed{41}", gold="42", bindings=_bindings(exact=Unavailable(reason)))

    assert results["exact"].correct is None
    assert results["exact"].available is False
    assert results["exact"].unavailable_reason == reason
    assert results["exact"].anchor is None

    # A wrong answer under a bound anchor is False, and False is not None.
    assert results["reference"].correct is False
    assert results["reference"].available is True


def test_the_relaxations_fall_back_to_reference_when_exact_is_absent() -> None:
    """D-108: relax `exact` where it exists, relax `reference` where it does not.

    The alternative -- relaxing `exact` strictly -- blanked three columns instead
    of one and would have removed the strict-to-lenient gap from AIME24 and
    AIME25, which is half of `math-standard` and the pair on which post-training
    moves that gap most. The user chose the fallback over always relaxing
    `reference`, with the objection stated: one column then carries two meanings
    across rows of one table.

    So the base cannot be left implicit, and the rest of this test is that
    mitigation rather than the rule itself.
    """
    reason = "no published grader"
    completion = "Working through it, the answer is 42"

    bound = score(completion, gold="42", bindings=_bindings())
    absent = score(completion, gold="42", bindings=_bindings(exact=Unavailable(reason)))

    for relaxation in ("lenient", "permissive"):
        # With a published grader bound, the base is `exact`, unchanged by D-108.
        assert bound[relaxation].relaxes == "exact"
        # Without one, the relaxation is a number again -- and a different one.
        assert absent[relaxation].relaxes == "reference"
        assert absent[relaxation].available is True


@pytest.mark.parametrize("completion", COMPLETIONS)
def test_a_relaxed_result_carries_the_base_it_relaxed(completion: str) -> None:
    """A relaxed number and the thing it relaxed are one object, not two columns.

    Under the fallback, `lenient` means something different depending on which
    anchor a taskset had available, so a reader handed `lenient` alone has been
    handed an ambiguous figure. The base travels inside the result: it cannot be
    dropped by a rendering surface that forgot about it, and a `ProtocolResult`
    that names a base without carrying it does not construct.
    """
    for bindings in (_bindings(), _bindings(exact=Unavailable("no published grader"))):
        results = score(completion, gold="42", bindings=bindings)
        for relaxation in ("lenient", "permissive"):
            result = results[relaxation]
            assert result.base is not None
            assert result.base.protocol == result.relaxes
            assert result.base is results[result.relaxes]
            assert result.anchor is not None and result.anchor.startswith(f"{result.relaxes} + ")


def test_a_relaxed_result_cannot_be_built_without_its_base() -> None:
    """The mitigation is a constructor invariant, not a convention to remember."""
    with pytest.raises(ValueError, match="names a relaxation base without carrying it"):
        ProtocolResult("lenient", anchor="exact + ...", correct=True, relaxes="exact")

    base = ProtocolResult("reference", anchor="a grader", correct=True)
    with pytest.raises(ValueError, match="says it relaxes 'exact' but carries"):
        ProtocolResult("lenient", anchor="exact + ...", correct=True, relaxes="exact", base=base)


def test_every_result_names_what_produced_it() -> None:
    """The point of the registry: a number carries its own provenance."""
    results = score("\\boxed{42}", gold="42", bindings=_bindings())
    assert results["reference"].anchor == "verifiers.v1.utils.score.verify_boxed_math_answer"
    assert results["exact"].anchor == "stand-in published grader"
    assert results["permissive"].anchor.startswith("exact + answer_anywhere, answer_phrase")
    assert list(results) == list(PROTOCOLS)


def test_a_grader_that_hangs_is_bounded_and_scores_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anchored grader that never returns must cost its bound, not the run.

    A published grader runs arbitrary sympy and regex over a whole model
    completion, and one pathological completion can loop it forever -- observed
    on aime26, where the same seeded rollout wedged two GPU jobs in a row. The
    bound gives such a call `GRADER_TIMEOUT` seconds and then scores it wrong,
    the same conservative reading `equivalence.Comparison.timed_out` states for
    `math-verify`'s own bound. `reference` is untouched: the anchors are
    independent, and only the grader that hung pays.
    """
    from limite_evals_core import protocols as protocols_module

    def spin(completion: str, gold: str) -> bool:
        while True:
            pass

    monkeypatch.setattr(protocols_module, "GRADER_TIMEOUT", 1)
    results = score(
        "\\boxed{42}", gold="42", bindings=_bindings(exact=Grader(name="spinning grader", grade=spin))
    )
    assert results["exact"].correct is False
    assert results["reference"].correct is True
