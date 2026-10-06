"""What ended a completion, and why the finish reason alone cannot say.

`finish_reason` reads `"stop"` in two entirely different situations: the model
emitted an end-of-sequence token, or one of the profile's stop strings matched
and the engine cut the completion there. A pretraining checkpoint has no learned
terminator and lives almost entirely in the second case; a fine-tuned one lives
in the first. That is the quantity a pretrain-versus-SFT comparison is actually
about, and with only `finish_reason` recorded the two are indistinguishable in
the record (F-11).

vLLM answers it directly, in a `stop_reason` field beside the finish reason,
carrying the matched stop string -- or the token id, when a stop token ended the
completion. These tests pin that the runner carries it out intact rather than
flattening it, at the boundary where the engine's answer arrives.
"""

from __future__ import annotations

import httpx

from limite_evals import profiles, templates
from limite_evals.runner import generate

STOP_STRING = "\nProblem:"
EOS_TOKEN_ID = 151643

KSHOT = templates.resolve(taskset="math500", template="base-kshot-math")


def _payload(*, finish_reason: str, stop_reason: object, present: bool = True) -> dict:
    choice: dict[str, object] = {
        "message": {"content": "six times seven is \\boxed{42}"},
        "finish_reason": finish_reason,
    }
    if present:
        choice["stop_reason"] = stop_reason
    return {"choices": [choice], "usage": {"completion_tokens": 17}}


async def _generate(payload: dict) -> tuple[str, str | None, str | int | None, int | None]:
    """One completion against an engine that returns exactly `payload`."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        return await generate(
            client,
            base_url="http://127.0.0.1:8000/v1",
            model="limite",
            prompt="What is six times seven?",
            profile=profiles.resolve("base-kshot"),
            rendering=KSHOT,
            max_tokens=64,
        )


async def test_the_matched_stop_string_is_surfaced() -> None:
    """Which string matched, not merely that one did."""
    _, finish_reason, stop_reason, _ = await _generate(
        _payload(finish_reason="stop", stop_reason=STOP_STRING)
    )
    assert finish_reason == "stop"
    assert stop_reason == STOP_STRING


async def test_a_stop_string_and_an_end_of_sequence_token_are_distinguishable() -> None:
    """The whole point: both report `finish_reason == "stop"`, and only the stop
    reason separates the checkpoint that invented the next problem from the one
    that decided it was finished."""
    _, by_string, string_reason, _ = await _generate(_payload(finish_reason="stop", stop_reason=STOP_STRING))
    _, by_eos, eos_reason, _ = await _generate(_payload(finish_reason="stop", stop_reason=None))
    assert by_string == by_eos == "stop"
    assert string_reason != eos_reason
    assert eos_reason is None


async def test_a_stop_token_id_is_carried_as_the_engine_reported_it() -> None:
    """vLLM sends an integer when a stop token id ended the completion. Coercing
    it to a string here would invent a stop string that was never configured."""
    _, _, stop_reason, _ = await _generate(_payload(finish_reason="stop", stop_reason=EOS_TOKEN_ID))
    assert stop_reason == EOS_TOKEN_ID


async def test_a_truncated_completion_has_no_stop_reason() -> None:
    """Nothing matched; the budget ran out. Decision 10 scores it wrong either
    way, and the stop reason must not suggest otherwise."""
    _, finish_reason, stop_reason, _ = await _generate(_payload(finish_reason="length", stop_reason=None))
    assert finish_reason == "length"
    assert stop_reason is None


async def test_an_engine_that_omits_the_field_is_not_an_error() -> None:
    """`stop_reason` is vLLM's extension, not part of the OpenAI schema. An
    engine that does not send it reports an unknown reason, not a failed run."""
    text, finish_reason, stop_reason, tokens = await _generate(
        _payload(finish_reason="stop", stop_reason=None, present=False)
    )
    assert stop_reason is None
    assert finish_reason == "stop"
    assert text.endswith("\\boxed{42}")
    assert tokens == 17
