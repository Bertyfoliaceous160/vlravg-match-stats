"""The rendering and suite surface."""

from __future__ import annotations

import pytest

from limite_evals import profiles, suites, templates


MATH_TASKSETS = (
    "aime24",
    "aime25",
    "math500",
    "gsm8k",
    "aime26",
    "hmmt25",
    "olympiadbench",
    "beyondaime",
    "apex-shortlist",
    "hmmt26",
)

RENDERING_PROBE = "Find $n$ such that \\(n^2 = 4\\).\nShow work."
RENDERED_TASKSET_PROMPTS = {
    "aime24": suites.AIME_INSTRUCTION + RENDERING_PROBE,
    "aime25": suites.AIME_INSTRUCTION + RENDERING_PROBE,
    "math500": RENDERING_PROBE + suites.MATH500_INSTRUCTION,
    "gsm8k": RENDERING_PROBE + suites.MATH500_INSTRUCTION,
}


def test_math_extended_is_the_public_suite() -> None:
    assert tuple(suites.SUITES) == ("math-extended",)
    assert tuple(spec.name for spec in suites.resolve("math-extended")) == MATH_TASKSETS


def test_only_chat_and_base_kshot_profiles_are_public() -> None:
    assert profiles.resolve("chat").name == "chat"
    assert profiles.resolve("base-kshot").name == "base-kshot"
    for unknown in ("unknown", "custom"):
        with pytest.raises(ValueError, match="unknown rendering profile"):
            profiles.resolve(unknown)


def test_math_group_sizes_are_pinned() -> None:
    sizes = {spec.name: spec.group_size for spec in suites.resolve("math-extended")}
    assert sizes == {
        "aime24": 32,
        "aime25": 32,
        "math500": 4,
        "gsm8k": 1,
        "aime26": 32,
        "hmmt25": 32,
        "olympiadbench": 4,
        "beyondaime": 32,
        "apex-shortlist": 32,
        "hmmt26": 32,
    }


def test_math_profiles_select_distinct_renderings() -> None:
    assert templates.for_profile("chat", "math500") == templates.ARTIFACT
    assert templates.for_profile("base-kshot", "math500") == "base-kshot-math"
    assert templates.for_profile("base-kshot", "gsm8k") == "base-kshot-gsm8k"


def test_every_taskset_uses_answer_scoring() -> None:
    for suite in suites.SUITES:
        for spec in suites.resolve(suite):
            assert spec.answer_format is not None


def test_unknown_suite_and_taskset_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        suites.resolve("math-standard")
    with pytest.raises(ValueError, match="has no taskset"):
        suites.taskset("math-extended", "mmlu_pro")
