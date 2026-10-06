"""Checkpoint chat and static base-kshot rendering profiles.

Template bytes and stop strings live in :mod:`limite_evals.templates`, where the
exact served rendering is hashed into the manifest. This module owns sampling
defaults and the committed exemplar sets used to rebuild and audit the static
base-kshot templates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

ProfileName = Literal["chat", "base-kshot"]

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"

#: Which exemplar file a taskset uses under `base-kshot`. GSM8K gets its own
#: eight-shot set; everything else shares the four-shot MATH set, including AIME,
#: which has no train split to draw from.
#:
#: The six `math-extended` tasksets join the shared set for the same reason
#: AIME already has: they are competition maths with no train split of their own
#: to draw exemplars from. Which bytes get served is no longer this mapping's
#: answer -- `templates.for_profile` reads `templates.KSHOT_TEMPLATES` for that,
#: and refuses before the engine starts. What a missing entry here costs is
#: decision 7's digest: `cli._exemplars_sha256` reaches for the set by name, so
#: A `base-kshot` checkpoint evaluated on a taskset absent from this mapping
#: raises `KeyError` while building the manifest, once the rollouts have already
#: been paid for.
#:
#: Nothing existing re-baselines: `template_sha256` is computed per taskset, so a
#: new key cannot perturb an old hash, and `KSHOT_STOP` -- the one genuinely
#: shared global, where a change *would* re-baseline every number in the
#: repository -- is untouched.
EXEMPLAR_SETS = {
    "gsm8k": "gsm8k",
    "math500": "math",
    "aime24": "math",
    "aime25": "math",
    "aime26": "math",
    "hmmt25": "math",
    "olympiadbench": "math",
    "beyondaime": "math",
    "apex-shortlist": "math",
    "hmmt26": "math",
}

#: Temperature is greater than zero because avg@k at temperature zero draws the
#: same sample k times, which makes the group size meaningless on AIME.
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95

#: Framing used by the committed base-kshot templates. Editing either constant
#: re-baselines every base-kshot number.
KSHOT_PROBLEM_PREFIX = "Problem: "
KSHOT_SOLUTION_CUE = "\nSolution:"

# Renders one user turn as the continuation of the k-shot block. Only the user
# role is emitted: a base model has no notion of a system or assistant turn.
_KSHOT_TAIL = (
    "{% for message in messages %}"
    + "{% if message['role'] == 'user' %}"
    + KSHOT_PROBLEM_PREFIX
    + "{{ message['content'] }}"
    + KSHOT_SOLUTION_CUE
    + "{% endif %}"
    + "{% endfor %}"
)


@dataclass(frozen=True)
class Profile:
    """A rendering profile, resolved against one taskset."""

    name: ProfileName
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    chat_template_kwargs: tuple[tuple[str, bool], ...] = ()

    def category_exemplars(self, taskset: str) -> None:
        """No taskset draws per-question exemplars."""
        del taskset
        return None

    def kshot_prompt(self, taskset: str, row: object) -> None:
        """Static base-kshot exemplars live entirely in the served template."""
        del taskset, row
        return None

    def prompt_identity(self, taskset: str) -> None:
        """The profiles add no prompt-side rendering payload."""
        del taskset
        return None

    def with_static_exemplars(self, taskset: str, problem: str) -> str:
        """The profiles never inject exemplars into the user prompt."""
        del taskset
        return problem


def resolve(name: str) -> Profile:
    if name not in ("chat", "base-kshot"):
        raise ValueError(f"unknown rendering profile: {name!r}")
    return Profile(name=name)  # type: ignore[arg-type]


@lru_cache(maxsize=None)
def load_exemplars(name: str) -> list[tuple[str, str]]:
    path = EXEMPLAR_DIR / f"{name}.json"
    data = json.loads(path.read_text())
    return [(item["problem"], item["solution"]) for item in data["exemplars"]]


def kshot_template_from_exemplars(taskset: str) -> str:
    """What `templates/base-kshot-<set>.jinja` has to contain, rebuilt from the files.

    **This is not what gets served.** The committed template is, and it is the
    only definition of the rendering. This rebuilds the same bytes from
    `exemplars/*.json` so that a test can assert the two still agree, which is
    the one guarantee that would otherwise be lost by moving the template into a
    committed file: editing an exemplar file would leave the served template
    stale while decision 7's `exemplars_sha256` moved underneath it, and nothing
    would say so.
    """
    return kshot_template_from_set(EXEMPLAR_SETS[taskset])


def kshot_template_from_set(name: str) -> str:
    """The same rebuild, keyed by exemplar set rather than by taskset.

    `EXEMPLAR_SETS` answers "which set is this taskset's default", and that is
    the only question the mapping above can answer. A set no taskset defaults to
    -- `math-v2`, reachable by naming its template -- still has a committed file
    that must be held in step with its exemplars, and the check is the same one.
    """
    return "{{ bos_token }}" + _raw_block(_exemplar_block(name)) + _KSHOT_TAIL


def _exemplar_block(name: str) -> str:
    """The k-shot preamble, in Limite-Pretrain's `Problem:`/`Solution:` format.

    Byte-identical to `export/eval_fixed.py::_prompt`, so a base number produced
    here is continuous with the ones that repository already reported.
    """
    parts = [f"Problem: {problem}\nSolution: {solution}\n" for problem, solution in load_exemplars(name)]
    return "\n".join([*parts, ""])


def _raw_block(text: str) -> str:
    """Wrap literal text so Jinja renders it unchanged.

    Exemplar solutions are LaTeX, and LaTeX is full of braces. `\\boxed{\\frac{1}{2}}`
    happens not to collide with Jinja's delimiters today, but a future exemplar
    containing `{{` or `{%` would silently become template syntax inside the
    served model's own renderer. Escaping is cheaper than that failure.
    """
    return "{% raw %}" + text + "{% endraw %}"
