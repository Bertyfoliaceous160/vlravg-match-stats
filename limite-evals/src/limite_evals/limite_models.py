"""Immutable Limite checkpoints admitted to the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from limite_evals.profiles import ProfileName


@dataclass(frozen=True)
class LimiteModel:
    """Hub identity and rendering contract for one released Limite checkpoint."""

    name: str
    repo_id: str
    revision: str
    profile: ProfileName
    stage: Literal["pretrain", "posttrain"]
    architecture: str = "LimiteForCausalLM"
    model_type: str = "limite"

    @property
    def identity(self) -> str:
        return f"{self.repo_id}@{self.revision}"


LIMITE_MODELS: tuple[LimiteModel, ...] = (
    LimiteModel(
        name="limite-1b-base",
        repo_id="paradigma-inc/limite-1b-base",
        revision="7e925d06b8f9a73ebdd0e571d8095eaaa4f954ff",
        profile="base-kshot",
        stage="pretrain",
    ),
    LimiteModel(
        name="limite-1b-base-soup",
        repo_id="paradigma-inc/limite-1b-base-soup",
        revision="7f6fa5440358a33828da612cfc3a0eb6e9368db8",
        profile="base-kshot",
        stage="pretrain",
    ),
    LimiteModel(
        name="limite-1b-violetto",
        repo_id="paradigma-inc/limite-1b-violetto",
        revision="b1f3d572ccacb6919f4d64c321b70ba034ddaef2",
        profile="chat",
        stage="posttrain",
    ),
)

BY_REPO = {model.repo_id.casefold(): model for model in LIMITE_MODELS}


def find(repo_id: str) -> LimiteModel | None:
    """Return the admitted Limite contract for ``repo_id``."""
    return BY_REPO.get(repo_id.casefold())
